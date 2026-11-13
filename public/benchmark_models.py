"""Benchmark about 60 models and combinations against the seven-question checklist, on the public CSV.

Run:   pip install -r requirements.txt lightgbm xgboost   &&   python benchmark_models.py        (10 to 20 minutes)

How the comparison is kept fair
  * Every model is trained on the oldest 70% of postings only.
  * Models are RANKED on the next 15% (the "tuning" slice, never used for training or early stopping).
  * The latest 15% (test) is used only to report the scores of the models already picked. Test AUCs for all models are
    shown too, but the top 10 is chosen on the tuning slice, so test scores are not inflated by picking the winner on them.
  * Early stopping for boosting uses the last 15% of the TRAINING slice, not the tuning slice.
Writes results/benchmark_all_models.csv, results/benchmark_top10.csv and results/benchmark_top10.png.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
SEED = 42

# ------------------------------------------------------------------ data and split
df = pd.read_csv(HERE / "upwork_features_public.csv").sort_values("time_rank").reset_index(drop=True)
y = df.pop("hired").to_numpy()
df = df.drop(columns="time_rank")
CAT = ["job_type_label", "duration_label", "engagement", "tier_text", "client_country", "country_group", "client_payment_verification_status"]
X = df.copy()
for c in CAT:
    X[c] = X[c].astype("object").where(X[c].notna(), "unknown").astype(str)
n = len(X)
i_tr, i_va = int(n * 0.70), int(n * 0.85)
tr, va, te = slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)
i_es = int(i_tr * 0.85)                       # inner split of the training slice, for early stopping only
fit_, es_ = slice(0, i_es), slice(i_es, i_tr)
ytr, yva, yte = y[tr], y[va], y[te]
print(f"train {i_tr}, tune {i_va - i_tr}, test {n - i_va}")

# ------------------------------------------------------------------ the checklist (the bar to beat)
spent = (X["client_total_spent"].fillna(0) > 0).astype(int)
hires = X["client_total_hires"].fillna(0)
months = X["duration_label"].map({"Less than 1 week": 0, "Less than 1 month": 1, "1 to 3 months": 2, "3 to 6 months": 3, "More than 6 months": 4}).fillna(99)
bits = pd.DataFrame({"fixed": (X["job_type_label"] == "Fixed Price").astype(int), "spent": spent, "hires10": (hires >= 10).astype(int),
                     "short": (months <= 2).astype(int), "no_crypto": 1 - X["has_crypto_term"], "no_agency": 1 - X["has_agency"] if "has_agency" in X else 1 - X["agency_tone"],
                     "words400": (X["description_word_count"] <= 400).astype(int)})
points = bits.sum(axis=1).to_numpy().astype(float)

# ------------------------------------------------------------------ matrices
LOGC = ["budget_fixed", "budget_hourly_min", "budget_hourly_max", "client_total_spent", "client_total_hires", "description_char_count",
        "description_word_count", "title_char_count", "desc_prior_copies", "title_prior_copies", "tags_count", "skills_count",
        "em_dash_count", "markdown_marks", "emoji_count", "line_count", "bullet_count", "exclamation_count", "age_at_capture_h"]
XL = X.copy()
for c in LOGC:
    XL[c] = np.log1p(XL[c].clip(lower=0))
NUM = [c for c in X.columns if c not in CAT]
prep = ColumnTransformer([("num", __import__("sklearn.pipeline", fromlist=["make_pipeline"]).make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM),
                          ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False), CAT)])
prep.fit(XL.iloc[tr])
Z = prep.transform(XL).astype(np.float32)
Ztr, Zva, Zte = Z[tr], Z[va], Z[te]
Zfit, Zes = Z[fit_], Z[es_]
cat_idx = [list(X.columns).index(c) for c in CAT]


def score(m, A):
    return m.predict_proba(A)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(A)


def sk(model):
    """sklearn-style model on the one-hot matrix, trained on the training slice."""
    def run():
        model.fit(Ztr, ytr)
        return score(model, Zva), score(model, Zte)
    return run


def mlp(layers, alpha, seeds=3):
    def run():
        sv, st = [], []
        for s in range(seeds):
            m = MLPClassifier(hidden_layer_sizes=layers, alpha=alpha, early_stopping=True, validation_fraction=0.15,
                              n_iter_no_change=15, max_iter=400, random_state=s).fit(Ztr, ytr)
            sv.append(m.predict_proba(Zva)[:, 1]); st.append(m.predict_proba(Zte)[:, 1])
        return np.mean(sv, 0), np.mean(st, 0)
    return run


def catboost(depth=6, lr=0.04, l2=3, balanced=False, cols=None, extra=None):
    def run():
        cols_ = cols or list(X.columns)
        Xc = X[cols_] if extra is None else pd.concat([X[cols_], extra], axis=1)
        ci = [list(Xc.columns).index(c) for c in CAT if c in Xc.columns]
        m = CatBoostClassifier(iterations=1500, learning_rate=lr, depth=depth, l2_leaf_reg=l2, eval_metric="AUC", random_seed=SEED, verbose=0,
                               early_stopping_rounds=100, allow_writing_files=False, thread_count=1,
                               auto_class_weights="Balanced" if balanced else None)
        m.fit(Xc.iloc[fit_], y[fit_], cat_features=ci, eval_set=(Xc.iloc[es_], y[es_]))
        return m.predict_proba(Xc.iloc[va])[:, 1], m.predict_proba(Xc.iloc[te])[:, 1]
    return run


def lgbm(leaves, lr=0.03, **kw):
    def run():
        import lightgbm as lgb
        m = LGBMClassifier(n_estimators=2000, learning_rate=lr, num_leaves=leaves, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=1, **kw)
        m.fit(Zfit, y[fit_], eval_set=[(Zes, y[es_])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        return score(m, Zva), score(m, Zte)
    return run


def xgb(depth, lr=0.03):
    def run():
        m = XGBClassifier(n_estimators=2000, learning_rate=lr, max_depth=depth, subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                          early_stopping_rounds=100, random_state=SEED, n_jobs=1, verbosity=0)
        m.fit(Zfit, y[fit_], eval_set=[(Zes, y[es_])], verbose=False)
        return score(m, Zva), score(m, Zte)
    return run


def tiny(cols_idx, model):
    """A model that sees only some columns of the matrix (used for the checklist bits)."""
    def run():
        model.fit(bits.iloc[tr], ytr)
        return score(model, bits.iloc[va]), score(model, bits.iloc[te])
    return run


C = []  # (family, name, runner)
add = lambda fam, name, run: C.append((fam, name, run))
# linear and simple probabilistic
for c in (0.01, 0.05, 0.1, 0.5, 2):
    add("Linear", f"Logistic regression L2, C={c}", sk(LogisticRegression(C=c, max_iter=4000)))
for c in (0.05, 0.2):
    add("Linear", f"Logistic regression L1, C={c}", sk(LogisticRegression(C=c, penalty="l1", solver="liblinear", max_iter=4000)))
add("Linear", "Elastic net logistic, C=0.2", sk(LogisticRegression(C=0.2, penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=4000)))
add("Linear", "Ridge classifier", sk(RidgeClassifier(alpha=50)))
for c in (0.01, 0.1):
    add("Linear", f"Linear SVM, C={c}", sk(LinearSVC(C=c, max_iter=20000)))
add("Linear", "SGD (modified Huber)", sk(SGDClassifier(loss="modified_huber", alpha=1e-3, random_state=SEED)))
add("Linear", "Linear discriminant analysis", sk(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")))
add("Linear", "Gaussian naive Bayes", sk(GaussianNB()))
# neighbours and kernels
for k in (25, 75, 200):
    add("Distance", f"k nearest neighbours, k={k}", sk(KNeighborsClassifier(n_neighbors=k, weights="distance")))
for c in (0.5, 2):
    add("Kernel", f"SVM, RBF kernel, C={c}", sk(SVC(C=c, gamma="scale", random_state=SEED)))
# trees and forests
for d in (3, 5, 8):
    add("Trees", f"Decision tree, depth {d}", sk(DecisionTreeClassifier(max_depth=d, min_samples_leaf=50, random_state=SEED)))
for leaf in (5, 20):
    add("Trees", f"Random forest, leaf>={leaf}", sk(RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, n_jobs=1, random_state=SEED)))
    add("Trees", f"Extra trees, leaf>={leaf}", sk(ExtraTreesClassifier(n_estimators=500, min_samples_leaf=leaf, n_jobs=1, random_state=SEED)))
add("Trees", "Random forest, balanced classes", sk(RandomForestClassifier(n_estimators=500, min_samples_leaf=10, class_weight="balanced_subsample", n_jobs=1, random_state=SEED)))
add("Boosting", "AdaBoost, stumps", sk(AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=SEED)))
add("Boosting", "AdaBoost, depth 2", sk(AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=300, learning_rate=0.3, random_state=SEED)))
add("Boosting", "sklearn gradient boosting, depth 3", sk(GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED)))
add("Boosting", "sklearn gradient boosting, depth 5", sk(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=SEED)))
for leaves in (15, 31):
    add("Boosting", f"HistGradientBoosting, {leaves} leaves", sk(HistGradientBoostingClassifier(max_leaf_nodes=leaves, learning_rate=0.05, early_stopping=True, random_state=SEED)))
for d in (4, 6, 8):
    add("Boosting", f"CatBoost, depth {d}", catboost(depth=d))
add("Boosting", "CatBoost, learning rate 0.1", catboost(lr=0.1))
add("Boosting", "CatBoost, strong regularisation (l2=20)", catboost(l2=20))
add("Boosting", "CatBoost, balanced classes", catboost(balanced=True))
for leaves in (7, 15, 31):
    add("Boosting", f"LightGBM, {leaves} leaves", lgbm(leaves))
add("Boosting", "LightGBM, 15 leaves, balanced", lgbm(15, class_weight="balanced"))
for d in (3, 6):
    add("Boosting", f"XGBoost, depth {d}", xgb(d))
# neural networks (scikit-learn MLPs, average of 3 seeds each; PyTorch is not available for this Python)
for layers, alpha in [((32,), 1e-2), ((64, 32), 1e-3), ((64, 32), 1e-2), ((64, 32), 1e-1), ((128, 64, 32), 1e-2), ((256, 128), 1e-2),
                      ((256, 128, 64, 32, 16), 1e-2)]:
    add("Neural net", f"MLP {layers}, alpha={alpha}", mlp(layers, alpha))
# checklist-based hybrids and small-feature models
add("Checklist+", "Logistic regression on the 7 checklist answers", tiny(None, LogisticRegression(C=1.0, max_iter=1000)))
add("Checklist+", "Shallow forest on the 7 checklist answers", tiny(None, RandomForestClassifier(n_estimators=300, min_samples_leaf=30, random_state=SEED)))
add("Checklist+", "CatBoost + checklist points as an extra fact", catboost(extra=pd.DataFrame({"checklist_points": points})))
KEY = ["job_type_label", "budget_fixed", "budget_hourly_min", "budget_hourly_max", "has_budget_fixed", "duration_label", "engagement",
       "client_total_spent", "client_total_hires", "description_word_count", "has_crypto_term", "agency_tone"]
add("Checklist+", "CatBoost on 12 key facts only", catboost(cols=KEY))

print(len(C), "base models; training in parallel ...", flush=True)
res = Parallel(n_jobs=8, verbose=5)(delayed(lambda f=f: f())() for _, _, f in C)
S = {name: (fam, sv, st) for (fam, name, _), (sv, st) in zip(C, res)}

# ------------------------------------------------------------------ combinations (members picked on the tuning slice only)
def auc_v(name):
    return roc_auc_score(yva, S[name][1])


def rank01(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def blend(names):
    return np.mean([rank01(S[k][1]) for k in names], 0), np.mean([rank01(S[k][2]) for k in names], 0)


ranked = sorted(S, key=auc_v, reverse=True)
best_in = lambda fam: max((k for k in S if S[k][0] == fam), key=auc_v)
combos = {
    "Blend: CatBoost depth 6 + logistic regression C=0.1": ["CatBoost, depth 6", "Logistic regression L2, C=0.1"],
    "Blend: best 3 on the tuning slice": ranked[:3],
    "Blend: best 5 on the tuning slice": ranked[:5],
    "Blend: best 10 on the tuning slice": ranked[:10],
    "Blend: best boosting + best neural net + best linear": [best_in("Boosting"), best_in("Neural net"), best_in("Linear")],
    "Blend: all boosting models": [k for k in S if S[k][0] == "Boosting"],
}
for name, members in combos.items():
    sv, st = blend(members)
    S[name] = ("Combination", sv, st)
# stacking: logistic regression on the base scores, meta-model fitted on the tuning slice (validation = 5-fold CV inside that slice)
for k in (5, 10):
    members = ranked[:k]
    Mv = np.column_stack([rank01(S[m][1]) for m in members]); Mt = np.column_stack([rank01(S[m][2]) for m in members])
    meta = LogisticRegression(C=1.0, max_iter=1000)
    sv = cross_val_predict(meta, Mv, yva, cv=KFold(5, shuffle=False), method="predict_proba")[:, 1]
    st = meta.fit(Mv, yva).predict_proba(Mt)[:, 1]
    S[f"Stack: logistic meta-model on best {k}"] = ("Combination", sv, st)

# ------------------------------------------------------------------ scoring table
rng = np.random.default_rng(SEED)
draws = [rng.integers(0, len(yte), len(yte)) for _ in range(1000)]
chk_te = points[te]
chk_auc_te, chk_auc_va = roc_auc_score(yte, chk_te), roc_auc_score(yva, points[va])


def boot(s):
    a = [roc_auc_score(yte[i], s[i]) for i in draws if yte[i].min() != yte[i].max()]
    return np.percentile(a, [2.5, 97.5])


def paired_vs_checklist(s):
    d = [roc_auc_score(yte[i], s[i]) - roc_auc_score(yte[i], chk_te[i]) for i in draws if yte[i].min() != yte[i].max()]
    return np.mean(d), *np.percentile(d, [2.5, 97.5])


def top30(s):
    k = int(round(len(s) * 0.30)); o = np.lexsort((np.random.default_rng(SEED).random(len(s)), -s)); return yte[o[:k]].mean()


rows = []
for name, (fam, sv, st) in S.items():
    lo, hi = boot(st); d, dlo, dhi = paired_vs_checklist(st)
    rows.append({"model": name, "family": fam, "val_auc": roc_auc_score(yva, sv), "test_auc": roc_auc_score(yte, st), "test_lo": lo, "test_hi": hi,
                 "test_pr_auc": average_precision_score(yte, st), "test_top30_hire_rate": top30(st),
                 "test_diff_vs_checklist": d, "diff_lo": dlo, "diff_hi": dhi})
tab = pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)
tab.insert(0, "val_rank", np.arange(1, len(tab) + 1))
tab.to_csv(OUT / "benchmark_all_models.csv", index=False)
top = tab.head(10)
top.to_csv(OUT / "benchmark_top10.csv", index=False)
pd.set_option("display.width", 220)
print(f"\nChecklist: tuning-slice AUC {chk_auc_va:.3f}, test AUC {chk_auc_te:.3f}")
print(f"{len(tab)} models and combinations. Better than the checklist on the tuning slice: {(tab.val_auc > chk_auc_va).sum()}; on test: {(tab.test_auc > chk_auc_te).sum()}; "
      f"clearly better on test (95% range of the difference above 0): {(tab.diff_lo > 0).sum()}")
print("\nTop 10 chosen on the tuning slice:")
print(top[["val_rank", "model", "val_auc", "test_auc", "test_lo", "test_hi", "test_top30_hire_rate", "test_diff_vs_checklist", "diff_lo", "diff_hi"]].round(3).to_string(index=False))
print("\nBest family on tuning slice:\n", tab.groupby("family").val_auc.max().sort_values(ascending=False).round(3).to_string())
print("\nTest AUC of the 10 models best on TEST (optimistic, shown only to size the winner's curse): mean %.3f vs %.3f for the validation-chosen top 10"
      % (tab.test_auc.nlargest(10).mean(), top.test_auc.mean()))

# ------------------------------------------------------------------ figure
pd.DataFrame([{"checklist": "seven-question checklist", "val_auc": chk_auc_va, "test_auc": chk_auc_te}]).to_csv(OUT / "benchmark_checklist.csv", index=False)
from replot_benchmark import make_figure
make_figure()
print("done")
