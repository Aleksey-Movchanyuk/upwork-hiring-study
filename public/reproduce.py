"""Reproduce the main results of "Which Upwork Jobs Will Hire? Simple Rules vs Machine Learning on 11,500 Postings".

Run:   pip install -r requirements.txt  &&  python reproduce.py        (about 3 to 5 minutes on a laptop)
Needs: upwork_features_public.csv (same folder). No job text is included, so the word model (TF-IDF) is NOT reproduced here.
Writes: results/*.csv and results/*.png

The file is split into cells marked "# %%" (open it in VS Code or Jupyter as a notebook if you prefer).
Money columns in the public CSV are rounded to 3 significant figures, so numbers can differ from the article in the third decimal.
"""
# %% 1. Load the data and make the time split --------------------------------------------------------------
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
SEED = 42

df = pd.read_csv(HERE / "upwork_features_public.csv").sort_values("time_rank").reset_index(drop=True)
y = df.pop("hired").to_numpy()
df = df.drop(columns="time_rank")
FEATURES = list(df.columns)  # the 52 facts
CATEGORICAL = ["job_type_label", "duration_label", "engagement", "tier_text", "client_country", "country_group",
               "client_payment_verification_status"]
X = df.copy()
for c in CATEGORICAL:
    X[c] = X[c].astype("object").where(X[c].notna(), "unknown").astype(str)   # missing category -> "unknown"

# Oldest 70% train, next 15% tune (early stopping), latest 15% test. The model never sees the future.
n = len(df)
i_tr, i_va = int(n * 0.70), int(n * 0.85)
tr, va, te = slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)
yt = y[te]
print(f"{n:,} postings | train {i_tr:,} | tune {i_va - i_tr:,} | test {n - i_va:,} | hire rate {y.mean():.3f} (test {yt.mean():.3f})")

# %% 2. Helpers: bootstrap AUC, "apply to the top 30%" statistics -------------------------------------------
def bootstrap_auc(y_, s_, n_boot=500, seed=SEED):
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_), len(y_))
        if y_[i].min() != y_[i].max():
            b.append(roc_auc_score(y_[i], s_[i]))
    return roc_auc_score(y_, s_), *np.percentile(b, [2.5, 97.5])


def top_share_stats(y_, s_, share=0.30, connects=15, usd=0.10, seed=SEED):
    """Hire rate among the top `share` by score, and USD per recorded hire (15 Connects at $0.10 per application).
    Ties are broken at random, which matters for rules with few distinct scores."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(s_)), -np.asarray(s_, dtype=float)))
    sel = order[: int(round(len(s_) * share))]
    hires = y_[sel].sum()
    return {"top30_hire_rate": float(y_[sel].mean()), "usd_per_hire": float(len(sel) * connects * usd / hires) if hires else float("nan")}


# %% 3. Rules that need no training ----------------------------------------------------------------------------
scores, kind = {}, {}
spent = (X["client_total_spent"].fillna(0) > 0).astype(int)
hires = X["client_total_hires"].fillna(0)
months = X["duration_label"].map({"Less than 1 week": 0, "Less than 1 month": 1, "1 to 3 months": 2, "3 to 6 months": 3, "More than 6 months": 4}).fillna(99)

scores["Coin flip"], kind["Coin flip"] = np.zeros(len(yt)), "Rule"
scores["Rule: client has spent money"], kind["Rule: client has spent money"] = spent.to_numpy()[te], "Rule"
warning_signs = ((spent == 0).astype(int) + X["has_offplatform_contact"] + X["has_crypto_term"] + (X["desc_prior_copies"].fillna(0) > 0).astype(int))
scores["Rule: Upwork's warning signs (4 checks)"], kind["Rule: Upwork's warning signs (4 checks)"] = -warning_signs.to_numpy()[te], "Rule"

checklist = ((X["job_type_label"] == "Fixed Price").astype(int) + spent + (hires >= 10).astype(int) + (months <= 2).astype(int)
             + (1 - X["has_crypto_term"]) + (1 - X["agency_tone"]) + (X["description_word_count"] <= 400).astype(int))
scores["Rule: seven-question checklist"], kind["Rule: seven-question checklist"] = checklist.to_numpy()[te], "Rule"

# %% 4. A decision tree, three questions deep, on yes/no facts --------------------------------------------------
T = pd.DataFrame({
    "fixed_price": X["job_type_label"] == "Fixed Price", "has_spend": spent == 1, "hires_10plus": hires >= 10,
    "crypto": X["has_crypto_term"] == 1, "copied": X["desc_prior_copies"].fillna(0) > 0, "agency": X["agency_tone"] == 1,
    "new_build": X["new_build"] == 1, "long_text_400w": X["description_word_count"] > 400,
    "offplatform": X["has_offplatform_contact"] == 1, "fix_title": X["fix_title"] == 1, "long_term": X["long_term_wording"] == 1}).astype(int)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=SEED).fit(T.iloc[tr], y[tr])
scores["Decision tree, three questions deep"], kind["Decision tree, three questions deep"] = tree.predict_proba(T.iloc[te])[:, 1], "Model"

# %% 5. Logistic regression, gradient boosting, neural network --------------------------------------------------
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), [c for c in FEATURES if c not in CATEGORICAL]),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CATEGORICAL)])
logit = make_pipeline(prep, LogisticRegression(max_iter=3000, C=0.5)).fit(X.iloc[tr], y[tr])
scores["Logistic regression"], kind["Logistic regression"] = logit.predict_proba(X.iloc[te])[:, 1], "Model"

cat_idx = [FEATURES.index(c) for c in CATEGORICAL]


def fit_catboost(seed):
    m = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6, eval_metric="AUC", random_seed=seed, verbose=0,
                           early_stopping_rounds=100, allow_writing_files=False)
    m.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx), eval_set=Pool(X.iloc[va], y[va], cat_features=cat_idx))
    return m


cb = fit_catboost(SEED)
p_te = cb.predict_proba(Pool(X.iloc[te], cat_features=cat_idx))[:, 1]
scores["Gradient boosting (CatBoost)"], kind["Gradient boosting (CatBoost)"] = p_te, "Model"

mlp_scores = []
for seed in range(5):  # average five initialisations: a single network run is noisy
    mlp = make_pipeline(prep, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True, validation_fraction=0.15,
                                            n_iter_no_change=15, max_iter=400, random_state=seed)).fit(X.iloc[tr], y[tr])
    mlp_scores.append(mlp.predict_proba(X.iloc[te])[:, 1])
scores["Neural network (2 hidden layers)"], kind["Neural network (2 hidden layers)"] = np.mean(mlp_scores, axis=0), "Model"

# %% 6. The main table: ROC AUC, PR AUC, top-30% hire rate, cost per hire --------------------------------------
rows = []
for name, s in scores.items():
    s = np.asarray(s, dtype=float)
    if name == "Coin flip":
        a, lo, hi, pr, top = 0.5, 0.5, 0.5, yt.mean(), {"top30_hire_rate": yt.mean(), "usd_per_hire": 1.5 / yt.mean()}
    else:
        a, lo, hi = bootstrap_auc(yt, s)
        pr, top = average_precision_score(yt, s), top_share_stats(yt, s)
    rows.append({"method": name, "kind": kind[name], "roc_auc": a, "auc_lo": lo, "auc_hi": hi, "pr_auc": pr, **top})
table = pd.DataFrame(rows).sort_values("roc_auc")
table.to_csv(OUT / "main_table.csv", index=False)
print("\nTest period results (PR AUC of a coin flip = base rate %.3f)" % yt.mean())
print(table.round(3).to_string(index=False))

# %% 7. Paired bootstrap: is gradient boosting really better than the checklist? --------------------------------
rng = np.random.default_rng(SEED)
draws = [rng.integers(0, len(yt), len(yt)) for _ in range(2000)]


def paired(sa, sb):
    d = [roc_auc_score(yt[i], sa[i]) - roc_auc_score(yt[i], sb[i]) for i in draws if yt[i].min() != yt[i].max()]
    return np.mean(d), *np.percentile(d, [2.5, 97.5])


gb, ck = scores["Gradient boosting (CatBoost)"], scores["Rule: seven-question checklist"]
print("\nGradient boosting minus checklist (AUC): %+.3f (%.3f to %.3f)" % paired(gb, ck))
print("Gradient boosting minus neural network:  %+.3f (%.3f to %.3f)" % paired(gb, scores["Neural network (2 hidden layers)"]))

# %% 8. Precision and recall if you apply to the top 30% ----------------------------------------------------------
order = np.argsort(-p_te)
top = order[: int(0.30 * len(order))]
precision, recall = yt[top].mean(), yt[top].sum() / yt.sum()
print(f"\nApply to the top 30% by gradient boosting: precision {precision:.3f}, recall {recall:.3f} (base rate {yt.mean():.3f})")

# %% 9. Calibration: do the probabilities mean what they say? -------------------------------------------------------
p_va = cb.predict_proba(Pool(X.iloc[va], cat_features=cat_idx))[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y[va])          # fitted on the tuning slice, never on test
p_cal = iso.predict(p_te)
dec = pd.qcut(p_te, 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"decile": dec, "predicted": p_te, "calibrated": p_cal, "actual": yt}).groupby("decile").mean().reset_index()
cal.to_csv(OUT / "calibration_deciles.csv", index=False)
print("\nBrier score: raw %.4f | after isotonic calibration %.4f | always-predict-base-rate %.4f" % (
    brier_score_loss(yt, p_te), brier_score_loss(yt, p_cal), brier_score_loss(yt, np.full(len(yt), y[tr].mean()))))
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 0.7], [0, 0.7], "--", color="grey")
ax.plot(cal["predicted"], cal["actual"], "o-", label="raw model")
ax.plot(cal["calibrated"], cal["actual"], "s-", label="after isotonic calibration")
ax.set_xlabel("Mean predicted probability (10 groups)"); ax.set_ylabel("Actual hire rate"); ax.legend(); ax.set_title("Calibration on the test period")
fig.tight_layout(); fig.savefig(OUT / "calibration.png", dpi=130); plt.close(fig)

# %% 10. Run-to-run noise: five seeds of the same boosting model ----------------------------------------------------
aucs = []
for seed in range(42, 47):
    m = fit_catboost(seed)
    aucs.append(roc_auc_score(yt, m.predict_proba(Pool(X.iloc[te], cat_features=cat_idx))[:, 1]))
print("\nGradient boosting, five seeds: AUC mean %.3f, sd %.4f, min %.3f, max %.3f" % (np.mean(aucs), np.std(aucs, ddof=1), min(aucs), max(aucs)))

# %% 11. Which facts push a single posting up or down (SHAP values from CatBoost itself) ---------------------------
shap = cb.get_feature_importance(Pool(X.iloc[te], yt, cat_features=cat_idx), type="ShapValues")[:, :-1]
imp = pd.Series(np.abs(shap).mean(axis=0), index=FEATURES).sort_values(ascending=False)
imp.to_csv(OUT / "shap_mean_abs.csv", header=["mean_abs_shap"])
top15 = imp.head(15)[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top15.index, top15.values, color="#3366CC")
ax.set_xlabel("Mean |SHAP value| (log-odds) on the test period"); ax.set_title("What the gradient boosting model leans on")
fig.tight_layout(); fig.savefig(OUT / "shap_top15.png", dpi=130); plt.close(fig)
print("\nTop 10 facts by mean |SHAP|:\n", imp.head(10).round(3).to_string())

# %% 12. Error analysis: which hires does the model miss? --------------------------------------------------------------
Dt = X.iloc[te].reset_index(drop=True)
rank = pd.Series(p_te).rank(ascending=False, pct=True).to_numpy()
hit = (yt == 1) & (rank <= 0.30)
missed = (yt == 1) & (rank > 0.50)
profile = pd.DataFrame({
    "share_fixed_price": [(Dt["job_type_label"] == "Fixed Price")[hit].mean(), (Dt["job_type_label"] == "Fixed Price")[missed].mean()],
    "share_client_has_spent": [(Dt["client_total_spent"].fillna(0) > 0)[hit].mean(), (Dt["client_total_spent"].fillna(0) > 0)[missed].mean()],
    "median_words": [Dt["description_word_count"][hit].median(), Dt["description_word_count"][missed].median()],
    "median_client_hires": [Dt["client_total_hires"][hit].median(), Dt["client_total_hires"][missed].median()],
    "n": [int(hit.sum()), int(missed.sum())]}, index=["hires in the top 30%", "hires ranked in the bottom half"])
profile.to_csv(OUT / "error_analysis.csv")
print("\nHires the model catches against hires it ranks in the bottom half:\n", profile.round(2).to_string())
print(f"\nShare of all test hires ranked in the bottom half: {missed.sum() / yt.sum():.1%}")
print("Done. Files are in", OUT)
