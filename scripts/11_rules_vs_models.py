"""Step 11: simple logical rules versus machine-learning models (including a small neural network).

Same time split as the main model; every number is on the latest 15% of postings.
  Rules (no training): spend-history rule; Upwork's own warning signs counted; a hand-made checklist.
    CAUTION: the hand-made checklist was chosen after looking at the whole dataset, so its test score is optimistic.
  Learned rules: shallow decision trees (depth 2 and 3) trained on the earliest 70%.
  Models: logistic regression, CatBoost (gradient boosting), a small neural network (MLP).
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from plotnine import aes, coord_flip, geom_hline, geom_pointrange, ggplot, labs, scale_color_manual, scale_y_continuous
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from _common import BLUE, DATA, ORANGE, base_theme, save
from _features import ALL, BOOL, CATEGORICAL, SEED, bootstrap_auc, model_frame, top_share_stats

df, X, y, (tr, va, te) = model_frame()
yt = y[te]
D = df.iloc[te]
scores, kinds = {}, {}


def add(name, kind, s):
    scores[name], kinds[name] = np.asarray(s, dtype=float), kind


has_spend = (df["client_total_spent"].fillna(0) > 0).astype(int).to_numpy()
add("Coin flip", "Rule", np.zeros(len(yt)) + 0.0)
add("Rule: client has spent money", "Rule", has_spend[te])

# Upwork's own warning signs that my data can test (guide, May 2026): new client, off-platform contact,
# crypto, copied post. Score = number of warning signs absent.
warn = (
    (df["client_total_spent"].fillna(0) == 0).astype(int) + df["has_offplatform_contact"].astype(int)
    + df["has_crypto_term"].astype(int) + (df["desc_prior_copies"].fillna(0) > 0).astype(int)
).to_numpy()
add("Rule: Upwork's warning signs (4 checks)", "Rule", -warn[te])

hand = (
    (df["job_type_label"] == "Fixed Price").astype(int) + has_spend
    + (df["client_total_hires"].fillna(0) >= 10).astype(int)
    + (~df["has_crypto_term"]).astype(int) + (df["desc_prior_copies"].fillna(0) == 0).astype(int)
    + (~df["agency_tone"]).astype(int) + (~df["new_build"]).astype(int)
).to_numpy()
add("Rule: my 7-point checklist (optimistic)", "Rule", hand[te])

# Learned logical rules: shallow trees on yes/no facts
tree_cols = {
    "fixed_price": (df["job_type_label"] == "Fixed Price"), "has_spend": has_spend == 1,
    "hires_10plus": df["client_total_hires"].fillna(0) >= 10, "crypto": df["has_crypto_term"],
    "copied": df["desc_prior_copies"].fillna(0) > 0, "agency": df["agency_tone"], "new_build": df["new_build"],
    "long_text_400w": df["description_word_count"] > 400, "offplatform": df["has_offplatform_contact"],
    "fix_title": df["fix_title"], "long_term": df["long_term_wording"],
}
T = pd.DataFrame(tree_cols).astype(int)
for depth in (2, 3):
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=100, random_state=SEED).fit(T.iloc[tr], y[tr])
    add(f"Learned rule: decision tree, depth {depth}", "Rule", dt.predict_proba(T.iloc[te])[:, 1])
    if depth == 3:
        print("Depth-3 tree (learned on the earliest 70%):\n", export_text(dt, feature_names=list(T.columns)))
        tr_leaf, te_leaf = dt.apply(T.iloc[tr]), dt.apply(T.iloc[te])
        tt = dt.tree_
        paths = {}

        def walk(node, path):
            if tt.children_left[node] == -1:
                paths[node] = path
                return
            nm = T.columns[tt.feature[node]]
            walk(tt.children_left[node], path + [f"NOT {nm}"])
            walk(tt.children_right[node], path + [nm])

        walk(0, [])
        lines = []
        for leaf, path in paths.items():
            a, b = tr_leaf == leaf, te_leaf == leaf
            if a.sum():
                lines.append((y[tr][a].mean(), f"IF {' AND '.join(path)}\n   THEN hire rate {y[tr][a].mean():.1%} in training (n={int(a.sum()):,}), "
                              f"{(y[te][b].mean() if b.sum() else float('nan')):.1%} in test (n={int(b.sum()):,})"))
        (DATA / "decision_tree_rules.txt").write_text("\n\n".join(t for _, t in sorted(lines, reverse=True)) + "\n", encoding="utf-8")

# Models
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), [c for c in ALL if c not in CATEGORICAL]),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CATEGORICAL),
])
logit = make_pipeline(prep, LogisticRegression(max_iter=3000, C=0.5)).fit(X.iloc[tr], y[tr])
add("Logistic regression", "Model", logit.predict_proba(X.iloc[te])[:, 1])

cat_idx = [ALL.index(c) for c in ALL if c in CATEGORICAL]
cb = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6, eval_metric="AUC", random_seed=SEED,
                        verbose=0, early_stopping_rounds=100, allow_writing_files=False)
cb.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx), eval_set=Pool(X.iloc[va], y[va], cat_features=cat_idx))
add("CatBoost (gradient boosting)", "Model", cb.predict_proba(Pool(X.iloc[te], yt, cat_features=cat_idx))[:, 1])

mlp_scores = []
for seed in range(5):  # average of five initialisations, because a single network run is noisy
    mlp = make_pipeline(prep, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                                            validation_fraction=0.15, n_iter_no_change=15, max_iter=400, random_state=seed))
    mlp.fit(X.iloc[tr], y[tr])
    mlp_scores.append(mlp.predict_proba(X.iloc[te])[:, 1])
add("Neural network (2 hidden layers)", "Model", np.mean(mlp_scores, axis=0))

pd.DataFrame({"y": yt, **scores}).to_parquet(DATA / "ml_test_scores.parquet", index=False)
rows = []
for name, s in scores.items():
    if name == "Coin flip":
        a, lo, hi, pr = 0.5, 0.5, 0.5, yt.mean()
        top = {"top_hire_rate": yt.mean(), "hires_kept": 0.30, "usd_per_hire": 1.5 / yt.mean()}
    else:
        a, lo, hi = bootstrap_auc(yt, s)
        pr = average_precision_score(yt, s)
        top = top_share_stats(yt, s)
    rows.append({"method": name, "kind": kinds[name], "roc_auc": a, "lo": lo, "hi": hi, "pr_auc": pr, **top})
res = pd.DataFrame(rows)
res.to_csv(DATA / "ml_rules_vs_models.csv", index=False)
print(res.round(3).to_string(index=False))

plot = res.sort_values("roc_auc").copy()
plot["method"] = pd.Categorical(plot["method"], categories=plot["method"], ordered=True)
p = (
    ggplot(plot, aes("method", "roc_auc", color="kind"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"))
    + geom_hline(yintercept=0.5, linetype="dashed", color="#444444")
    + coord_flip()
    + scale_color_manual({"Rule": ORANGE, "Model": BLUE})
    + scale_y_continuous(limits=(0.45, 0.8))
    + labs(title="Simple rules versus machine-learning models",
           subtitle="Test ROC AUC (latest 15% of postings), 95% bootstrap range; 0.5 = coin flip", x="", y="ROC AUC", color="")
    + base_theme()
)
save(p, "28_rules_vs_models", 9.5, 5.5)
