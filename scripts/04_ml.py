"""Step 4: Practical machine learning.

Question: before spending Connects, can we rank new postings by their chance
of ending in a hire?

Design choices (each one is a leakage or honesty guard):
- Time-based split by capture time (train 70% / validation 15% / test 15%),
  never a random split, because posting patterns drift.
- Only features known when the posting is first seen (duplicate counts use only
  copies captured BEFORE the posting). Excluded on purpose:
  applicant counts, interviews, invites (they exist only after posting),
  proposals tier, is_applied, and the seen_in_* flags.
- Three baselines: base rate, a one-feature rule (client spend), and logistic
  regression, against CatBoost.
- Business metric: what share of hires you keep, and how many Connects you
  spend per hire, if you only apply to the top-scored postings.
"""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from plotnine import (
    aes, geom_abline, geom_col, geom_density, geom_line, geom_point, geom_tile, geom_text,
    ggplot, labs, scale_color_manual, scale_fill_gradient, scale_fill_manual,
    scale_x_continuous, scale_y_continuous, coord_flip, coord_equal,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from _common import (
    BLUE, DATA, ORANGE, base_theme, load_features, pct_labels, save,
)

# From the problem statement: about 15 Connects (10-20) cost roughly $1.50.
CONNECTS_PER_APPLICATION = 15
USD_PER_CONNECT = 0.10
SEED = 42

df = load_features()
df["has_budget_fixed"] = df["budget_fixed"].fillna(0) > 0
df["budget_fixed"] = df["budget_fixed"].where(df["budget_fixed"] > 0)

NUMERIC = [
    "budget_fixed", "budget_hourly_min", "budget_hourly_max", "tags_count", "skills_count",
    "client_total_spent", "client_total_hires", "published_hour", "published_dow",
    "description_char_count", "description_word_count", "em_dash_count", "em_dash_ratio",
    "curly_quote_ratio", "non_ascii_ratio", "line_count", "exclamation_count",
    "bullet_count", "title_char_count", "desc_prior_copies", "title_prior_copies",
    "age_at_capture_h", "ai_style_signals", "markdown_marks", "emoji_count",
    "ai_phrase_per_100w", "sentence_len_mean", "sentence_len_cv", "ai_marker_score",
]
BOOL = [
    "location_mandatory", "enterprise_job", "premium", "client_top_client",
    "desc_has_url", "desc_has_email", "has_crypto_term", "has_wallet_term",
    "has_offplatform_contact", "has_budget_fixed",
    "fix_title", "new_build", "agency_tone", "long_term_wording", "screening_prompt", "quick_task_wording",
]
CATEGORICAL = ["job_type_label", "duration_label", "engagement", "tier_text", "client_country", "country_group",
               "client_payment_verification_status"]
FEATURES = NUMERIC + BOOL + CATEGORICAL

X = df[FEATURES].copy()
for c in BOOL:
    X[c] = X[c].astype(int)
for c in CATEGORICAL:
    X[c] = X[c].astype("object").where(X[c].notna(), "unknown").astype(str)
y = df["hired"].to_numpy()

n = len(df)
i_train, i_valid = int(n * 0.70), int(n * 0.85)
tr, va, te = slice(0, i_train), slice(i_train, i_valid), slice(i_valid, n)
print(f"train {i_train:,} | valid {i_valid - i_train:,} | test {n - i_valid:,}")
print("capture windows:",
      df["first_captured_at"].iloc[0].date(), df["first_captured_at"].iloc[i_train].date(),
      df["first_captured_at"].iloc[i_valid].date(), df["first_captured_at"].iloc[-1].date())
print(f"hire rate  train {y[tr].mean():.3f}  valid {y[va].mean():.3f}  test {y[te].mean():.3f}")

# Models
scores = {}

scores["Base rate (no model)"] = np.full(n - i_valid, y[tr].mean())

spend = np.log1p(df["client_total_spent"].fillna(0).to_numpy())
scores["Rule: client spend only"] = spend[te]

prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC + BOOL),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CATEGORICAL),
])
logit = make_pipeline(prep, LogisticRegression(max_iter=3000, C=0.5))
logit.fit(X.iloc[tr], y[tr])
scores["Logistic regression"] = logit.predict_proba(X.iloc[te])[:, 1]

cat_idx = [FEATURES.index(c) for c in CATEGORICAL]
cb = CatBoostClassifier(
    iterations=1500, learning_rate=0.04, depth=6, loss_function="Logloss",
    eval_metric="AUC", random_seed=SEED, verbose=0, early_stopping_rounds=100, allow_writing_files=False,
)
cb.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx),
       eval_set=Pool(X.iloc[va], y[va], cat_features=cat_idx))
test_pool = Pool(X.iloc[te], y[te], cat_features=cat_idx)
scores["CatBoost"] = cb.predict_proba(test_pool)[:, 1]
print(f"CatBoost best iteration: {cb.get_best_iteration()}")

# Metrics with bootstrap confidence intervals on the test set
rng = np.random.default_rng(SEED)
yt = y[te]
metrics = []
for name, s in scores.items():
    if name == "Base rate (no model)":
        metrics.append({"model": name, "roc_auc": 0.5, "roc_ci": "", "pr_auc": yt.mean(),
                        "brier": brier_score_loss(yt, s)})
        continue
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(yt), len(yt))
        if yt[idx].min() == yt[idx].max():
            continue
        boots.append(roc_auc_score(yt[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    prob_like = s.min() >= 0 and s.max() <= 1
    metrics.append({
        "model": name, "roc_auc": roc_auc_score(yt, s), "roc_ci": f"[{lo:.3f}, {hi:.3f}]",
        "pr_auc": average_precision_score(yt, s),
        "brier": brier_score_loss(yt, s) if prob_like else np.nan,
    })
mt = pd.DataFrame(metrics)
mt.to_csv(DATA / "ml_metrics.csv", index=False)
print(mt.round(3).to_string(index=False))

colors = {"Base rate (no model)": "#999999", "Rule: client spend only": "#7A9CC6",
          "Logistic regression": BLUE, "CatBoost": ORANGE}

# ROC
roc = []
for name, s in scores.items():
    if name == "Base rate (no model)":
        continue
    fpr, tpr, _ = roc_curve(yt, s)
    roc.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "model": name}))
roc = pd.concat(roc)
p = (
    ggplot(roc, aes("fpr", "tpr", color="model"))
    + geom_abline(slope=1, intercept=0, linetype="dashed", color="#777777")
    + geom_line(size=1)
    + scale_color_manual(colors)
    + coord_equal()
    + labs(title="ROC curves on the held-out, latest 15% of postings",
           subtitle="Time-based split: the model never sees the future",
           x="False positive rate", y="True positive rate", color="")
    + base_theme()
)
save(p, "16_roc_curves", 7, 6)

# Precision-recall
pr = []
for name, s in scores.items():
    if name == "Base rate (no model)":
        continue
    prec, rec, _ = precision_recall_curve(yt, s)
    pr.append(pd.DataFrame({"recall": rec, "precision": prec, "model": name}))
pr = pd.concat(pr)
p = (
    ggplot(pr, aes("recall", "precision", color="model"))
    + geom_line(size=1)
    + scale_color_manual(colors)
    + scale_y_continuous(labels=pct_labels, limits=(0, 1))
    + labs(title="Precision-recall curves", subtitle=f"Dashed baseline = test hire rate ({yt.mean():.1%})",
           x="Recall (share of hires kept)", y="Precision (hire rate among selected)", color="")
    + base_theme()
)
p = p + __import__("plotnine").geom_hline(yintercept=yt.mean(), linetype="dashed", color="#777777")
save(p, "17_precision_recall", 7.5, 5)

# Calibration (deciles of CatBoost score)
cal = pd.DataFrame({"p": scores["CatBoost"], "y": yt})
cal["bin"] = pd.qcut(cal["p"], 10, duplicates="drop")
cal = cal.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
p = (
    ggplot(cal, aes("pred", "obs"))
    + geom_abline(slope=1, intercept=0, linetype="dashed", color="#777777")
    + geom_line(color=ORANGE, size=1)
    + geom_point(color=ORANGE, size=3)
    + scale_x_continuous(labels=pct_labels)
    + scale_y_continuous(labels=pct_labels)
    + labs(title="Calibration of the CatBoost scores",
           subtitle="Deciles of predicted probability vs observed hire rate; on the diagonal = well calibrated",
           x="Mean predicted probability", y="Observed hire rate")
    + base_theme()
)
save(p, "18_calibration", 6.5, 5.5)

# Score distribution by outcome
dist = pd.DataFrame({"score": scores["CatBoost"], "outcome": np.where(yt == 1, "hired", "not hired")})
p = (
    ggplot(dist, aes("score", fill="outcome"))
    + geom_density(alpha=0.55, color="none")
    + scale_fill_manual({"hired": ORANGE, "not hired": "#7A7A7A"})
    + scale_x_continuous(labels=pct_labels)
    + labs(title="Model score by actual outcome",
           subtitle="CatBoost predicted probability of a hire on the test set",
           x="Predicted probability", y="Density", fill="")
    + base_theme()
)
save(p, "19_score_distribution", 7.5, 4.5)

# Feature importance (mean |SHAP|)
shap = cb.get_feature_importance(test_pool, type="ShapValues")[:, :-1]
imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(shap).mean(axis=0)})
imp = imp.sort_values("mean_abs_shap", ascending=False).head(15).iloc[::-1]
imp["feature"] = pd.Categorical(imp["feature"], categories=imp["feature"], ordered=True)
imp.sort_values("mean_abs_shap", ascending=False).to_csv(DATA / "ml_feature_importance.csv", index=False)
p = (
    ggplot(imp, aes("feature", "mean_abs_shap"))
    + geom_col(fill=BLUE)
    + coord_flip()
    + labs(title="What the model relies on", subtitle="Top 15 features by mean absolute SHAP value on the test set",
           x="", y="Mean absolute SHAP value (log-odds)")
    + base_theme()
)
save(p, "20_feature_importance", 8, 5.5)

# Business value: apply only to the top-scored share of postings
rows = []
order = np.argsort(-scores["CatBoost"])
for share in [1.0, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1]:
    k = max(1, int(round(share * len(yt))))
    sel = order[:k]
    hires = yt[sel].sum()
    rows.append({
        "share_applied": share,
        "applications": k,
        "hire_rate_among_applied": hires / k,
        "share_of_all_hires_kept": hires / yt.sum(),
        "connects_per_hire": (k * CONNECTS_PER_APPLICATION) / max(hires, 1),
        "usd_per_hire": (k * CONNECTS_PER_APPLICATION * USD_PER_CONNECT) / max(hires, 1),
        "usd_spent_on_no_hire_jobs": (k - hires) * CONNECTS_PER_APPLICATION * USD_PER_CONNECT,
    })
biz = pd.DataFrame(rows)
biz.to_csv(DATA / "ml_business_value.csv", index=False)
print(biz.round(3).to_string(index=False))

biz["label"] = biz["share_applied"].map(lambda v: f"Top {v:.0%}" if v < 1 else "Apply to all")
biz["label"] = pd.Categorical(biz["label"], categories=biz["label"], ordered=True)
p = (
    ggplot(biz, aes("label", "usd_per_hire"))
    + geom_col(fill=ORANGE, width=0.65)
    + geom_text(aes(label=[f"${v:.2f}" for v in biz["usd_per_hire"]]), va="bottom", nudge_y=0.3, size=9)
    + scale_y_continuous(expand=(0, 0, 0.15, 0))
    + labs(title="Cost of Connects per recorded hire",
           subtitle=f"Applying only to the top-scored postings; assumes {CONNECTS_PER_APPLICATION} Connects at ${USD_PER_CONNECT:.2f} each. A recorded hire is not a win for you.",
           x="", y="USD of Connects per hire in the market")
    + base_theme()
)
save(p, "21_cost_per_hire", 8, 4.8)

# Gains curve: share of hires captured vs share of postings applied to
frac = np.linspace(0, 1, 101)
gains = []
for name, s in scores.items():
    if name == "Base rate (no model)":
        continue
    o = np.argsort(-s)
    cum = np.concatenate([[0], np.cumsum(yt[o])]) / yt.sum()
    pos = np.linspace(0, len(yt), len(cum)) / len(yt)
    gains.append(pd.DataFrame({"applied": frac, "hires": np.interp(frac, pos, cum), "model": name}))
gains = pd.concat(gains)
p = (
    ggplot(gains, aes("applied", "hires", color="model"))
    + geom_abline(slope=1, intercept=0, linetype="dashed", color="#777777")
    + geom_line(size=1)
    + scale_color_manual(colors)
    + scale_x_continuous(labels=pct_labels)
    + scale_y_continuous(labels=pct_labels)
    + labs(title="Cumulative gains: how many hires you keep",
           subtitle="Rank postings by score and apply from the top; the dashed line is random choice",
           x="Share of postings applied to", y="Share of all hires kept", color="")
    + base_theme()
)
save(p, "22_gains_curve", 7.5, 5.5)

# Confusion matrix when applying to the top 30%
thr_share = 0.30
k = int(round(thr_share * len(yt)))
pred = np.zeros(len(yt), dtype=int)
pred[order[:k]] = 1
cm = pd.DataFrame({
    "actual": np.where(yt == 1, "Hired", "Not hired"),
    "decision": np.where(pred == 1, "Apply", "Skip"),
}).value_counts().reset_index(name="n")
cm["share"] = cm["n"] / cm["n"].sum()
p = (
    ggplot(cm, aes("decision", "actual", fill="n"))
    + geom_tile(color="white")
    + geom_text(aes(label=[f"{n:,}\n({s:.0%})" for n, s in zip(cm["n"], cm["share"])]), size=11)
    + scale_fill_gradient(low="#EAF1FB", high="#7A9CC6", guide=None)
    + labs(title=f"Decisions if you apply to the top {thr_share:.0%} by score",
           subtitle="Held-out test set", x="Decision", y="What happened")
    + base_theme()
)
save(p, "23_confusion_matrix", 6, 4.5)

out = {
    "split": {"train": i_train, "valid": i_valid - i_train, "test": n - i_valid},
    "test_hire_rate": float(yt.mean()),
    "catboost_best_iteration": int(cb.get_best_iteration()),
    "connects_per_application": CONNECTS_PER_APPLICATION,
    "usd_per_connect": USD_PER_CONNECT,
    "excluded_features": ["final_total_applicants", "final_total_invited_to_interview",
                          "proposals_tier", "is_applied", "seen_in_*", "hired_at", "days_to_hire"],
}
(DATA / "ml_summary.json").write_text(json.dumps(out, indent=2))
