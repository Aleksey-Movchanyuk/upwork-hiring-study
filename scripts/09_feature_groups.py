"""Step 9: which KIND of information predicts a hire?

Same time split and CatBoost settings as 04_ml.py. Each model sees only one group of
features (or all but one group). Test AUC on the latest 15% of postings, with a bootstrap
95% interval. Groups:
  client   who the client is: spend, past hires, badge, country
  job      what kind of job: type, duration, engagement, budget, tier, tags
  text     what the text looks like and says: length, style, AI markers, crypto, content flags, copies
  timing   when it was posted
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from plotnine import aes, coord_flip, geom_hline, geom_pointrange, ggplot, labs, scale_y_continuous
from sklearn.metrics import roc_auc_score

from _common import BLUE, DATA, base_theme, load_features, save

SEED = 42
GROUPS = {
    "client": ["client_total_spent", "client_total_hires", "client_top_client",
               "client_payment_verification_status", "client_country", "country_group"],
    "job": ["job_type_label", "duration_label", "engagement", "tier_text", "budget_fixed", "budget_hourly_min",
            "budget_hourly_max", "has_budget_fixed", "location_mandatory", "enterprise_job", "premium",
            "tags_count", "skills_count"],
    "text": ["description_char_count", "description_word_count", "title_char_count", "em_dash_count",
             "em_dash_ratio", "curly_quote_ratio", "non_ascii_ratio", "line_count", "exclamation_count",
             "bullet_count", "ai_style_signals", "markdown_marks", "emoji_count", "ai_phrase_per_100w",
             "sentence_len_mean", "sentence_len_cv", "ai_marker_score", "desc_has_url", "desc_has_email",
             "has_crypto_term", "has_wallet_term", "has_offplatform_contact", "fix_title", "new_build",
             "agency_tone", "long_term_wording", "screening_prompt", "quick_task_wording",
             "desc_prior_copies", "title_prior_copies"],
    "timing": ["published_hour", "published_dow", "age_at_capture_h"],
}
CATEGORICAL = {"job_type_label", "duration_label", "engagement", "tier_text", "client_country",
               "country_group", "client_payment_verification_status"}
BOOL = {"client_top_client", "has_budget_fixed", "location_mandatory", "enterprise_job", "premium",
        "desc_has_url", "desc_has_email", "has_crypto_term", "has_wallet_term", "has_offplatform_contact",
        "fix_title", "new_build", "agency_tone", "long_term_wording", "screening_prompt", "quick_task_wording"}

df = load_features()
df["has_budget_fixed"] = df["budget_fixed"].fillna(0) > 0
df["budget_fixed"] = df["budget_fixed"].where(df["budget_fixed"] > 0)
y = df["hired"].astype(int).to_numpy()
n = len(df)
i_tr, i_va = int(n * 0.70), int(n * 0.85)
tr, va, te = slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)


def auc_for(cols):
    X = df[cols].copy()
    for c in cols:
        if c in BOOL:
            X[c] = X[c].fillna(False).astype(int)
        elif c in CATEGORICAL:
            X[c] = X[c].astype("object").where(X[c].notna(), "unknown").astype(str)
    cat_idx = [cols.index(c) for c in cols if c in CATEGORICAL]
    cb = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6, eval_metric="AUC", random_seed=SEED,
                            verbose=0, early_stopping_rounds=100, allow_writing_files=False)
    cb.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx), eval_set=Pool(X.iloc[va], y[va], cat_features=cat_idx))
    s = cb.predict_proba(Pool(X.iloc[te], y[te], cat_features=cat_idx))[:, 1]
    rng, yt, boots = np.random.default_rng(SEED), y[te], []
    for _ in range(500):
        i = rng.integers(0, len(yt), len(yt))
        boots.append(roc_auc_score(yt[i], s[i]))
    return roc_auc_score(yt, s), *np.percentile(boots, [2.5, 97.5])


allc = [c for g in GROUPS.values() for c in g]
runs = {
    "Client only": GROUPS["client"],
    "Client + job type": GROUPS["client"] + GROUPS["job"],
    "Job only": GROUPS["job"],
    "Text only": GROUPS["text"],
    "Timing only": GROUPS["timing"],
    "All features": allc,
    "All except text": [c for c in allc if c not in GROUPS["text"]],
    "All except client": [c for c in allc if c not in GROUPS["client"]],
    "All except job": [c for c in allc if c not in GROUPS["job"]],
}
rows = []
for name, cols in runs.items():
    a, lo, hi = auc_for(cols)
    rows.append({"features": name, "n_features": len(cols), "roc_auc": a, "lo": lo, "hi": hi})
    print(f"{name:20s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]")
res = pd.DataFrame(rows)
res.to_csv(DATA / "ml_feature_groups.csv", index=False)

plot = res.sort_values("roc_auc").copy()
plot["features"] = pd.Categorical(plot["features"], categories=plot["features"], ordered=True)
p = (
    ggplot(plot, aes("features", "roc_auc"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), color=BLUE)
    + geom_hline(yintercept=0.5, linetype="dashed", color="#444444")
    + coord_flip()
    + scale_y_continuous(limits=(0.45, 0.8))
    + labs(title="The kind of job and the client matter most; the text adds little on top",
           subtitle="Test ROC AUC (latest 15% of postings) for models that see only the listed information; 0.5 = guessing",
           x="", y="ROC AUC")
    + base_theme()
)
save(p, "26_feature_groups", 9, 5)
