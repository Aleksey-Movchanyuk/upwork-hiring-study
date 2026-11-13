"""Shared feature lists, human-readable feature groups and the time split for steps 10-12."""
import numpy as np

from _common import load_features

SEED = 42
CATEGORICAL = ["job_type_label", "duration_label", "engagement", "tier_text", "client_country", "country_group",
               "client_payment_verification_status"]
BOOL = ["client_top_client", "has_budget_fixed", "location_mandatory", "enterprise_job", "premium",
        "desc_has_url", "desc_has_email", "has_crypto_term", "has_wallet_term", "has_offplatform_contact",
        "fix_title", "new_build", "agency_tone", "long_term_wording", "screening_prompt", "quick_task_wording"]

# Human-readable groups: used to rank "which facts matter" and to name them in the article.
FEATURE_GROUPS = {
    "Job type (fixed price or hourly)": ["job_type_label"],
    "Budget": ["budget_fixed", "budget_hourly_min", "budget_hourly_max", "has_budget_fixed"],
    "Duration, engagement, tier": ["duration_label", "engagement", "tier_text"],
    "Client spend history": ["client_total_spent"],
    "Client past hires": ["client_total_hires"],
    "Client badge and payment check": ["client_top_client", "client_payment_verification_status"],
    "Client country": ["client_country", "country_group"],
    "Other job settings (tags, skills, premium...)": ["location_mandatory", "enterprise_job", "premium", "tags_count", "skills_count"],
    "Description length": ["description_char_count", "description_word_count"],
    "Title length": ["title_char_count"],
    "Text copied by others": ["desc_prior_copies", "title_prior_copies"],
    "Crypto and wallet words": ["has_crypto_term", "has_wallet_term"],
    "Asks to leave the platform": ["has_offplatform_contact"],
    "Agency wording": ["agency_tone"],
    "Asks to build something new": ["new_build"],
    "Other request wording (fix, long-term, quick, screening)": ["fix_title", "long_term_wording", "screening_prompt", "quick_task_wording"],
    "AI-style marks (dashes, quotes, markdown, dictionary score)": [
        "em_dash_count", "em_dash_ratio", "curly_quote_ratio", "ai_style_signals", "markdown_marks", "emoji_count",
        "ai_phrase_per_100w", "ai_marker_score", "non_ascii_ratio"],
    "Writing style (sentences, lines, bullets, !)": ["sentence_len_mean", "sentence_len_cv", "line_count", "exclamation_count", "bullet_count"],
    "Links and emails in the text": ["desc_has_url", "desc_has_email"],
    "Posting time (hour, weekday)": ["published_hour", "published_dow", "age_at_capture_h"],
}
# desc_has_url / desc_has_email are listed once (text group) and removed from the BOOL-only lists below if repeated
ALL = []
for _cols in FEATURE_GROUPS.values():
    for _c in _cols:
        if _c not in ALL:
            ALL.append(_c)


def model_frame():
    """Return df, X (model-ready), y, (train, valid, test slices): same 70/15/15 time split as 04_ml.py."""
    df = load_features()
    df["has_budget_fixed"] = df["budget_fixed"].fillna(0) > 0
    df["budget_fixed"] = df["budget_fixed"].where(df["budget_fixed"] > 0)
    y = df["hired"].astype(int).to_numpy()
    n = len(df)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    X = df[ALL].copy()
    for c in ALL:
        if c in BOOL:
            X[c] = X[c].fillna(False).astype(int)
        elif c in CATEGORICAL:
            X[c] = X[c].astype("object").where(X[c].notna(), "unknown").astype(str)
    return df, X, y, (slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n))


def bootstrap_auc(y, s, n_boot=500, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), *np.percentile(b, [2.5, 97.5])


def top_share_stats(y, s, share=0.30, connects=15, usd=0.10, seed=SEED):
    """Hire rate among the top `share` by score, share of all hires kept, USD per recorded hire.
    Ties are broken at random, which matters for rules with few distinct scores."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(s)), -np.asarray(s, dtype=float)))
    k = int(round(len(s) * share))
    sel = order[:k]
    hires = y[sel].sum()
    return {"top_hire_rate": float(y[sel].mean()), "hires_kept": float(hires / y.sum()),
            "usd_per_hire": float(k * connects * usd / hires) if hires else float("nan")}
