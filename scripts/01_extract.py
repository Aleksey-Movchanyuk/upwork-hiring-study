"""Step 1: pull labelled jobs from Postgres and cache a de-identified feature table.

The cached parquet has NO title, description, URL or job id, only derived
numeric/categorical features, so it can be shared and plotted from safely.
Everything downstream (02..04) reads this file, not the database.
"""
import hashlib
import re
from datetime import timezone

import pandas as pd
import numpy as np
import pycountry
from dotenv import load_dotenv

from _common import DATA, FEATURES, ROOT

load_dotenv(ROOT / ".env")

from upwork_ml.features.structured import extract_structured_features  # noqa: E402
from upwork_ml.features.text_signals import extract_text_features  # noqa: E402
from upwork_ml.db import dict_cursor, get_connection  # noqa: E402
from ai_markers import ai_score  # noqa: E402  (dictionary of LLM fingerprints, same folder)

# "AI-style" proxy signals. These are stylistic tells, not a detector: plenty of
# humans use em dashes and plenty of clients paste LLM text without them.
MARKDOWN_RE = re.compile(r"\*\*[^*\n]+\*\*|^#{1,4}\s", re.M)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2705\u274C]")
AI_PHRASE_RE = re.compile(
    r"\b(delve\w*|leverag\w*|seamless\w*|robust|cutting-edge|tailored|elevate|unlock\w*|"
    r"game-?changer|navigat\w*|fast-paced|dynamic|passionate|dedicated professional|"
    r"key responsibilities|ideal candidate|we are seeking|we are looking for a|"
    r"look forward to|streamlin\w*|holistic|spearhead\w*|proven track record)\b", re.I)
SENTENCE_RE = re.compile(r"[.!?]+\s+|\n+")
URL_RE = re.compile(r"https?://|www\.", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
BULLET_RE = re.compile(r"^\s*([-*•●▪]|\d+[.)])\s+", re.M)
# Hypotheses from the problem statement: crypto jobs that harvest wallet keys,
# requests to leave the platform, and re-posted ("arbitrage") listings.
CRYPTO_RE = re.compile(
    r"\b(crypto\w*|bitcoin|btc|ethereum|blockchain|web3|nft|defi|solana|binance|"
    r"metamask|airdrop|memecoin|dapp)\b", re.I)
WALLET_RE = re.compile(r"(seed phrase|private key|wallet|metamask|ledger|recovery phrase)", re.I)
OFFPLATFORM_RE = re.compile(r"(telegram|t\.me/|whatsapp|skype|discord|wechat|signal app)", re.I)

# What the client is asking for (content type). Word lists were written before looking at
# hire outcomes; "fixed price" is excluded from the fix pattern on purpose.
FIX_RE = re.compile(
    r"\b(fix|fixes|fixing|fixed(?![\s\-]*(?:price|priced|pricing|budget|rate|fee|scope|cost|amount|bid|payment|milestone))|"
    r"bug|bugs|bugfix|debug|debugging|broken|not working|doesn'?t work|stopped working|troubleshoot\w*|"
    r"repair\w*|crash\w*|glitch\w*)\b", re.I)
NEW_BUILD_RE = re.compile(
    r"\b(from scratch|build (?:a|an|me|our)|develop (?:a|an|our)|create (?:a|an|our)|"
    r"new (?:website|app|application|platform|site|system|product|tool|bot)|mvp|prototype|startup|brand new)\b", re.I)
AGENCY_RE = re.compile(
    r"\b(our client|my client|on behalf of|white[- ]?label|our agency|we are an agency|"
    r"we are a (?:digital |marketing |software |development )?agency|reseller|subcontract\w*)\b", re.I)
LONG_TERM_RE = re.compile(
    r"\b(long[- ]?term|ongoing|full[- ]?time|part[- ]?time|retainer|permanent|for months|6 months|12 months)\b", re.I)
SCREENING_RE = re.compile(
    r"(start your (?:proposal|application|cover letter)|begin your (?:proposal|application)|"
    r"include the (?:word|phrase|code)|first line of your (?:proposal|application)|"
    r"answer the following|answer these questions|answer the questions)", re.I)
QUICK_RE = re.compile(
    r"\b(quick|small|simple|minor|one[- ]?time|one[- ]?off|short task|few hours|couple of hours)\b", re.I)

NUMERIC = [
    "budget_fixed", "budget_hourly_min", "budget_hourly_max", "freelancers_to_hire",
    "connect_price", "client_total_spent", "client_total_hires", "client_total_reviews",
    "client_total_feedback", "client_recruiting_recency_days", "final_total_applicants",
    "final_total_invited_to_interview", "client_payment_verification_status",
    "days_to_hire", "observed_days", "age_at_capture_h", "checks_count", "desc_dup_count", "title_dup_count",
    "desc_prior_copies", "title_prior_copies",
]


# Country groups follow the hypotheses from experience: English-speaking markets,
# Europe, the Gulf, Latin America, East Asia.
GROUPS = {
    "Anglosphere": {"US", "GB", "AU", "CA", "NZ"},
    "Europe": {"DE", "FR", "NL", "ES", "IT", "CH", "SE", "NO", "DK", "FI", "BE", "AT", "IE",
               "PT", "PL", "CZ", "GR", "RO", "HU", "UA", "EE", "LV", "LT", "LU", "IS", "MT",
               "CY", "HR", "SI", "SK", "BG", "RS", "BA", "AL", "MK", "MD", "BY", "ME", "LI",
               "MC", "AD", "SM"},
    "Gulf": {"SA", "QA", "BH", "AE", "KW", "OM"},
    "Latin America": {"MX", "BR", "AR", "CL", "CO", "PE", "VE", "EC", "UY", "PY", "BO", "CR",
                      "PA", "DO", "GT", "HN", "SV", "NI", "CU", "PR"},
    "East Asia": {"JP", "KR", "TW", "HK", "SG", "CN", "MO"},
    "South Asia": {"IN", "PK", "BD", "LK", "NP"},
    "South-East Asia": {"PH", "ID", "VN", "TH", "MY", "KH", "MM", "LA"},
    "Africa": {"NG", "KE", "ZA", "EG", "GH", "MA", "TN", "DZ", "ET", "UG", "TZ", "ZW", "SN", "CM"},
}


def country_group(name: str) -> str:
    if not name or name == "unknown":
        return "unknown"
    try:
        code = pycountry.countries.lookup(name).alpha_2
    except LookupError:
        return "Other"
    for group, codes in GROUPS.items():
        if code in codes:
            return group
    return "Other"


COUNTRY_OVERRIDES = {"USA": "United States", "GBR": "United Kingdom", "KOR": "South Korea",
                     "RUS": "Russia", "VNM": "Vietnam", "TUR": "Turkey"}


def normalise_country(value: str | None) -> str:
    """The feed mixes ISO-3 codes (USA, PAK) with names (United States, Pakistan)."""
    if not value:
        return "unknown"
    value = value.strip()
    if len(value) == 3 and value.isupper():
        if value in COUNTRY_OVERRIDES:
            return COUNTRY_OVERRIDES[value]
        match = pycountry.countries.get(alpha_3=value)
        return getattr(match, "common_name", None) or match.name if match else value
    return value


def extra_text_features(title: str | None, description: str | None) -> dict:
    text = description or ""
    n = len(text)
    both = f"{title or ''} {text}"
    sentences = [x for x in SENTENCE_RE.split(text) if len(x.split()) >= 3]
    lens = np.array([len(x.split()) for x in sentences]) if sentences else np.array([0])
    words = max(len(text.split()), 1)
    return {
        "markdown_marks": len(MARKDOWN_RE.findall(text)),
        "emoji_count": len(EMOJI_RE.findall(text)),
        "ai_phrase_per_100w": 100 * len(AI_PHRASE_RE.findall(text)) / words,
        "sentence_len_mean": float(lens.mean()),
        "sentence_len_cv": float(lens.std() / lens.mean()) if lens.mean() else 0.0,
        "has_crypto_term": bool(CRYPTO_RE.search(both)),
        "has_wallet_term": bool(WALLET_RE.search(both)),
        "has_offplatform_contact": bool(OFFPLATFORM_RE.search(both)),
        "fix_title": bool(FIX_RE.search(title or "")),
        "new_build": bool(NEW_BUILD_RE.search(both)),
        "agency_tone": bool(AGENCY_RE.search(both)),
        "long_term_wording": bool(LONG_TERM_RE.search(both)),
        "screening_prompt": bool(SCREENING_RE.search(both)),
        "quick_task_wording": bool(QUICK_RE.search(both)),
        "ai_marker_score": ai_score(f"{title or ''}\n{text}")["score"],
        "title_char_count": len(title or ""),
        "non_ascii_ratio": (sum(ord(c) > 127 for c in text) / n) if n else 0.0,
        "line_count": text.count("\n") + 1 if text else 0,
        "exclamation_count": text.count("!"),
        "bullet_count": len(BULLET_RE.findall(text)),
        "desc_has_url": bool(URL_RE.search(text)),
        "desc_has_email": bool(EMAIL_RE.search(text)),
    }


# *_dup_count: total copies of the same normalised description / title across ALL
# captured jobs (descriptive use only, it looks into the future).
# *_prior_copies: copies captured before this one (safe to use as a model feature).
QUERY = r"""
with j as (
    select jobs.*,
           count(*) over (partition by md5(lower(regexp_replace(coalesce(description, ''), '\s+', ' ', 'g')))) as desc_dup_count,
           count(*) over (partition by lower(trim(coalesce(title, '')))) as title_dup_count,
           -- copies captured strictly BEFORE this posting: what a freelancer could know at decision time
           row_number() over (partition by md5(lower(regexp_replace(coalesce(description, ''), '\s+', ' ', 'g')))
                              order by first_captured_at, uid) - 1 as desc_prior_copies,
           row_number() over (partition by lower(trim(coalesce(title, '')))
                              order by first_captured_at, uid) - 1 as title_prior_copies
    from jobs
)
select j.*, o.hired, o.hired_at, o.resolved_at, o.checks_count,
       o.final_total_applicants, o.final_total_invited_to_interview
from j
join job_outcomes o on o.uid = j.uid
order by j.published_on asc
"""


def load_rows() -> list[dict]:
    """Jobs with a resolved 24h outcome. Jobs still inside the window are
    excluded, not treated as negatives."""
    conn = get_connection()
    try:
        with dict_cursor(conn) as cur:
            cur.execute(QUERY)
            return cur.fetchall()
    finally:
        conn.close()


def main():
    rows = load_rows()
    records = []
    for r in rows:
        raw = r.get("raw_json") or {}
        if r.get("freelancers_to_hire") is None:
            r["freelancers_to_hire"] = raw.get("freelancersToHire")
        published = r["published_on"].astimezone(timezone.utc) if r["published_on"] else None
        rec = {
            **extract_structured_features(r),
            **extract_text_features(r["description"]),
            **extra_text_features(r["title"], r["description"]),
            # one-way key so manual labels can be joined back without storing the job id
            "job_key": hashlib.sha1(r["uid"].encode()).hexdigest()[:12],
            "published_on": published,
            "published_hour": published.hour if published else None,  # UTC
            "published_dow": published.weekday() if published else None,  # 0 = Monday, UTC
            "final_total_applicants": r.get("final_total_applicants"),
            "final_total_invited_to_interview": r.get("final_total_invited_to_interview"),
            "client_country": normalise_country(r["client_country"]),
            "country_group": country_group(normalise_country(r["client_country"])),
            "source_feed": r.get("source_feed"),
            "first_captured_at": r["first_captured_at"].astimezone(timezone.utc),
            "age_at_capture_h": (r["first_captured_at"] - r["published_on"]).total_seconds() / 3600,
            # Observation length and time-to-hire are measured from capture, not from posting.
            "observed_days": (r["resolved_at"] - r["first_captured_at"]).total_seconds() / 86400,
            "days_to_hire": (
                (r["hired_at"] - r["first_captured_at"]).total_seconds() / 86400
                if r["hired"] and r["hired_at"] else None
            ),
            "checks_count": r.get("checks_count"),
            "desc_dup_count": r["desc_dup_count"],
            "title_dup_count": r["title_dup_count"],
            "desc_prior_copies": r["desc_prior_copies"],
            "title_prior_copies": r["title_prior_copies"],
            "hired": bool(r["hired"]),
        }
        records.append(rec)

    df = pd.DataFrame(records)
    # Transparent proxy: how many of five stylistic tells appear (0-5).
    df["ai_style_signals"] = (
        (df["em_dash_count"] > 0).astype(int)
        + (df["curly_quote_ratio"] > 0.5).astype(int)
        + (df["markdown_marks"] > 0).astype(int)
        + (df["ai_phrase_per_100w"] >= 0.5).astype(int)
        + (df["emoji_count"] > 0).astype(int)
    )
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["client_payment_verification_status"] = df["client_payment_verification_status"].astype("Float64")

    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES, index=False)
    print(f"{len(df):,} labelled jobs, hire rate {df['hired'].mean():.1%} -> {FEATURES.relative_to(ROOT)}")
    print(f"window: {df['published_on'].min():%Y-%m-%d} to {df['published_on'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
