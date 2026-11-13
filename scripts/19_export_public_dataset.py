"""Step 19 (author only): export the anonymised feature table for GitHub.

Needs the private data/features.parquet. Writes public/upwork_features_public.csv with:
  time_rank (1 = oldest posting), hired, and the 52 model facts. NO text, titles, links, ids, dates or client names.
Large money values are rounded to 3 significant figures so that one client's exact totals cannot be looked up.
"""
import numpy as np
import pandas as pd

from _common import ART
from _features import ALL, model_frame

df, X, y, _ = model_frame()
out = X.copy()


def sig3(s):
    s = pd.to_numeric(s, errors="coerce")
    mag = 10 ** (np.floor(np.log10(s.where(s > 0))) - 2)
    return (np.round(s / mag) * mag).where(s > 0, s)


for c in ("client_total_spent", "budget_fixed", "budget_hourly_min", "budget_hourly_max", "client_total_hires"):
    out[c] = sig3(out[c])
out.insert(0, "hired", y)
out.insert(0, "time_rank", np.arange(1, len(out) + 1))
assert list(out.columns) == ["time_rank", "hired"] + ALL
pub = ART / "public"
pub.mkdir(exist_ok=True)
out.to_csv(pub / "upwork_features_public.csv", index=False)
print(out.shape, "hire rate", round(out["hired"].mean(), 4))
print("columns:", len(ALL), "| text-like columns:", [c for c in out.columns if out[c].dtype == object and c not in ("job_type_label", "duration_label", "engagement", "tier_text", "client_country", "country_group", "client_payment_verification_status")])
