"""Step 7: apply the AI-marker dictionary (ai_markers.py) to every labelled posting.

Outputs (de-identified; no text):
  data/ai_markers.parquet      job_key, ai_marker_score, ai_marker_level, one column per marker
  data/ai_marker_summary.csv   how common each marker is and the hire rate with it
  data/ai_marker_levels.csv    hire rate by level (no markers / possible / likely AI)
If data/label_sample_PRIVATE.csv has your ai/human labels, also reports accuracy vs them.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

from _common import DATA, ROOT, load_features, rate_table
from ai_markers import ai_score

load_dotenv(ROOT / ".env")
from upwork_ml.db import get_connection  # noqa: E402  (after sys.path set by _common)

with get_connection() as conn:
    t = pd.read_sql(
        "select j.uid, j.title, j.description from data_upwork.jobs j "
        "join data_upwork.job_outcomes o on o.uid = j.uid", conn)
t["job_key"] = t["uid"].map(lambda u: hashlib.sha1(u.encode()).hexdigest()[:12])
text = (t["title"].fillna("") + "\n" + t["description"].fillna(""))
res = [ai_score(s) for s in text]
out = pd.DataFrame({"job_key": t["job_key"], "ai_marker_score": [r["score"] for r in res],
                    "ai_marker_level": [r["level"] for r in res]})
markers = sorted({m for r in res for m in r["hits"]})
for m in markers:
    out["m: " + m] = [r["hits"].get(m, 0) for r in res]
out.to_parquet(DATA / "ai_markers.parquet", index=False)
del t, text  # raw text never leaves memory

df = load_features().merge(out, on="job_key", how="inner")
df["hired"] = df["hired"].astype(int)

# 1) hire rate by level
lv = rate_table(df, "ai_marker_level")
lv["share"] = lv["n"] / lv["n"].sum()
lv.to_csv(DATA / "ai_marker_levels.csv", index=False)
print("Hire rate by AI-marker level"); print(lv.round(3).to_string(index=False))

# 2) per marker
rows = []
for m in markers:
    f = df["m: " + m] > 0
    if f.sum() < 30:
        continue
    rows.append({"marker": m, "n": int(f.sum()), "share": f.mean(), "hire_rate": df.loc[f, "hired"].mean(),
                 "hire_rate_without": df.loc[~f, "hired"].mean(), "median_words": df.loc[f, "description_word_count"].median()})
ms = pd.DataFrame(rows).sort_values("n", ascending=False)
ms.to_csv(DATA / "ai_marker_summary.csv", index=False)
print("\nPer marker"); print(ms.round(3).to_string(index=False))

# 3) adjusted effect of the score (length, client history, job type)
X = pd.DataFrame({
    "ai_score": df["ai_marker_score"],
    "fixed_price": (df["job_type_label"] == "Fixed Price").astype(int),
    "no_spend": (df["client_total_spent"].fillna(0) == 0).astype(int)})
for n, s in {"log_words": np.log10(df["description_word_count"] + 1),
             "log_spend": np.log10(df["client_total_spent"].fillna(0) + 1),
             "log_hires": np.log1p(df["client_total_hires"].fillna(0))}.items():
    X[n] = (s - s.mean()) / s.std()
m = sm.Logit(df["hired"], sm.add_constant(X)).fit(disp=0)
ci = np.exp(m.conf_int())
adj = {"or_per_score_point": round(float(np.exp(m.params["ai_score"])), 3),
       "ci": [round(float(ci.loc["ai_score", 0]), 3), round(float(ci.loc["ai_score", 1]), 3)],
       "p": float(m.pvalues["ai_score"]),
       "corr_score_vs_words": round(float(df["ai_marker_score"].corr(np.log10(df["description_word_count"] + 1), method="spearman")), 3)}
print("\nAdjusted effect:", adj)

# 4) validation against manual labels, if present
summary = {"adjusted": adj, "levels": lv[["ai_marker_level", "n", "rate"]].round(4).to_dict("records")}
lab_path = DATA / "label_sample_PRIVATE.csv"
if lab_path.exists():
    lab = pd.read_csv(lab_path)
    lab["your_label"] = lab["your_label"].astype(str).str.strip().str.lower()
    lab = lab[lab["your_label"].isin(["ai", "human"])]
    v = df.merge(lab[["job_key", "your_label"]], on="job_key")
    if len(v) >= 40 and v["your_label"].nunique() == 2:
        y = (v["your_label"] == "ai").astype(int)
        summary["validation"] = {"n": int(len(v)), "auc_new_score": round(float(roc_auc_score(y, v["ai_marker_score"])), 3),
                                 "auc_old_proxy": round(float(roc_auc_score(y, v["ai_style_signals"])), 3)}
        for name, thr in (("possible_or_higher", 1.0), ("likely", 3.0)):
            p = (v["ai_marker_score"] >= thr).astype(int)
            tp = int(((p == 1) & (y == 1)).sum())
            summary["validation"][name] = {"precision": tp / max(int(p.sum()), 1), "recall": tp / max(int(y.sum()), 1)}
    else:
        summary["validation"] = f"only {len(v)} usable manual labels; fill your_label (ai/human) for at least 40 rows"
else:
    summary["validation"] = "label sheet missing; run 05_label_sample.py"
(DATA / "ai_marker_summary.json").write_text(json.dumps(summary, indent=2))
print("\nValidation:", summary["validation"])
