"""Step 6: validate the AI-style proxy against your manual labels.

Run after filling `your_label` in data/label_sample_PRIVATE.csv. Reports how well
`ai_style_signals` separates postings you judged AI-written from hand-written,
and whether your AI/human label itself relates to hiring.
"""
import json

import numpy as np
import pandas as pd
from plotnine import aes, geom_col, geom_linerange, geom_text, ggplot, labs, scale_fill_manual, scale_y_continuous
from sklearn.metrics import roc_auc_score

from _common import DATA, ORANGE, base_theme, load_features, pct_labels, rate_table, save

path = DATA / "label_sample_PRIVATE.csv"
lab = pd.read_csv(path)
lab["your_label"] = lab["your_label"].astype(str).str.strip().str.lower()
lab = lab[lab["your_label"].isin(["ai", "human"])]
if lab["your_label"].nunique() < 2 or len(lab) < 40:
    raise SystemExit(f"Need at least 40 labelled rows with both classes (have {len(lab)}). Fill {path.name} first.")

df = load_features().merge(lab[["job_key", "your_label"]], on="job_key", how="inner")
is_ai = (df["your_label"] == "ai").astype(int)
auc = roc_auc_score(is_ai, df["ai_style_signals"])
rule = (df["ai_style_signals"] >= 2).astype(int)
tp = int(((rule == 1) & (is_ai == 1)).sum())
res = {
    "n_labelled": int(len(df)), "share_ai": float(is_ai.mean()),
    "proxy_auc_vs_manual": round(float(auc), 3),
    "rule_2plus_signals": {"precision": tp / max(int(rule.sum()), 1), "recall": tp / max(int(is_ai.sum()), 1)},
}
t = rate_table(df.assign(label=np.where(is_ai == 1, "AI-written (manual)", "Hand-written (manual)")), "label")
res["hire_rate_by_manual_label"] = t[["label", "n", "rate", "lo", "hi"]].round(4).to_dict("records")
(DATA / "ai_proxy_validation.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

t["text"] = t.apply(lambda r: f"{r['rate']:.1%}\nn={int(r['n'])}", axis=1)
p = (
    ggplot(t, aes("label", "rate", fill="label"))
    + geom_col(width=0.55, show_legend=False)
    + geom_linerange(aes(ymin="lo", ymax="hi"))
    + geom_text(aes(label="text", y="hi"), va="bottom", nudge_y=0.01, size=9)
    + scale_fill_manual({"AI-written (manual)": ORANGE, "Hand-written (manual)": "#7A9CC6"})
    + scale_y_continuous(labels=pct_labels, expand=(0, 0, 0.25, 0))
    + labs(title="Hire rate by manual AI / human judgement",
           subtitle="Postings hand-labelled by the author; 95% Wilson interval", x="", y="Hire rate")
    + base_theme()
)
save(p, "24_manual_label_hire_rate", 6.5, 4.5)
