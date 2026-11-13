"""Step 14: the freelancer's view.

1. Market money per posting: hire rate x typical budget, by fixed-price budget bucket. This is NOT what the
   freelancer would earn (it ignores the chance of winning), but it shows that cheap jobs hire most and pay least.
2. A 60-second checklist of seven yes/no facts visible at application time, and the hire rate by points scored.
   CAUTION: the cut-offs (10+ hires, 3 months, 400 words) were chosen after looking at all the data, so the scores
   are optimistic. They are shown for all postings and, separately, for the latest 15% only.
3. What kind of jobs these are: share of postings mentioning broad topic words (the skills field is mostly empty).
4. Robustness: does the job-length effect survive when only postings observed for 14+ days are used?
Text stays in memory; only counts and shares are saved.
"""
import hashlib
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from plotnine import (aes, element_text, geom_col, geom_pointrange, geom_text, ggplot, labs, scale_color_manual,
                      scale_fill_manual, scale_y_continuous, theme)

from _common import BLUE, DATA, ORANGE, ROOT, base_theme, load_features, pct_labels, save

load_dotenv(ROOT / ".env")
df = load_features()
df["hired"] = df["hired"].astype(int)
n = len(df)
test = df.iloc[int(n * 0.85):]

# 1. Market money per posting, fixed-price
f = df[(df["job_type_label"] == "Fixed Price") & (df["budget_fixed"] > 0)].copy()
f["bucket"] = pd.cut(f["budget_fixed"], [0, 20, 80, 220, 650, 1e9],
                     labels=["up to $20", "$20-80", "$80-220", "$220-650", "over $650"])
money = f.groupby("bucket", observed=True).agg(postings=("hired", "size"), hire_rate=("hired", "mean"),
                                                median_budget=("budget_fixed", "median"))
money["market_money_per_posting"] = money["hire_rate"] * money["median_budget"]
money = money.reset_index()
money.to_csv(DATA / "market_money_by_budget.csv", index=False)
print(money.round(3).to_string(index=False))
money["label"] = money.apply(lambda r: f"{r['hire_rate']:.0%} hire\nmedian ${r['median_budget']:,.0f}", axis=1)
p = (
    ggplot(money, aes("bucket", "market_money_per_posting"))
    + geom_col(fill=BLUE, width=0.65)
    + geom_text(aes(label="label"), va="bottom", nudge_y=6, size=8)
    + scale_y_continuous(limits=(0, 250))
    + labs(title="Cheap jobs hire most, but big jobs are worth far more per application",
           subtitle="Fixed-price jobs: hire rate x median budget = typical market money per posting (not what you would win)",
           x="Fixed-price budget", y="Hire rate x median budget ($)")
    + base_theme()
)
save(p, "31_market_money_by_budget", 8, 4.8)

# 2. Seven-point checklist
dur_short = df["duration_label"].isin(["Less than 1 week", "Less than 1 month", "1 to 3 months"])
items = {
    "Fixed-price job": df["job_type_label"] == "Fixed Price",
    "Client has spent money": df["client_total_spent"].fillna(0) > 0,
    "Client has 10+ past hires": df["client_total_hires"].fillna(0) >= 10,
    "Project length under 3 months": dur_short,
    "No crypto wording": ~df["has_crypto_term"],
    "No agency wording": ~df["agency_tone"],
    "Description up to 400 words": df["description_word_count"] <= 400,
}
rows = []
for name, m in items.items():
    m = m.fillna(False).astype(bool)
    rows.append({"item": name, "share_yes": m.mean(), "n_yes": int(m.sum()), "hire_rate_yes": df.loc[m, "hired"].mean(),
                 "n_no": int((~m).sum()), "hire_rate_no": df.loc[~m, "hired"].mean()})
chk = pd.DataFrame(rows)
chk.to_csv(DATA / "checklist_items.csv", index=False)
print(chk.round(3).to_string(index=False))

df["points"] = sum(m.fillna(False).astype(int) for m in items.values())
test = df.iloc[int(n * 0.85):]
band = lambda s: pd.cut(s, [-1, 2, 3, 4, 5, 7], labels=["0-2", "3", "4", "5", "6-7"])
out = []
for label, d in (("All postings", df), ("Latest 15% only", test)):
    for b, g in d.groupby(band(d["points"]), observed=True):
        r, k = g["hired"].mean(), len(g)
        se = np.sqrt(r * (1 - r) / k)
        out.append({"period": label, "points": b, "postings": k, "hire_rate": r, "lo": max(r - 1.96 * se, 0), "hi": r + 1.96 * se})
pts = pd.DataFrame(out)
pts.to_csv(DATA / "checklist_by_points.csv", index=False)
print(pts.round(3).to_string(index=False))
p = (
    ggplot(pts, aes("points", "hire_rate", color="period"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), position=__import__("plotnine").position_dodge(width=0.4))
    + scale_color_manual({"All postings": BLUE, "Latest 15% only": ORANGE})
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="The more boxes you tick, the more often the post hires",
           subtitle="Hire rate by points on the seven-point checklist (cut-offs chosen after seeing the data, so a little optimistic)",
           x="Points (out of 7)", y="Hire rate", color="")
    + base_theme()
)
save(p, "32_checklist_points", 8, 4.5)

# 3. What kind of jobs these are (share of postings mentioning broad topic words)
from upwork_ml.db import get_connection  # noqa: E402

with get_connection() as conn:
    t = pd.read_sql("select j.uid, j.title, j.description from data_upwork.jobs j "
                    "join data_upwork.job_outcomes o on o.uid = j.uid", conn)
txt = (t["title"].fillna("") + " " + t["description"].fillna("")).str.lower()
topics = {
    "AI / LLM": r"\b(ai|llm|llms|gpt|chatgpt|openai|claude|gemini|machine learning|langchain|rag)\b",
    "Automation / workflows": r"\b(automation|automations|n8n|zapier|make\.com|workflow|workflows|bot|chatbot)\b",
    "Software development": r"\b(api|python|javascript|typescript|react|node\.?js|backend|frontend|full-?stack|django|flask|developer)\b",
    "Data work": r"\b(data entry|scraping|scraper|excel|spreadsheet|google sheets?|research|dataset|data analysis)\b",
    "Crypto / web3": r"\b(crypto\w*|blockchain|web3|solidity|smart contracts?|defi|nft)\b",
    "Design / video / audio": r"\b(logo|graphic design|ui/ux|figma|video editing|animation|voice ?over|photoshop|illustrat\w+)\b",
    "Writing / translation": r"\b(copywriting|copywriter|blog posts?|articles?|translation|translator|proofreading|ghostwrit\w+)\b",
    "Marketing / sales": r"\b(seo|lead generation|cold email|social media|marketing|sales)\b",
}
niche = pd.DataFrame([{"topic": k, "share_of_postings": txt.str.contains(rx, regex=True).mean()} for k, rx in topics.items()])
niche = niche.sort_values("share_of_postings", ascending=False)
niche.to_csv(DATA / "niche_topic_shares.csv", index=False)
print(niche.round(3).to_string(index=False))
del t, txt

# 4. Robustness of the job-length effect to the observation window
rob = []
for lo in (0, 14):
    g = df[df["observed_days"] >= lo]
    for lab, gg in g.groupby("duration_label"):
        if len(gg) >= 100:
            rob.append({"observed_at_least_days": lo, "duration": lab, "postings": len(gg), "hire_rate": gg["hired"].mean()})
rob = pd.DataFrame(rob)
rob.to_csv(DATA / "robustness_duration_observation.csv", index=False)
print(rob.round(3).to_string(index=False))
