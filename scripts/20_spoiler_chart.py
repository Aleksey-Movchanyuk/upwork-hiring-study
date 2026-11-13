"""Step 20: the opening chart of the Ukrainian article (DOU): how rare a hire is, and what the checklist saves.

Left: 100 postings as squares, hired (orange) against not hired (grey), from the whole public CSV.
Right: 30 applications, picked at random against picked by the checklist (top 30% by points); numbers from public/results/main_table.csv (test period).
Output: figures/39_spoiler_hire_share.png (English labels, like all charts)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import ART, FIG

ORANGE, GREY = "#E8833A", "#D9D9D9"
df = pd.read_csv(ART / "public" / "upwork_features_public.csv")
n, hired = len(df), int(df["hired"].sum())
share = hired / n
tab = pd.read_csv(ART / "public" / "results" / "main_table.csv").set_index("method")
rate_random = float(tab.loc["Coin flip", "top30_hire_rate"])
rate_check = float(tab.loc["Rule: seven-question checklist", "top30_hire_rate"])
usd_random = float(tab.loc["Coin flip", "usd_per_hire"])
usd_check = float(tab.loc["Rule: seven-question checklist", "usd_per_hire"])
saving = 1 - usd_check / usd_random
fmt = lambda x, d=1: f"{x:.{d}f}"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
k = int(round(share * 100))
for i in range(100):
    r, c = divmod(i, 10)
    a1.add_patch(plt.Rectangle((c, 9 - r), 0.86, 0.86, color=ORANGE if i < k else GREY))
a1.set_xlim(-0.1, 10); a1.set_ylim(-0.1, 10); a1.set_aspect("equal"); a1.axis("off")
a1.set_title(f"Only {k} of 100 postings end in a hire", fontsize=13, fontweight="bold", loc="left")
a1.text(0, -0.9, f"{hired:,} of {n:,} postings (orange: hired, grey: not hired)", fontsize=10, va="top")

labels = ["30 random postings", "30 postings with the\nbest checklist scores"]
vals = [rate_random * 30, rate_check * 30]
bars = a2.bar(labels, vals, color=[GREY, ORANGE], width=0.55)
for b, v, usd in zip(bars, vals, [usd_random, usd_check]):
    a2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{fmt(v)} hires\n${fmt(usd, 2)} per hire", ha="center", va="bottom", fontsize=11)
a2.set_ylim(0, max(vals) * 1.35); a2.set_ylabel("Hires per 30 applications"); a2.spines[["top", "right"]].set_visible(False)
a2.set_title(f"Checklist: {saving * 100:.0f}% fewer Connects per hire", fontsize=13, fontweight="bold", loc="left")
a2.text(0.5, -0.2, "Test period; 15 Connects at $0.10 per application (assumption)", transform=a2.transAxes, ha="center", fontsize=9, color="#555555")
fig.tight_layout()
fig.savefig(FIG / "39_spoiler_hire_share.png", dpi=140)
print(k, hired, n, vals, usd_random, usd_check, round(saving, 3))
