"""Step 15: one dashboard that compares all the methods (rules and models) on the latest 15% of postings.

Reads the test-set scores saved by 11_rules_vs_models.py and 12_ngrams.py (no retraining).
Panels: A) ROC curves  B) cumulative gains  C) hire rate in the top 30%  D) Connects cost per recorded hire.
Colours show the family: grey = coin flip, orange = rules, blue = models, green = best (facts + word model).
Output: figures/33_method_comparison.png (simple, for the article), figures/33b_method_comparison_full.png (all nine methods) and data/method_comparison.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve

from _common import DATA, FIG
from _features import SEED, bootstrap_auc

CONNECTS, USD, SHARE = 15, 0.10, 0.30
a = pd.read_parquet(DATA / "ml_test_scores.parquet")
b = pd.read_parquet(DATA / "ml_test_scores_stack.parquet")
c = pd.read_parquet(DATA / "checklist_test_scores.parquet")   # made by 16_publication_checks.py
y = a["y"].to_numpy().astype(int)
assert (y == b["y"].to_numpy()).all() and (y == c["y"].to_numpy()).all()

# (column, label shown, family colour, line style)
GREY, ORANGE, RED, BLUE_L, BLUE_M, BLUE_D, GREEN = "#8A8A8A", "#F3A34A", "#D9531E", "#8DB5E8", "#4F8FD8", "#1F4FA3", "#1E9E5A"
SPEC = [
    ("Coin flip", "Coin flip", GREY, "--", a),
    ("Rule: client has spent money", "Rule: client has spent money", "#F6C27A", "-", a),
    ("Rule: Upwork's warning signs (4 checks)", "Rule: Upwork's warning signs", ORANGE, "-", a),
    ("Rule: 7-point checklist", "Rule: 7-question checklist", "#B8660C", "-", c),
    ("Learned rule: decision tree, depth 3", "Decision tree (3 questions deep)", RED, "-", a),
    ("Logistic regression", "Logistic regression", BLUE_L, "-", a),
    ("Neural network (2 hidden layers)", "Neural network", BLUE_M, "-", a),
    ("CatBoost (gradient boosting)", "Gradient boosting (CatBoost)", BLUE_D, "-", a),
    ("CatBoost + word model", "Gradient boosting + word model", GREEN, "-", b),
]
rng = np.random.default_rng(SEED)
tie = rng.random(len(y))  # one fixed random tie-break, shared by all methods


def order(s):
    return np.lexsort((tie, -np.asarray(s, dtype=float)))


rows, curves = [], {}
for col, label, color, ls, src in SPEC:
    s = src[col].to_numpy().astype(float)
    o = order(s)
    k = int(round(len(y) * SHARE))
    hires_top = y[o[:k]].sum()
    gains_x = np.arange(1, len(y) + 1) / len(y)
    gains_y = np.cumsum(y[o]) / y.sum()
    fpr, tpr, _ = roc_curve(y, s)
    auc = 0.5 if col == "Coin flip" else roc_auc_score(y, s)
    # bootstrap interval for the hire rate in the top 30% (rank-based, same tie-break)
    boots = []
    for _ in range(400):
        i = rng.integers(0, len(y), len(y))
        oi = np.lexsort((tie[i], -s[i]))
        boots.append(y[i][oi[:k]].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if col == "Coin flip":  # no information: by definition the top 30% is a random 30%, so use the exact base rate
        hires_top = y.mean() * k
    rows.append({"method": label, "family": color, "roc_auc": auc, "top30_hire_rate": hires_top / k, "lo": lo, "hi": hi,
                 "hires_kept": hires_top / y.sum(), "usd_per_hire": k * CONNECTS * USD / hires_top})
    curves[label] = (fpr, tpr, gains_x, gains_y, color, ls)
res = pd.DataFrame(rows)
res.drop(columns="family").to_csv(DATA / "method_comparison.csv", index=False)
base = y.mean()

plt.style.use("ggplot")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.labelsize": 10.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.facecolor": "#EBEBEB", "figure.facecolor": "white"})
fig = plt.figure(figsize=(16, 10.6))
gs = fig.add_gridspec(2, 2, left=0.235, right=0.985, top=0.835, bottom=0.075, wspace=0.17, hspace=0.34)
axA, axB, axC, axD = (fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1))
HI = ("Gradient boosting + word model", "Decision tree (3 questions deep)")

# A) ROC
for label, (fpr, tpr, gx, gy, color, ls) in curves.items():
    axA.plot(fpr, tpr, color=color, ls=ls, lw=2.7 if label in HI else 1.7, alpha=0.95)
axA.set_title("A. ROC curve: catching hires without chasing dead posts")
axA.set_xlabel("Share of non-hiring postings I'd apply to (lower is better)")
axA.set_ylabel("Share of hiring postings caught")
axA.set_xlim(0, 1); axA.set_ylim(0, 1.01)

# B) gains
for label, (fpr, tpr, gx, gy, color, ls) in curves.items():
    axB.plot(np.r_[0, gx], np.r_[0, gy], color=color, ls=ls, lw=2.7 if label in HI else 1.7, alpha=0.95)
axB.axvline(SHARE, color="#333333", ls=":", lw=1.3)
best = res.iloc[-1]; tree = res[res.method.str.startswith("Decision tree")].iloc[0]; chk = res[res.method.str.contains("checklist")].iloc[0]
axB.annotate(f"Apply to the best-scored 30%:\nword + boosting keeps {best.hires_kept:.1%} of all hires,\n"
             f"the 7-question checklist {chk.hires_kept:.1%}, the 3-question tree {tree.hires_kept:.1%},\na coin flip 30%",
             xy=(SHARE, best.hires_kept), xytext=(0.43, 0.17), fontsize=9.2, color="#222222",
             arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1), bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB"))
axB.set_title("B. Cumulative gains: how many hires you keep")
axB.set_xlabel("Share of postings applied to (best-scored first)")
axB.set_ylabel("Share of all hires caught")
axB.set_xlim(0, 1); axB.set_ylim(0, 1.01)
axB.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); axB.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

# C and D: horizontal bars, same order and colours; method names only on C
def bars(ax, col, title, xlabel, fmt, baseline, err=False, names=True):
    d = res.sort_values(col, ascending=(col == "top30_hire_rate"))
    ypos = np.arange(len(d))
    ax.barh(ypos, d[col], color=d["family"], edgecolor="white", height=0.72)
    if err:
        ax.errorbar(d[col], ypos, xerr=[d[col] - d["lo"], d["hi"] - d[col]], fmt="none", ecolor="#333333", elinewidth=1.1, capsize=2.5)
    pad = 0.012 if err else 0.09
    for yp, v, h in zip(ypos, d[col], d["hi"] if err else d[col]):
        ax.text(h + pad, yp, fmt(v), va="center", ha="left", fontsize=9.4, fontweight="bold", color="#222222")
    ax.set_yticks(ypos)
    ax.set_yticklabels(d["method"] if names else [], fontsize=9.4)
    ax.set_ylim(-0.6, len(d) + 0.25)
    ax.axvline(baseline[0], color="#333333", ls="--", lw=1.2)
    ax.text(baseline[0] + pad, len(d) - 0.05, baseline[1], ha="left", va="center", fontsize=8.8, color="#333333")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)


bars(axC, "top30_hire_rate", "C. Hire rate if you only apply to the top 30%", "Share of those postings that recorded a hire (95% range)",
     lambda v: f"{v:.1%}", (base, f"all postings: {base:.1%}"), err=True)
axC.set_xlim(0, 0.64); axC.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
bars(axD, "usd_per_hire", "D. Connects spent per recorded hire", "US dollars (15 Connects per proposal at $0.10 each)",
     lambda v: f"${v:.2f}", (res.usd_per_hire.iloc[0], f"coin flip: ${res.usd_per_hire.iloc[0]:.2f}"), names=False)
axD.set_xlim(0, 7.6)

# colour key in the free strip on the left of the top row
handles = [Line2D([0], [0], color=c, lw=3.2, ls=ls) for _, _, c, ls, _ in SPEC]
fig.legend(handles, [f"{r.method}\nROC AUC {r.roc_auc:.3f}" for r in res.itertuples()], loc="upper left", bbox_to_anchor=(0.008, 0.825),
           title="Method and its ROC AUC", frameon=True, fontsize=9.0, title_fontsize=9.6, labelspacing=0.75, handlelength=2.2,
           facecolor="white", edgecolor="#CCCCCC")
fig.text(0.008, 0.965, "Simple rules vs machine learning: nine ways to pick Upwork jobs", fontsize=19.5, fontweight="bold", ha="left")
fig.text(0.008, 0.925, "Latest 15% of the 11,507 postings (1,727 postings, 25.2% recorded a hire). Every method scores each posting, and I 'apply' to the best-scored ones.\n"
         "Orange = rules, blue = models, green = best, grey = coin flip. ROC AUC: 0.5 is a coin flip, 1.0 is perfect. The checklist's cut-offs were tuned on the oldest 70% of postings only.",
         fontsize=10.2, ha="left", va="top", color="#333333", linespacing=1.55)
fig.savefig(FIG / "33b_method_comparison_full.png", dpi=170)   # the full nine-method dashboard (for the appendix)
print(res.drop(columns="family").round(3).to_string(index=False))
print("saved figures/33b_method_comparison_full.png")


# ----------------------------------------------------------------------------------------------------------
# SIMPLE VERSION FOR THE ARTICLE: only the methods a reader chooses between, direct labels, no legend
# ----------------------------------------------------------------------------------------------------------
pt = pd.read_csv(DATA / "paired_tests.csv").set_index(["method_A", "method_B"])
gap_cb = pt.loc[("Gradient boosting", "7-point checklist")]
gap_rule = pt.loc[("7-point checklist", "Client has spent money")]
KEEP = [("Coin flip", "Coin flip (no skill)", "#8A8A8A"),
        ("Rule: client has spent money", "One question: has the client spent money?", "#F3B66B"),
        ("Rule: Upwork's warning signs", "Upwork's own four warning signs", "#E8943A"),
        ("Rule: 7-question checklist", "Seven-question checklist", "#B8660C"),
        ("Gradient boosting (CatBoost)", "Gradient boosting (machine learning)", "#1F4FA3"),
        ("Gradient boosting + word model", "Gradient boosting + word model", "#1E9E5A")]
sel = res.set_index("method").loc[[k for k, _, _ in KEEP]].reset_index()
sel["label"] = [lab for _, lab, _ in KEEP]
sel["color"] = [c for _, _, c in KEEP]

plt.rcParams.update({"axes.titlesize": 15, "axes.labelsize": 12.5, "xtick.labelsize": 11.5, "ytick.labelsize": 12.5})
fig2 = plt.figure(figsize=(15.5, 7.0))
g2 = fig2.add_gridspec(1, 2, width_ratios=[1.5, 1], left=0.285, right=0.975, top=0.78, bottom=0.19, wspace=0.40)
a1 = fig2.add_subplot(g2[0, 0]); a2 = fig2.add_subplot(g2[0, 1])

ypos = np.arange(len(sel))[::-1]            # best method at the top
a1.barh(ypos, sel["top30_hire_rate"], color=sel["color"], edgecolor="white", height=0.66)
a1.errorbar(sel["top30_hire_rate"], ypos, xerr=[sel["top30_hire_rate"] - sel["lo"], sel["hi"] - sel["top30_hire_rate"]],
            fmt="none", ecolor="#333333", elinewidth=1.3, capsize=3)
for yp, r in zip(ypos, sel.itertuples()):
    a1.text(r.hi + 0.012, yp, f"{r.top30_hire_rate:.0%}", va="center", ha="left", fontsize=14, fontweight="bold", color="#1a1a1a")
    a1.text(0.755, yp, f"${r.usd_per_hire:.2f}", va="center", ha="right", fontsize=12.5, color="#444444")
a1.text(0.755, len(sel) - 0.45, "Connects per\nrecorded hire", va="bottom", ha="right", fontsize=11, color="#444444", linespacing=1.15)
a1.set_yticks(ypos); a1.set_yticklabels(sel["label"])
a1.axvline(base, color="#333333", ls="--", lw=1.3)
a1.text(base + 0.006, -0.78, f"all postings: {base:.0%}", fontsize=11, color="#333333", va="center")
a1.set_xlim(0, 0.76); a1.set_ylim(-1.05, len(sel) + 0.55)
a1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
a1.set_xlabel("Share of the chosen postings that recorded a hire")
a1.set_title("A. Hire rate in the best-scored 30%", loc="left", fontweight="bold")
a1.grid(axis="y", visible=False)

for label, name, color, lw, ls in [("Coin flip", "Coin flip", "#8A8A8A", 2.2, "--"),
                                   ("Rule: 7-question checklist", "Checklist", "#B8660C", 3.2, "-"),
                                   ("Gradient boosting + word model", "Best model", "#1E9E5A", 3.2, "-")]:
    fpr, tpr, gx, gy, _, _ = curves[label if label != "Coin flip" else "Coin flip"]
    a2.plot(np.r_[0, gx], np.r_[0, gy], color=color, lw=lw, ls=ls)
a2.axvline(SHARE, color="#333333", ls=":", lw=1.3)
kept = res.set_index("method")["hires_kept"]
for name, key, color, dy in [("Best model", "Gradient boosting + word model", "#1E9E5A", 0.04), ("Checklist", "Rule: 7-question checklist", "#B8660C", -0.045),
                             ("Coin flip", "Coin flip", "#5f5f5f", -0.045)]:
    v = kept[key]
    a2.scatter([SHARE], [v], color=color, s=70, zorder=5, edgecolor="white", linewidth=1.2)
    a2.text(SHARE + 0.035, v + dy, f"{name}: {v:.1%}", color=color, fontsize=12.5, fontweight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
a2.set_xlim(0, 1); a2.set_ylim(0, 1.01)
a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
a2.set_xlabel("Share of postings applied to\n(best-scored first)"); a2.set_ylabel("Share of all hires you keep")
a2.set_title("B. Hires you keep", loc="left", fontweight="bold")

fig2.text(0.015, 0.955, "A seven-question checklist gets most of the way to a machine-learning model", fontsize=20, fontweight="bold", ha="left")
fig2.text(0.015, 0.905, "Test on the latest 15% of postings (1,727), which no method saw while it was built. Every method scores each posting;\n"
          "I apply only to the 30% it scores highest. Whiskers show 95% ranges.", fontsize=12, ha="left", va="top", color="#333333", linespacing=1.55)
fig2.text(0.015, 0.045, f"Gradient boosting (without words) beats the checklist by {gap_cb.top30_diff_pp:.1f} points of hire rate (95% range {gap_cb.top30_lo_pp:.1f} to {gap_cb.top30_hi_pp:.1f}).   "
          f"The checklist beats the one-question rule by {gap_rule.top30_diff_pp:.1f} points ({gap_rule.top30_lo_pp:.1f} to {gap_rule.top30_hi_pp:.1f}).",
          fontsize=11.8, ha="left", color="#222222")
fig2.savefig(FIG / "33_method_comparison.png", dpi=170)
print("saved figures/33_method_comparison.png (simple version)")
