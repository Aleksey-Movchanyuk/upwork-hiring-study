"""Step 16: checks an editor would ask for before publication.

1. An honest checklist: the seven questions are fixed in advance, but every cut-off (how many past hires, how
   short a project, how many words) is tuned on the OLDEST 70% of postings only, then scored once on the latest 15%.
2. Paired bootstrap tests: is one method really better than another on the same test postings?
3. Competition: does the "big budgets are worth more" result survive dividing by the number of applicants?
4. Adjusted odds ratios (from 03_inference.py) in one small table for the article.
Outputs: data/checklist_*.csv, data/paired_tests.csv, data/money_vs_competition.csv, data/adjusted_odds_ratios_article.csv,
         data/checklist_test_scores.parquet (used by 15_method_comparison.py)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _common import DATA, load_features
from _features import SEED, bootstrap_auc

df = load_features()
df["hired"] = df["hired"].astype(int)
n = len(df)
i_tr, i_va = int(n * 0.70), int(n * 0.85)
tr, va, te = df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]
y = df["hired"].to_numpy()

# ------------------------------------------------------------------ 1. honest checklist
DUR_ORDER = {"Less than 1 week": 0, "Less than 1 month": 1, "1 to 3 months": 2, "3 to 6 months": 3, "More than 6 months": 4}
dur = df["duration_label"].map(DUR_ORDER)
hires = df["client_total_hires"].fillna(0)
words = df["description_word_count"]


def best_threshold(values, candidates, direction):
    """Pick the cut-off whose yes/no answer separates hires best on TRAINING postings only (by ROC AUC)."""
    best = None
    for c in candidates:
        ans = (values >= c) if direction == ">=" else (values <= c)
        a = ans.iloc[:i_tr]
        if a.nunique() < 2:
            continue
        auc = roc_auc_score(y[:i_tr], a.astype(int))
        if best is None or auc > best[1]:
            best = (c, auc)
    return best


t_hires = best_threshold(hires, [1, 2, 3, 5, 10, 20, 50, 100], ">=")
t_dur = best_threshold(dur.fillna(99), [1, 2, 3], "<=")
t_words = best_threshold(words, [100, 150, 200, 250, 300, 400, 500, 600, 800], "<=")
print("cut-offs chosen on the oldest 70%:", {"past hires >=": t_hires[0], "duration code <=": t_dur[0], "words <=": t_words[0]})
dur_names = {v: k for k, v in DUR_ORDER.items()}
items = {
    "Fixed-price job": df["job_type_label"] == "Fixed Price",
    "Client has spent money": df["client_total_spent"].fillna(0) > 0,
    f"Client has {t_hires[0]}+ past hires": hires >= t_hires[0],
    f"Project length up to '{dur_names[t_dur[0]]}'": dur.fillna(99) <= t_dur[0],
    "No crypto or web3 wording": ~df["has_crypto_term"],
    "No agency wording": ~df["agency_tone"],
    f"Description {t_words[0]} words or fewer": words <= t_words[0],
}
pts = sum(v.fillna(False).astype(int) for v in items.values())
df["points"] = pts
te = df.iloc[i_va:]   # refresh the slice so it includes the new column
bands = pd.cut(pts, [-1, 2, 3, 4, 5, 7], labels=["0 to 2", "3", "4", "5", "6 or 7"])
rows = []
for label, d in (("Oldest 70% (training)", df.iloc[:i_tr]), ("Next 15% (validation)", df.iloc[i_tr:i_va]), ("Latest 15% (test)", df.iloc[i_va:])):
    for b, g in d.groupby(bands.loc[d.index], observed=True):
        r, k = g["hired"].mean(), len(g)
        se = np.sqrt(r * (1 - r) / k)
        rows.append({"period": label, "points": b, "postings": k, "hire_rate": r, "lo": max(r - 1.96 * se, 0), "hi": r + 1.96 * se})
cl = pd.DataFrame(rows)
cl.to_csv(DATA / "checklist_honest_by_points.csv", index=False)
print(cl.round(3).to_string(index=False))
yt = te["hired"].to_numpy()
auc_c, lo_c, hi_c = bootstrap_auc(yt, te["points"].to_numpy().astype(float))
print(f"checklist test ROC AUC {auc_c:.3f} [{lo_c:.3f}, {hi_c:.3f}]")
pd.DataFrame({"item": list(items), "cut_off_tuned_on_training": ["", "", t_hires[0], dur_names[t_dur[0]], "", "", t_words[0]]}).to_csv(DATA / "checklist_honest_items.csv", index=False)
item_rows = []
for k, v in items.items():
    v = v.fillna(False).astype(bool)
    item_rows.append({"item": k, "share_yes_test": v.iloc[i_va:].mean(), "hire_yes_test": te["hired"][v.iloc[i_va:].to_numpy()].mean(), "hire_no_test": te["hired"][~v.iloc[i_va:].to_numpy()].mean(),
                      "hire_yes_train": df["hired"].iloc[:i_tr][v.iloc[:i_tr].to_numpy()].mean(), "hire_no_train": df["hired"].iloc[:i_tr][~v.iloc[:i_tr].to_numpy()].mean()})
pd.DataFrame(item_rows).to_csv(DATA / "checklist_honest_items_rates.csv", index=False)
print(pd.DataFrame(item_rows).round(3).to_string(index=False))
pd.DataFrame({"y": yt, "Rule: 7-point checklist": te["points"].to_numpy().astype(float)}).to_parquet(DATA / "checklist_test_scores.parquet", index=False)


# ------------------------------------------------------------------ 1b. checklist chosen AUTOMATICALLY from a pool of yes/no facts
POOL = {
    "Fixed-price job": df["job_type_label"] == "Fixed Price", "Client has spent money": df["client_total_spent"].fillna(0) > 0,
    "Client spent $1,000+": df["client_total_spent"].fillna(0) >= 1000, "Client spent $10,000+": df["client_total_spent"].fillna(0) >= 10000,
    "Client has 1+ past hires": hires >= 1, "Client has 10+ past hires": hires >= 10, "Client has 50+ past hires": hires >= 50,
    "Project up to 3 months": dur.fillna(99) <= 2, "Project under 1 month": dur.fillna(99) <= 1,
    "No crypto or web3 wording": ~df["has_crypto_term"], "No agency wording": ~df["agency_tone"], "No 'build something new' wording": ~df["new_build"],
    "Text not copied before": df["desc_prior_copies"].fillna(0) == 0, "No request to leave the platform": ~df["has_offplatform_contact"],
    "No em dash": df["em_dash_count"].fillna(0) == 0, "No long-term wording": ~df["long_term_wording"], "No screening prompt": ~df["screening_prompt"],
    "Description 400 words or fewer": words <= 400, "Description 200 words or fewer": words <= 200, "Title 60 characters or fewer": df["title_char_count"] <= 60,
    "Client in an English-speaking country": df["country_group"] == "Anglosphere", "Not a premium listing": ~df["premium"],
}
# oriented yes/no answers; orientation (which answer is "good") is decided on the training postings only
ORI = {}
for k, v in POOL.items():
    v = v.fillna(False).astype(bool)
    ORI[k] = (v if roc_auc_score(y[:i_tr], v.iloc[:i_tr].astype(int)) >= 0.5 else ~v).astype(int)
chosen, total = [], pd.Series(0, index=df.index)
for _ in range(7):   # forward selection: add the question that most improves the TRAINING AUC of the points
    best = None
    for k in ORI:
        if k in [c[0] for c in chosen]:
            continue
        a = roc_auc_score(y[:i_tr], (total + ORI[k]).iloc[:i_tr])
        if best is None or a > best[1]:
            best = (k, a)
    chosen.append(best); total = total + ORI[best[0]]
auto_pts = total
auc_a, lo_a, hi_a = bootstrap_auc(yt, auto_pts.iloc[i_va:].to_numpy().astype(float))
print("AUTO checklist (forward selection of 7 from", len(POOL), "facts, training only):", [k for k, a in chosen])
print(f"AUTO checklist test ROC AUC {auc_a:.3f} [{lo_a:.3f}, {hi_a:.3f}]  (training AUC {chosen[-1][1]:.3f})")
pd.DataFrame({"question": [k for k, a in chosen], "training_auc_after_adding": [a for k, a in chosen]}).to_csv(DATA / "checklist_auto_items.csv", index=False)
pd.DataFrame([{"test_auc": auc_a, "lo": lo_a, "hi": hi_a}]).to_csv(DATA / "checklist_auto_auc.csv", index=False)

# ------------------------------------------------------------------ 2. paired bootstrap tests
a = pd.read_parquet(DATA / "ml_test_scores.parquet")
b = pd.read_parquet(DATA / "ml_test_scores_stack.parquet")
assert (a["y"].to_numpy() == yt).all() and (b["y"].to_numpy() == yt).all()
S = {"Coin flip": a["Coin flip"], "Client has spent money": a["Rule: client has spent money"], "Upwork's warning signs": a["Rule: Upwork's warning signs (4 checks)"],
     "7-point checklist": te["points"].to_numpy().astype(float), "Decision tree (3 questions)": a["Learned rule: decision tree, depth 3"],
     "Logistic regression": a["Logistic regression"], "Neural network": a["Neural network (2 hidden layers)"],
     "Gradient boosting": a["CatBoost (gradient boosting)"], "Gradient boosting + word model": b["CatBoost + word model"]}
S = {k: np.asarray(v, dtype=float) for k, v in S.items()}
rng = np.random.default_rng(SEED)
tie = rng.random(len(yt))
K = int(round(len(yt) * 0.30))


def top_rate(s, idx):
    o = np.lexsort((tie[idx], -s[idx]))
    return yt[idx][o[:K]].mean()


PAIRS = [("Gradient boosting", "Decision tree (3 questions)"), ("Gradient boosting", "7-point checklist"), ("Gradient boosting", "Logistic regression"),
         ("Gradient boosting", "Neural network"), ("Neural network", "Logistic regression"), ("Gradient boosting + word model", "Gradient boosting"),
         ("Gradient boosting + word model", "Decision tree (3 questions)"), ("Decision tree (3 questions)", "Client has spent money"),
         ("7-point checklist", "Client has spent money"), ("Decision tree (3 questions)", "Upwork's warning signs"), ("Upwork's warning signs", "Client has spent money")]
B = 2000
draws = [rng.integers(0, len(yt), len(yt)) for _ in range(B)]
res = []
for p, q in PAIRS:
    da, dt = [], []
    for i in draws:
        if yt[i].min() == yt[i].max():
            continue
        da.append(roc_auc_score(yt[i], S[p][i]) - roc_auc_score(yt[i], S[q][i]))
        dt.append(top_rate(S[p], i) - top_rate(S[q], i))
    res.append({"method_A": p, "method_B": q, "auc_A": roc_auc_score(yt, S[p]), "auc_B": roc_auc_score(yt, S[q]),
                "auc_diff": np.mean(da), "auc_lo": np.percentile(da, 2.5), "auc_hi": np.percentile(da, 97.5), "share_A_better": float(np.mean(np.array(da) > 0)),
                "top30_diff_pp": 100 * np.mean(dt), "top30_lo_pp": 100 * np.percentile(dt, 2.5), "top30_hi_pp": 100 * np.percentile(dt, 97.5)})
pt = pd.DataFrame(res)
pt.to_csv(DATA / "paired_tests.csv", index=False)
print(pt.round(3).to_string(index=False))

# ------------------------------------------------------------------ 3. money vs competition
f = df[(df["job_type_label"] == "Fixed Price") & (df["budget_fixed"] > 0)].copy()
f["bucket"] = pd.cut(f["budget_fixed"], [0, 20, 80, 220, 650, 1e9], labels=["up to $20", "$20-80", "$80-220", "$220-650", "over $650"])
f["applicants"] = pd.to_numeric(f["final_total_applicants"], errors="coerce")
print("applicants known for", f["applicants"].notna().mean().round(3), "of fixed-price postings")
g = f.dropna(subset=["applicants"]).copy()
g["per_application"] = g["hired"] * g["budget_fixed"] / g["applicants"].clip(lower=1)
mc = g.groupby("bucket", observed=True).agg(postings=("hired", "size"), hire_rate=("hired", "mean"), median_budget=("budget_fixed", "median"),
                                              median_applicants=("applicants", "median"), mean_applicants=("applicants", "mean"),
                                              pay_per_application=("per_application", "mean")).reset_index()
mm_full = pd.read_csv(DATA / "market_money_by_budget.csv").set_index("bucket")
mc["hire_rate_all_postings"] = mc["bucket"].astype(str).map(mm_full["hire_rate"])
mc["market_money_per_posting"] = mc["hire_rate_all_postings"] * mc["median_budget"]
mc["money_per_applicant"] = mc["market_money_per_posting"] / mc["median_applicants"]   # if every applicant had an equal chance
mc.to_csv(DATA / "money_vs_competition.csv", index=False)
print(mc.round(2).to_string(index=False))

# ------------------------------------------------------------------ 4. adjusted odds ratios for the article
orr = pd.read_csv(DATA / "inference_logistic_odds_ratios.csv")
keep = {"Description copied 2+ times": "Text copied by others (2+ times)", "No client spend history": "Client has no spend history", "Crypto / web3 terms": "Crypto or web3 wording",
        "Agency / intermediary wording": "Agency wording", "Asks to build something new": "Asks to build something new", "Fixed-price job": "Fixed-price job",
        "Client past hires (log, per SD)": "Client past hires (per standard deviation)", "log10 words (per SD)": "Description length (per standard deviation)",
        "Em dash present": "Em dash present", "AI-marker dictionary score (per point)": "AI-marker score (per point)", "Title about fixing": "Title about fixing",
        "Client in Philippines": "Client in the Philippines", "Asks to leave platform": "Asks to leave the platform"}
o = orr[orr["term"].isin(keep)].copy(); o["term"] = o["term"].map(keep)
o = o[["term", "odds_ratio", "ci_lo", "ci_hi", "p_value"]]
o.to_csv(DATA / "adjusted_odds_ratios_article.csv", index=False)
print(o.round(3).to_string(index=False))

# ------------------------------------------------------------------ 5. figures for the article
from plotnine import (aes, element_text, facet_wrap, geom_col, geom_pointrange, geom_text, ggplot, labs, position_dodge, scale_color_manual,
                      scale_fill_manual, scale_y_continuous, theme)
from _common import BLUE, ORANGE, base_theme, pct_labels, save

cl2 = cl[cl["period"].isin(["Oldest 70% (training)", "Latest 15% (test)"])].copy()
cl2["period"] = cl2["period"].map({"Oldest 70% (training)": "Oldest 70% (cut-offs tuned here)", "Latest 15% (test)": "Latest 15% (never used for tuning)"})
p = (
    ggplot(cl2, aes("points", "hire_rate", color="period"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), position=position_dodge(width=0.45), size=0.7)
    + scale_color_manual({"Oldest 70% (cut-offs tuned here)": BLUE, "Latest 15% (never used for tuning)": ORANGE})
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="The more questions you answer yes to, the more often the post hires",
           subtitle="Hire rate by points on the seven-question checklist, with 95% ranges",
           x="Points (out of 7)", y="Hire rate", color="")
    + base_theme()
)
save(p, "32_checklist_points", 8.5, 4.6)

m = mc.copy()
long = pd.concat([
    pd.DataFrame({"bucket": m["bucket"], "measure": "Per posting: hire rate x median budget ($)", "value": m["market_money_per_posting"]}),
    pd.DataFrame({"bucket": m["bucket"], "measure": "Per applicant: the same, divided by the median number of applicants ($)", "value": m["money_per_applicant"]}),
])
long["bucket"] = pd.Categorical(long["bucket"].astype(str), categories=["up to $20", "$20-80", "$80-220", "$220-650", "over $650"], ordered=True)
long["label"] = long["value"].map(lambda v: f"${v:,.0f}" if v >= 10 else f"${v:,.2f}")
p = (
    ggplot(long, aes("bucket", "value", fill="measure"))
    + geom_col(width=0.65, show_legend=False)
    + geom_text(aes(label="label"), va="bottom", nudge_y=0.02, size=8.5)
    + facet_wrap("measure", ncol=2, scales="free_y")
    + scale_fill_manual(["#3366CC", "#E8843C"])
    + labs(title="Big jobs are worth more per application, even after counting the competition",
           subtitle="Fixed-price jobs by budget. Assumes every applicant has an equal chance, which is generous for big jobs",
           x="Fixed-price budget", y="")
    + base_theme()
    + theme(axis_text_x=element_text(rotation=20, ha="right"), strip_text=element_text(size=8.5))
)
save(p, "34_money_vs_competition", 10.5, 4.8)
