"""Step 3: Statistical inference.

1. Two-proportion comparisons for each hypothesis (risk difference with a
   Newcombe 95% interval, chi-square p-value, Benjamini-Hochberg correction).
2. A multivariable logistic regression, so effects are adjusted for each other
   (e.g. is "crypto" just a proxy for "new client"?).
3. A bootstrap interval for the em-dash difference.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from plotnine import (
    aes, element_text, geom_errorbarh, geom_histogram, geom_point, geom_vline, ggplot,
    labs, scale_color_manual, scale_x_continuous, scale_x_log10, theme,
)
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_chisquare

from _common import BLUE, DATA, ORANGE, base_theme, load_features, save

df = load_features()
rng = np.random.default_rng(42)

# 1. Hypotheses as binary flags
hyp = {
    "Crypto / web3 terms": df["has_crypto_term"],
    "Mentions wallet / seed / key": df["has_wallet_term"],
    "Asks to leave the platform (Telegram, WhatsApp...)": df["has_offplatform_contact"],
    "Description copied 2+ times": df["desc_dup_count"] > 1,
    "Contains an em dash": df["em_dash_count"] > 0,
    "Contains a URL": df["desc_has_url"],
    "Client has no spend history": df["client_total_spent"].fillna(0) == 0,
    "Payment method verified": df["client_payment_verification_status"] == 1,
    "Top client badge": df["client_top_client"],
    "Premium / enterprise listing": df["premium"] | df["enterprise_job"],
    "Fixed-price (vs hourly)": df["job_type_label"] == "Fixed Price",
    "Long description (above median words)": df["description_word_count"] > df["description_word_count"].median(),
    "Client in the Philippines": df["client_country"] == "Philippines",
    "Client in Anglosphere (US/UK/AU/CA/NZ)": df["country_group"] == "Anglosphere",
    "Client in Europe": df["country_group"] == "Europe",
    "Client in the Gulf": df["country_group"] == "Gulf",
    "Client in Latin America": df["country_group"] == "Latin America",
    "Client in East Asia": df["country_group"] == "East Asia",
    "2+ AI-style signals": df["ai_style_signals"] >= 2,
    # What the client asks for (content type)
    "Title is about fixing (fix / bug / debug)": df["fix_title"],
    "Asks to build something new (from scratch, MVP...)": df["new_build"],
    "Agency / intermediary wording (our client, white-label)": df["agency_tone"],
    "Long-term / ongoing / full-time wording": df["long_term_wording"],
    "Asks to start the proposal with a phrase / answer questions": df["screening_prompt"],
    "Quick / small / one-off task wording": df["quick_task_wording"],
    "Many AI markers (dictionary score 3+)": df["ai_marker_score"] >= 3,
    "Crypto posting from the Philippines": df["has_crypto_term"] & (df["client_country"] == "Philippines"),
}

rows = []
for name, flag in hyp.items():
    flag = flag.fillna(False).astype(bool)
    x1, n1 = int(df.loc[flag, "hired"].sum()), int(flag.sum())
    x0, n0 = int(df.loc[~flag, "hired"].sum()), int((~flag).sum())
    if min(n1, n0) < 100:
        continue
    lo, hi = confint_proportions_2indep(x1, n1, x0, n0, method="newcomb", compare="diff")
    chi2, p, _ = proportions_chisquare([x1, x0], [n1, n0])[:3]
    rows.append({
        "hypothesis": name, "n_with": n1, "rate_with": x1 / n1, "n_without": n0,
        "rate_without": x0 / n0, "diff_pp": 100 * (x1 / n1 - x0 / n0),
        "ci_lo_pp": 100 * lo, "ci_hi_pp": 100 * hi, "p_value": p,
    })
res = pd.DataFrame(rows)
res["p_adj_bh"] = multipletests(res["p_value"], method="fdr_bh")[1]
res["significant"] = np.where(res["p_adj_bh"] < 0.05, "significant (BH 5%)", "not significant")
res = res.sort_values("diff_pp").reset_index(drop=True)
res.to_csv(DATA / "inference_two_proportion.csv", index=False)
print(res.round(4).to_string())

res["hypothesis"] = pd.Categorical(res["hypothesis"], categories=res["hypothesis"], ordered=True)
p = (
    ggplot(res, aes(y="hypothesis", x="diff_pp", color="significant"))
    + geom_errorbarh(aes(xmin="ci_lo_pp", xmax="ci_hi_pp"), height=0.25)
    + geom_point(size=2.5)
    + geom_vline(xintercept=0, linetype="dashed")
    + scale_color_manual({"significant (BH 5%)": ORANGE, "not significant": "#7A7A7A"})
    + labs(title="Which signals move the hire rate?",
           subtitle="Difference in hire rate (percentage points) with vs without the signal; 95% Newcombe interval",
           x="Difference in hire rate (pp)", y="", color="")
    + base_theme()
)
save(p, "13_forest_risk_differences", 10, 8)

# 2. Multivariable logistic regression (adjusted odds ratios)
X = pd.DataFrame({
    "Crypto / web3 terms": df["has_crypto_term"].astype(int),
    "Wallet / seed / key terms": df["has_wallet_term"].astype(int),
    "Asks to leave platform": df["has_offplatform_contact"].astype(int),
    "Description copied 2+ times": (df["desc_dup_count"] > 1).astype(int),
    "Em dash present": (df["em_dash_count"] > 0).astype(int),
    "No client spend history": (df["client_total_spent"].fillna(0) == 0).astype(int),
    "Fixed-price job": (df["job_type_label"] == "Fixed Price").astype(int),
    "Client in Philippines": (df["client_country"] == "Philippines").astype(int),
    "AI-style signals (per signal)": df["ai_style_signals"],
    "AI-marker dictionary score (per point)": df["ai_marker_score"],
    "Title about fixing": df["fix_title"].astype(int),
    "Asks to build something new": df["new_build"].astype(int),
    "Agency / intermediary wording": df["agency_tone"].astype(int),
    "Long-term / ongoing wording": df["long_term_wording"].astype(int),
    "Screening prompt in the post": df["screening_prompt"].astype(int),
    "Quick / small task wording": df["quick_task_wording"].astype(int),
    "Crypto and Philippines together": (df["has_crypto_term"] & (df["client_country"] == "Philippines")).astype(int),
})
for g in ["Europe", "Gulf", "Latin America", "East Asia", "South Asia", "South-East Asia", "Africa"]:
    X[f"Region: {g} (vs Anglosphere)"] = (df["country_group"] == g).astype(int)
for name, series in {
    "log10 words (per SD)": np.log10(df["description_word_count"] + 1),
    "log10 client spend (per SD)": np.log10(df["client_total_spent"].fillna(0) + 1),
    "Client past hires (log, per SD)": np.log1p(df["client_total_hires"].fillna(0)),
}.items():
    X[name] = (series - series.mean()) / series.std()

model = sm.Logit(df["hired"], sm.add_constant(X)).fit(disp=0)
ci = model.conf_int()
orr = pd.DataFrame({
    "term": model.params.index, "odds_ratio": np.exp(model.params.values),
    "ci_lo": np.exp(ci[0].values), "ci_hi": np.exp(ci[1].values), "p_value": model.pvalues.values,
}).query("term != 'const'").sort_values("odds_ratio").reset_index(drop=True)
orr["significant"] = np.where(orr["p_value"] < 0.05, "p < 0.05", "p >= 0.05")
orr.to_csv(DATA / "inference_logistic_odds_ratios.csv", index=False)
print(f"\nLogit pseudo R2 = {model.prsquared:.3f}, n = {int(model.nobs):,}")
print(orr.round(3).to_string())

orr["term"] = pd.Categorical(orr["term"], categories=orr["term"], ordered=True)
p = (
    ggplot(orr, aes(y="term", x="odds_ratio", color="significant"))
    + geom_errorbarh(aes(xmin="ci_lo", xmax="ci_hi"), height=0.25)
    + geom_point(size=2.5)
    + geom_vline(xintercept=1, linetype="dashed")
    + scale_x_log10()
    + scale_color_manual({"p < 0.05": ORANGE, "p >= 0.05": "#7A7A7A"})
    + labs(title="Adjusted odds ratios from a logistic regression",
           subtitle="Odds of a recorded hire, all signals fitted together; 95% CI, log scale",
           x="Odds ratio (1 = no effect)", y="", color="")
    + base_theme()
)
save(p, "14_forest_odds_ratios", 9.5, 8)

# 3. Bootstrap for the em-dash gap
a = df.loc[df["em_dash_count"] > 0, "hired"].to_numpy()
b = df.loc[df["em_dash_count"] == 0, "hired"].to_numpy()
boot = np.array([
    rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean() for _ in range(5000)
]) * 100
lo, hi = np.percentile(boot, [2.5, 97.5])
p = (
    ggplot(pd.DataFrame({"diff": boot}), aes("diff"))
    + geom_histogram(bins=50, fill=BLUE, color="white")
    + geom_vline(xintercept=0, linetype="dashed")
    + geom_vline(xintercept=[lo, hi], color=ORANGE, size=1)
    + labs(title="Bootstrap distribution of the em-dash gap",
           subtitle=f"5,000 resamples; 95% interval [{lo:.1f}, {hi:.1f}] pp (orange); 0 = no difference",
           x="Hire rate with em dash minus without (pp)", y="Resamples")
    + base_theme()
)
save(p, "15_bootstrap_em_dash", 7.5, 4.5)

# Other tests quoted in the text
words_h = df.loc[df["hired"] == 1, "description_word_count"]
words_n = df.loc[df["hired"] == 0, "description_word_count"]
mw = stats.mannwhitneyu(words_h, words_n)
spent = df["client_total_spent"].fillna(0)
rho, rho_p = stats.spearmanr(spent, df["hired"])
summary = {
    "em_dash_boot_ci_pp": [round(lo, 2), round(hi, 2)],
    "mannwhitney_words_p": float(mw.pvalue),
    "median_words_hired": float(words_h.median()),
    "median_words_not_hired": float(words_n.median()),
    "spearman_spend_vs_hired": [round(float(rho), 3), float(rho_p)],
    "logit_pseudo_r2": round(float(model.prsquared), 4),
    "n": int(model.nobs),
}
(DATA / "inference_summary.json").write_text(json.dumps(summary, indent=2))
print(summary)
