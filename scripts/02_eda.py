"""Step 2: Exploratory data analysis, following the principles of analytic graphics
from the JHU Exploratory Data Analysis course: show comparisons, show
multivariate data, integrate evidence, describe and document.

Every chart is a ggplot2-style (plotnine) figure saved to article/figures/.
"""
import numpy as np
import pandas as pd
from plotnine import (
    aes, coord_flip, element_text, facet_wrap, geom_bar, geom_col, geom_density,
    geom_histogram, geom_hline, geom_line, geom_linerange, geom_point,
    geom_pointrange, geom_ribbon, geom_smooth, geom_text, geom_tile, ggplot,
    labs, scale_color_manual, scale_fill_gradient, scale_fill_manual,
    scale_x_continuous, scale_x_log10, scale_y_continuous, theme,
)

from _common import (
    BLUE, HIRED_COLORS, ORANGE, base_theme, load_features, pct_labels, rate_table, save,
)

df = load_features()
BASE = df["hired"].mean()
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
N = len(df)
OBS = df["observed_days"].median()


def hline_base():
    return geom_hline(yintercept=BASE, linetype="dashed", color="#444444")


# 1. Overall picture: how many postings record a hire at all?
overall = df["outcome"].value_counts().rename_axis("outcome").reset_index(name="n")
overall["share"] = overall["n"] / N
overall["label"] = overall.apply(lambda r: f"{r['n']:,}\n({r['share']:.1%})", axis=1)
p = (
    ggplot(overall, aes("outcome", "n", fill="outcome"))
    + geom_col(width=0.6, show_legend=False)
    + geom_text(aes(label="label"), va="bottom", nudge_y=150, size=10)
    + scale_fill_manual(HIRED_COLORS)
    + scale_y_continuous(expand=(0, 0, 0.15, 0))
    + labs(title="Four out of five postings never record a hire",
           subtitle=f"{N:,} Upwork postings with a resolved outcome (median observation {OBS:.0f} days)",
           x="", y="Postings")
    + base_theme()
)
save(p, "01_overall_outcome", 6, 4.5)

# 2. Time of posting: day of week x hour (UTC), hire rate heatmap
hm = rate_table(df, ["published_dow", "published_hour"])
hm = hm[hm["n"] >= 40].copy()
hm["day"] = pd.Categorical(hm["published_dow"].map(dict(enumerate(DAYS))), categories=DAYS[::-1])
p = (
    ggplot(hm, aes("published_hour", "day", fill="rate"))
    + geom_tile(color="white")
    + scale_fill_gradient(low="#FFF3E0", high="#C0392B", labels=pct_labels, name="Hire rate")
    + scale_x_continuous(breaks=range(0, 24, 3))
    + labs(title="No clear weekday or hour pattern",
           subtitle="Hire rate by weekday and hour posted (UTC); cells with fewer than 40 postings hidden",
           x="Hour of day (UTC)", y="")
    + base_theme()
)
save(p, "02_hire_rate_weekday_hour", 8, 4.2)

# 3. Client spend history (comparison)
spent = df["client_total_spent"].fillna(0)
df["spend_bucket"] = pd.cut(
    spent, [-1, 0, 1_000, 10_000, 100_000, np.inf],
    labels=["$0 / new", "< $1k", "$1k-10k", "$10k-100k", "$100k+"],
)
t = rate_table(df, "spend_bucket")
p = (
    ggplot(t, aes("spend_bucket", "rate"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), color=BLUE, size=0.8)
    + hline_base()
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="Clients with no spending history hire half as often",
           subtitle="Hire rate with 95% Wilson interval; dashed line is the overall rate",
           x="Client total spent on Upwork", y="Hire rate")
    + base_theme()
)
save(p, "03_hire_rate_client_spend", 7, 4.5)

# 4. Client country (top 15 by volume)
top = df["client_country"].value_counts().head(15).index
t = rate_table(df[df["client_country"].isin(top)], "client_country").sort_values("rate")
t["client_country"] = pd.Categorical(t["client_country"], categories=t["client_country"], ordered=True)
p = (
    ggplot(t, aes("client_country", "rate"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), color=BLUE)
    + hline_base()
    + coord_flip()
    + scale_y_continuous(labels=pct_labels)
    + labs(title="Hire rate by client country",
           subtitle="15 most frequent client countries (ISO codes merged), 95% Wilson interval",
           x="", y="Hire rate")
    + base_theme()
)
save(p, "04_hire_rate_country", 7, 5.5)

# 5. Show multivariate data: budget vs competition, coloured by outcome (the slide's idea)
fx = df[(df["job_type_label"] == "Fixed Price") & (df["budget_fixed"] > 0)
        & df["final_total_applicants"].notna()].copy()
fx["log_budget"] = np.log10(fx["budget_fixed"])
fx["log_applicants"] = np.log10(fx["final_total_applicants"] + 1)
p = (
    ggplot(fx, aes("log_budget", "log_applicants", color="outcome", fill="outcome"))
    + geom_point(alpha=0.35, size=1.2)
    + geom_smooth(method="lm")
    + scale_color_manual(HIRED_COLORS)
    + scale_fill_manual(HIRED_COLORS)
    + labs(title="Budget, competition and outcome together",
           subtitle="Fixed-price jobs; linear fit with 95% band per outcome; applicants counted at the last check",
           x="log10(fixed budget, USD)", y="log10(applicants + 1)", color="", fill="")
    + base_theme()
)
save(p, "05_budget_vs_applicants_by_outcome", 8, 5)

# 6. Competition: applicants per job by outcome (integrate evidence)
ap = df[df["final_total_applicants"].notna()].copy()
ap["applicants_plus1"] = ap["final_total_applicants"] + 1
p = (
    ggplot(ap, aes("applicants_plus1", fill="outcome"))
    + geom_density(alpha=0.55, color="none")
    + scale_x_log10()
    + scale_fill_manual(HIRED_COLORS)
    + labs(title="Applicant counts look alike for hired and not-hired jobs",
           subtitle=f"Applicants per job at the last check (median {ap['final_total_applicants'].median():.0f})",
           x="Applicants + 1 (log scale)", y="Density", fill="")
    + base_theme()
)
save(p, "06_applicants_distribution", 7.5, 4.5)

# 7. The crypto and wallet hypothesis
def flag_rate(col, label_true, label_false, title, name, subtitle):
    d = df.copy()
    d["group"] = np.where(d[col], label_true, label_false)
    t = rate_table(d, "group")
    t["label"] = t.apply(lambda r: f"{r['rate']:.1%}\nn={int(r['n']):,}", axis=1)
    p = (
        ggplot(t, aes("group", "rate", fill="group"))
        + geom_col(width=0.55, show_legend=False)
        + geom_linerange(aes(ymin="lo", ymax="hi"))
        + geom_text(aes(label="label", y="hi"), va="bottom", nudge_y=0.01, size=9)
        + scale_fill_manual({label_true: ORANGE, label_false: "#7A9CC6"})
        + scale_y_continuous(labels=pct_labels, expand=(0, 0, 0.2, 0))
        + labs(title=title, subtitle=subtitle, x="", y="Hire rate")
        + base_theme()
    )
    save(p, name, 6, 4.5)


flag_rate("has_crypto_term", "Crypto / web3 terms", "No crypto terms",
          "Crypto postings rarely end in a hire", "07_crypto_vs_hire",
          "Share of postings with a recorded hire, 95% Wilson interval")
flag_rate("has_wallet_term", "Mentions wallet / seed / key", "No wallet terms",
          "Postings that mention wallets or keys hire even less", "08_wallet_vs_hire",
          "Share of postings with a recorded hire, 95% Wilson interval")

# 8. Re-posted ("arbitrage") listings: identical description seen several times
df["repost_group"] = pd.cut(df["desc_dup_count"], [0, 1, 3, 10, np.inf],
                            labels=["Unique", "2-3 copies", "4-10 copies", "11+ copies"])
t = rate_table(df, "repost_group")
p = (
    ggplot(t, aes("repost_group", "rate"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), color=BLUE, size=0.8)
    + hline_base()
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="Copy-pasted postings hire much less often",
           subtitle="Hire rate by how many times the same description was captured",
           x="Identical description seen...", y="Hire rate")
    + base_theme()
)
save(p, "09_reposted_listings", 7, 4.5)

# 9. Writing style: description length and the em-dash hypothesis
df["words_plus1"] = df["description_word_count"] + 1
p = (
    ggplot(df, aes("words_plus1", fill="outcome"))
    + geom_density(alpha=0.55, color="none")
    + scale_x_log10()
    + scale_fill_manual(HIRED_COLORS)
    + labs(title="Description length looks similar for both outcomes",
           subtitle="Words per description (log scale)", x="Words + 1", y="Density", fill="")
    + base_theme()
)
save(p, "10_description_length", 7.5, 4.5)

flag_rate_df = df.assign(em=np.where(df["em_dash_count"] > 0, "Has an em dash", "No em dash"))
t = rate_table(flag_rate_df, "em")
p = (
    ggplot(t, aes("em", "rate", fill="em"))
    + geom_col(width=0.55, show_legend=False)
    + geom_linerange(aes(ymin="lo", ymax="hi"))
    + geom_text(aes(label=[f"{r:.1%}\nn={n:,}" for r, n in zip(t["rate"], t["n"])], y="hi"),
                va="bottom", nudge_y=0.01, size=9)
    + scale_fill_manual({"Has an em dash": ORANGE, "No em dash": "#7A9CC6"})
    + scale_y_continuous(labels=pct_labels, expand=(0, 0, 0.2, 0))
    + labs(title="The em-dash 'AI tell': a small gap",
           subtitle="Share of postings with a recorded hire, 95% Wilson interval", x="", y="Hire rate")
    + base_theme()
)
save(p, "11_em_dash_vs_hire", 6, 4.5)

# 10. Is the pattern stable over time? Weekly hire rate
wk = df.assign(week=df["first_captured_at"].dt.tz_localize(None).dt.to_period("W").dt.start_time)
t = rate_table(wk, "week")
t = t[t["n"] >= 100]
p = (
    ggplot(t, aes("week", "rate"))
    + geom_ribbon(aes(ymin="lo", ymax="hi"), fill=BLUE, alpha=0.25)
    + geom_line(color=BLUE, size=1)
    + geom_point(color=BLUE)
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="Weekly hire rate by capture week",
           subtitle="Weeks with at least 100 labelled postings; sharp dips may reflect pipeline changes, not the market",
           x="", y="Hire rate")
    + base_theme()
    + theme(axis_text_x=element_text(rotation=30, ha="right"))
)
save(p, "12_weekly_hire_rate", 8, 4.5)

# 11. Hypothesis (experience): short text hires more, long text hires less
df["length_quintile"] = pd.qcut(df["description_word_count"], 5,
                                labels=["Shortest 20%", "20-40%", "40-60%", "60-80%", "Longest 20%"])
t = rate_table(df, "length_quintile")
edges = df["description_word_count"].quantile([0, .2, .4, .6, .8, 1]).round().astype(int).tolist()
p = (
    ggplot(t, aes("length_quintile", "rate"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), color=BLUE, size=0.8)
    + hline_base()
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="Shorter descriptions hire more often",
           subtitle=f"Hire rate by description-length quintile (word cut points: {edges})",
           x="Description length", y="Hire rate")
    + base_theme()
)
save(p, "10b_hire_rate_by_length", 7.5, 4.5)

# 11a. Same, in deciles: the pattern is a hump (very short posts are only average), not a straight line
dec, bins = pd.qcut(df["description_word_count"], 10, retbins=True, duplicates="drop")
bins = np.round(bins).astype(int)
dec_labels = [f"<={bins[1]}"] + [f"{bins[i] + 1}-{bins[i + 1]}" for i in range(1, len(bins) - 2)] + [f">{bins[-2]}"]
df["length_decile"] = pd.Categorical.from_codes(dec.cat.codes.to_numpy(), categories=dec_labels, ordered=True)
t = rate_table(df, "length_decile")
t["length_decile"] = pd.Categorical(t["length_decile"], categories=dec_labels, ordered=True)
t = t.sort_values("length_decile")
p = (
    ggplot(t, aes("length_decile", "rate", group=1))
    + geom_line(color=BLUE, alpha=0.5)
    + geom_pointrange(aes(ymin="lo", ymax="hi"), color=BLUE, size=0.7)
    + hline_base()
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="Hire rate peaks at 65-200 words, then falls",
           subtitle="Hire rate by description-length decile (words). The very shortest posts are only average",
           x="Description length (words)", y="Hire rate")
    + base_theme()
    + theme(axis_text_x=element_text(rotation=30, ha="right"))
)
save(p, "10c_hire_rate_by_length_decile", 8, 4.5)

# 12. Hypothesis (experience): client market groups
order = ["Anglosphere", "Europe", "Gulf", "Latin America", "East Asia", "South Asia",
         "South-East Asia", "Africa", "Other"]
t = rate_table(df[df["country_group"].isin(order)], "country_group")
t = t.sort_values("rate")  # ascending, so the highest hire rate ends up on top after coord_flip
t["country_group"] = pd.Categorical(t["country_group"], categories=t["country_group"], ordered=True)
t["label"] = t.apply(lambda r: f"n={int(r['n']):,}", axis=1)
p = (
    ggplot(t, aes("country_group", "rate"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), color=BLUE)
    + geom_text(aes(label="label", y="hi"), ha="left", nudge_y=0.012, size=8)
    + hline_base()
    + coord_flip()
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="Hire rate by client market group",
           subtitle="Sorted by hire rate. Anglosphere = US, UK, AU, CA, NZ; Gulf = Saudi, Qatar, Bahrain, UAE, Kuwait, Oman",
           x="", y="Hire rate")
    + base_theme()
)
save(p, "04b_hire_rate_by_market_group", 8, 5)

# 12b. The Philippines gap: is it the country or the crypto postings?
ph = df["client_country"].eq("Philippines")
cr = df["has_crypto_term"].fillna(False).astype(bool)
df["ph_crypto"] = np.select(
    [ph & cr, ph & ~cr, ~ph & cr],
    ["Philippines, crypto", "Philippines, other", "Other countries, crypto"],
    "Other countries, other")
t = rate_table(df, "ph_crypto").sort_values("rate")
t["ph_crypto"] = pd.Categorical(t["ph_crypto"], categories=t["ph_crypto"], ordered=True)
t["label"] = t.apply(lambda r: f"n={int(r['n']):,}", axis=1)
p = (
    ggplot(t, aes("ph_crypto", "rate"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), color=BLUE)
    + geom_text(aes(label="label", y="hi"), ha="left", nudge_y=0.012, size=8)
    + hline_base()
    + coord_flip()
    + scale_y_continuous(labels=pct_labels, limits=(0, 0.32))
    + labs(title="The Philippines gap sits mostly in crypto postings",
           subtitle="Hire rate by client country (Philippines or not) and crypto wording; 'Philippines, other' is a small group",
           x="", y="Hire rate")
    + base_theme()
)
save(p, "04c_philippines_vs_crypto", 8, 4)

# 13. Hypothesis (experience): AI-written vs hand-written text (proxy score)
df["ai_group"] = pd.cut(df["ai_style_signals"], [-1, 0, 1, 2, 5],
                        labels=["0 signals", "1 signal", "2 signals", "3+ signals"])
t = rate_table(df, "ai_group")
t["label"] = t.apply(lambda r: f"n={int(r['n']):,}", axis=1)
p = (
    ggplot(t, aes("ai_group", "rate"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), color=BLUE, size=0.8)
    + geom_text(aes(label="label", y="lo"), va="top", nudge_y=-0.01, size=8)
    + hline_base()
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="AI-style writing signals and hiring",
           subtitle="Count of stylistic tells: em dash, curly quotes, markdown, stock phrases, emoji (a proxy, not a detector)",
           x="AI-style signals in the description", y="Hire rate")
    + base_theme()
)
save(p, "11b_ai_style_signals", 7.5, 4.5)

# 14. "Dead" postings: no interview invited and no hire at all (0 hire, 0 interview)
dd = df[df["final_total_invited_to_interview"].notna()].copy()
dd["dead"] = ((dd["final_total_invited_to_interview"] == 0) & (dd["hired"] == 0)).astype(int)
DEAD_BASE = dd["dead"].mean()
checks = {
    "All postings": pd.Series(True, index=dd.index),
    "Client has no spend history": dd["client_total_spent"].fillna(0) == 0,
    "Crypto / web3 terms": dd["has_crypto_term"],
    "Mentions wallet / seed / key": dd["has_wallet_term"],
    "Description copied 2+ times": dd["desc_dup_count"] > 1,
    "Asks to leave the platform": dd["has_offplatform_contact"],
    "Client in the Philippines": dd["client_country"] == "Philippines",
    "Longest 20% descriptions": dd["description_word_count"] >= dd["description_word_count"].quantile(0.8),
    "Fixed-price": dd["job_type_label"] == "Fixed Price",
    "Hourly": dd["job_type_label"] == "Hourly",
}
from statsmodels.stats.proportion import proportion_confint
rows = []
for name, mask in checks.items():
    x, n_ = int(dd.loc[mask, "dead"].sum()), int(mask.sum())
    lo, hi = proportion_confint(x, n_, method="wilson")
    rows.append({"group": name, "n": n_, "rate": x / n_, "lo": lo, "hi": hi})
dead = pd.DataFrame(rows)
dead.to_csv(__import__("_common").DATA / "eda_dead_postings.csv", index=False)
dead["group"] = pd.Categorical(dead["group"], categories=dead.sort_values("rate")["group"], ordered=True)
p = (
    ggplot(dead, aes("group", "rate"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), color=BLUE)
    + geom_hline(yintercept=DEAD_BASE, linetype="dashed", color="#444444")
    + coord_flip()
    + scale_y_continuous(labels=pct_labels, limits=(0, 1))
    + labs(title="Dead postings: no interview and no hire",
           subtitle=f"Share of postings that never invite anyone to interview and never hire (overall {DEAD_BASE:.0%}, n={len(dd):,})",
           x="", y="Share of dead postings")
    + base_theme()
)
save(p, "14b_dead_postings", 8, 5)

# Summary table for the article text
summary = {
    "n_jobs": int(N),
    "hire_rate": round(float(BASE), 4),
    "window_start": str(df["first_captured_at"].min().date()),
    "window_end": str(df["first_captured_at"].max().date()),
    "median_observation_days": round(float(OBS), 1),
    "share_from_email_alerts": round(float((df["source_feed"] == "upworkAlertsEmail").mean()), 4),
    "median_applicants": float(ap["final_total_applicants"].median()),
    "crypto_share": round(float(df["has_crypto_term"].mean()), 4),
    "repost_share": round(float((df["desc_dup_count"] > 1).mean()), 4),
    "dead_posting_share": round(float(DEAD_BASE), 4),
    "dead_posting_n": int(len(dd)),
}
pd.Series(summary).to_json(__import__("_common").DATA / "eda_summary.json", indent=2)
print(summary)
