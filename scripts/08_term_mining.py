"""Step 8: data-driven mining of words and two-word phrases.

For every word / bigram found in at least MIN_DF postings (title + description):
  1. hire rate with vs without it, chi-square p-value, Benjamini-Hochberg correction;
  2. for the strongest candidates, an adjusted odds ratio (job type, no spend, length,
     client spend, client past hires), again BH-corrected;
  3. a robustness check: the hire rate when each distinct description is counted once
     (so one client re-posting the same text cannot drive a term).

Output (terms and counts only, no job text): data/term_mining.csv, figure 25.
"""
import hashlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from plotnine import (aes, coord_flip, facet_wrap, geom_hline, geom_pointrange, ggplot, labs,
                      scale_color_manual, scale_y_continuous, theme)
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from statsmodels.stats.multitest import multipletests

from _common import BLUE, DATA, ORANGE, ROOT, base_theme, load_features, pct_labels, save

load_dotenv(ROOT / ".env")
from upwork_ml.db import get_connection  # noqa: E402

MIN_DF = 100
N_CANDIDATES = 120

with get_connection() as conn:
    t = pd.read_sql("select j.uid, j.title, j.description from data_upwork.jobs j "
                    "join data_upwork.job_outcomes o on o.uid = j.uid", conn)
t["job_key"] = t["uid"].map(lambda u: hashlib.sha1(u.encode()).hexdigest()[:12])
df = load_features().merge(t[["job_key", "title", "description"]], on="job_key", how="inner")
df["hired"] = df["hired"].astype(int)
txt = (df["title"].fillna("") + " . " + df["description"].fillna("")).str.lower()
txt = txt.str.replace(r"https?://\S+|www\.\S+|\S+@\S+", " ", regex=True)
dhash = df["description"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip().map(
    lambda s: hashlib.md5(s.encode()).hexdigest()).to_numpy()
y = df["hired"].to_numpy()
n, pos = len(y), int(y.sum())

cv = CountVectorizer(ngram_range=(1, 2), min_df=MIN_DF, binary=True, stop_words="english",
                     token_pattern=r"(?u)\b[a-z][a-z\+\#\.]{2,}\b")
M = cv.fit_transform(txt).tocsc()
terms = np.array(cv.get_feature_names_out())
df_ = np.asarray(M.sum(0)).ravel()
hit = np.asarray(M.T @ y).ravel()
tab = np.vstack([hit, df_ - hit, pos - hit, (n - df_) - (pos - hit)]).T
pvals = np.array([stats.chi2_contingency(r.reshape(2, 2), correction=False)[1] for r in tab])
res = pd.DataFrame({"term": terms, "n": df_, "rate": hit / df_, "rate_without": (pos - hit) / (n - df_),
                    "p": pvals})
res["diff_pp"] = 100 * (res["rate"] - res["rate_without"])
res["q"] = multipletests(res["p"], method="fdr_bh")[1]
print(f"terms tested: {len(res):,}; significant at BH 5%: {(res['q'] < 0.05).sum():,}")

ctrl = pd.DataFrame({
    "fixed": (df["job_type_label"] == "Fixed Price").astype(int),
    "no_spend": (df["client_total_spent"].fillna(0) == 0).astype(int)})
for name, s in {"lw": np.log10(df["description_word_count"] + 1),
                "ls": np.log10(df["client_total_spent"].fillna(0) + 1),
                "lh": np.log1p(df["client_total_hires"].fillna(0))}.items():
    ctrl[name] = (s - s.mean()) / s.std()
base = sm.add_constant(ctrl)

cand = res[res["q"] < 0.05].assign(a=lambda x: x["diff_pp"].abs()).sort_values("a", ascending=False).head(N_CANDIDATES)
rows = []
idx = {term: i for i, term in enumerate(terms)}
for _, r in cand.iterrows():
    x = np.asarray(M[:, idx[r["term"]]].todense()).ravel()
    try:
        m = sm.Logit(y, base.assign(term=x)).fit(disp=0)
    except Exception:  # perfect separation etc.
        continue
    sel = x == 1
    first = pd.Series(dhash[sel]).drop_duplicates(keep="first").index
    rows.append({"term": r["term"], "adj_or": float(np.exp(m.params["term"])), "adj_p": float(m.pvalues["term"]),
                 "unique_texts": int(len(first)), "rate_one_per_text": float(y[sel][first].mean()),
                 "top_text_share": float(pd.Series(dhash[sel]).value_counts().iloc[0] / sel.sum())})
adj = pd.DataFrame(rows)
adj["adj_q"] = multipletests(adj["adj_p"], method="fdr_bh")[1]
out = res.merge(adj, on="term", how="left").sort_values("diff_pp")
out.to_csv(DATA / "term_mining.csv", index=False)

keep = out[(out["adj_q"] < 0.05) & (out["n"] >= MIN_DF) & (out["top_text_share"] <= 0.2)].copy()
hi = keep[keep["adj_or"] > 1].sort_values("adj_or", ascending=False).head(10)
lo = keep[keep["adj_or"] < 1].sort_values("adj_or").head(10)
print("\nHIGHER:\n", hi[["term", "n", "rate", "adj_or", "rate_one_per_text"]].round(3).to_string(index=False))
print("\nLOWER:\n", lo[["term", "n", "rate", "adj_or", "rate_one_per_text"]].round(3).to_string(index=False))

plot = pd.concat([hi.assign(group="Hire more often"), lo.assign(group="Hire less often")])
plot["se"] = np.sqrt(plot["rate"] * (1 - plot["rate"]) / plot["n"])
plot["lo"] = (plot["rate"] - 1.96 * plot["se"]).clip(lower=0)
plot["hi"] = plot["rate"] + 1.96 * plot["se"]
plot["term"] = plot["term"] + "  (n=" + plot["n"].astype(int).astype(str) + ")"
plot = plot.sort_values("rate")
plot["term"] = pd.Categorical(plot["term"], categories=plot["term"], ordered=True)
p = (
    ggplot(plot, aes("term", "rate", color="group"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"))
    + geom_hline(yintercept=df["hired"].mean(), linetype="dashed", color="#444444")
    + coord_flip()
    + scale_color_manual({"Hire more often": BLUE, "Hire less often": ORANGE})
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="Words and phrases linked to unusually high or low hire rates",
           subtitle=f"Found by scanning {len(res):,} terms; kept if they survive adjustment for job type, "
                    "length and client history (BH 5%)",
           x="", y="Hire rate", color="")
    + base_theme()
)
save(p, "25_term_mining", 9, 7)
