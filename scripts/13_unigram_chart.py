"""Step 13: chart of the strongest single words (unigrams), training versus test hire rate.

Reads data/unigram_top_terms.csv written by 12_ngrams.py. Shows the 15 words most linked to hiring and the
15 most linked to not hiring (words in at least 100 training postings). Blue = training period,
orange = the later test period. The dashed line is the overall hire rate of that period.
"""
import numpy as np
import pandas as pd
from plotnine import (aes, coord_flip, element_text, facet_wrap, geom_hline, geom_pointrange, ggplot, labs,
                      position_dodge, scale_color_manual, scale_y_continuous, theme)

from _common import BLUE, DATA, ORANGE, base_theme, load_features, pct_labels, save

u = pd.read_csv(DATA / "unigram_top_terms.csv")
u = u[u["postings_train"] >= 100]
u = pd.concat([u[u["direction"] == "hires"].head(15), u[u["direction"] == "does not hire"].head(15)])
u["direction"] = u["direction"].map({"hires": "Linked to hiring", "does not hire": "Linked to not hiring"})

df = load_features()
n = len(df)
base_tr = df["hired"].iloc[: int(n * 0.70)].astype(int).mean()
base_te = df["hired"].iloc[int(n * 0.85):].astype(int).mean()

rows = []
for _, r in u.iterrows():
    for period, k, rate in (("Training period", r["postings_train"], r["hire_rate_train"]),
                            ("Test period (later)", r["postings_test"], r["hire_rate_test"])):
        if k and not np.isnan(rate):
            se = np.sqrt(rate * (1 - rate) / k)
            rows.append({"word": r["word"], "direction": r["direction"], "period": period, "rate": rate,
                         "lo": max(rate - 1.96 * se, 0), "hi": rate + 1.96 * se, "n": int(k)})
p_df = pd.DataFrame(rows)

# order words by training hire rate within each panel
order = (p_df[p_df["period"] == "Training period"].sort_values("rate")["word"]).tolist()
p_df["word"] = pd.Categorical(p_df["word"], categories=order, ordered=True)
p_df["direction"] = pd.Categorical(p_df["direction"], categories=["Linked to hiring", "Linked to not hiring"], ordered=True)

p = (
    ggplot(p_df, aes("word", "rate", color="period"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"), position=position_dodge(width=0.6), size=0.5)
    + geom_hline(yintercept=base_tr, linetype="dashed", color=BLUE, alpha=0.7)
    + geom_hline(yintercept=base_te, linetype="dotted", color=ORANGE, alpha=0.9)
    + coord_flip()
    + facet_wrap("direction", ncol=1, scales="free_y")
    + scale_color_manual({"Training period": BLUE, "Test period (later)": ORANGE})
    + scale_y_continuous(labels=pct_labels, limits=(0, None))
    + labs(title="The single words most linked to hiring and not hiring",
           subtitle=f"Hire rate of postings with the word, 95% range. Lines: overall rate (training {base_tr:.1%} dashed, test {base_te:.1%} dotted)",
           x="", y="Hire rate", color="")
    + base_theme()
    + theme(subplots_adjust={"hspace": 0.25})
)
save(p, "30_unigram_words", 9, 9.5)
