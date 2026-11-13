"""Step 10: ranking of facts by how much they matter (grouped permutation importance).

The model is trained once on all features. On the test set (latest 15% of postings) each group of
related features is shuffled (all its columns together, same shuffle) and we measure how much the
ROC AUC falls. A bigger fall means the model relied on that fact more. A fall indistinguishable from
zero means the fact carries no detectable signal once the other facts are known.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from plotnine import aes, coord_flip, geom_col, geom_errorbar, ggplot, labs, scale_fill_manual, scale_y_continuous
from sklearn.metrics import roc_auc_score

from _common import BLUE, DATA, ORANGE, base_theme, save
from _features import ALL, CATEGORICAL, FEATURE_GROUPS, SEED, model_frame

df, X, y, (tr, va, te) = model_frame()
cat_idx = [ALL.index(c) for c in ALL if c in CATEGORICAL]
cb = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6, eval_metric="AUC", random_seed=SEED,
                        verbose=0, early_stopping_rounds=100, allow_writing_files=False)
cb.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx), eval_set=Pool(X.iloc[va], y[va], cat_features=cat_idx))
Xt, yt = X.iloc[te].reset_index(drop=True), y[te]


def auc(frame):
    return roc_auc_score(yt, cb.predict_proba(Pool(frame, yt, cat_features=cat_idx))[:, 1])


base = auc(Xt)
print(f"test AUC with all features: {base:.3f}")
rng = np.random.default_rng(SEED)
rows = []
for name, cols in FEATURE_GROUPS.items():
    drops = []
    for _ in range(30):
        p = rng.permutation(len(Xt))
        shuffled = Xt.copy()
        shuffled[cols] = Xt[cols].iloc[p].to_numpy()
        drops.append(base - auc(shuffled))
    m, sd = float(np.mean(drops)), float(np.std(drops))
    rows.append({"group": name, "auc_drop": m, "sd": sd, "lo": m - 1.96 * sd, "hi": m + 1.96 * sd,
                 "matters": "Matters (drop clearly above zero)" if m - 1.96 * sd > 0.0005 else "No detectable effect"})
res = pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)
res["rank"] = res.index + 1
res.to_csv(DATA / "feature_group_importance.csv", index=False)
print(res.round(4).to_string(index=False))

plot = res.sort_values("auc_drop").copy()
plot["group"] = pd.Categorical(plot["group"], categories=plot["group"], ordered=True)
p = (
    ggplot(plot, aes("group", "auc_drop", fill="matters"))
    + geom_col(width=0.7)
    + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.3, color="#333333")
    + coord_flip()
    + scale_fill_manual({"Matters (drop clearly above zero)": BLUE, "No detectable effect": "#9A9A9A"})
    + labs(title="Which facts matter, from most to least",
           subtitle="How much the model's ROC AUC falls when that group of facts is scrambled (test set, 30 shuffles, 95% range)",
           x="", y="Fall in ROC AUC", fill="")
    + base_theme()
)
save(p, "27_feature_importance_groups", 10, 7)
