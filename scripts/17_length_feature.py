"""Step 17: how much does the length of the job description help predict a hire?

1. The feature alone: a model that sees ONLY description length (hump-shaped lookup learned on the oldest 70%).
2. The model with and without the length features (CatBoost, same settings, 3 seeds averaged), paired bootstrap.
3. The checklist with and without its "400 words or fewer" question.
4. Is the pattern robust? Hire rate by length band within fixed-price/hourly, client with/without spend history, and training vs test period.
Outputs: data/length_*.csv
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from _common import DATA
from _features import ALL, CATEGORICAL, SEED, bootstrap_auc, model_frame

df, X, y, (tr, va, te) = model_frame()
yt, ytr = y[te], y[tr]
words = df["description_word_count"].to_numpy()
rng = np.random.default_rng(SEED)
draws = [rng.integers(0, len(yt), len(yt)) for _ in range(2000)]


def paired(sa, sb):
    d = [roc_auc_score(yt[i], sa[i]) - roc_auc_score(yt[i], sb[i]) for i in draws if yt[i].min() != yt[i].max()]
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- 1. the feature alone
edges = np.unique(np.quantile(words[tr], np.linspace(0, 1, 11)))
edges[0], edges[-1] = -1, 1e9
dec_tr = np.digitize(words[tr], edges[1:-1])
rate_by_dec = pd.Series(ytr).groupby(dec_tr).mean()
lookup = lambda w: rate_by_dec.reindex(np.digitize(w, edges[1:-1])).to_numpy()
s_hump = lookup(words[te])
s_mono = -words[te].astype(float)
rows = []
for name, s in [("Length alone, hump-shaped (decile lookup learned on the oldest 70%)", s_hump), ("Length alone, 'shorter is better' (minus the word count)", s_mono)]:
    a, lo, hi = bootstrap_auc(yt, s)
    rows.append({"model": name, "test_auc": a, "lo": lo, "hi": hi})
spent = (df["client_total_spent"].fillna(0) > 0).to_numpy().astype(float)
a, lo, hi = bootstrap_auc(yt, spent[te])
rows.append({"model": "For comparison: 'client has spent money' alone", "test_auc": a, "lo": lo, "hi": hi})
alone = pd.DataFrame(rows)
alone.to_csv(DATA / "length_alone.csv", index=False)
print(alone.round(3).to_string(index=False))
d_h = paired(s_hump, spent[te])
print("hump-shaped length minus 'has spent money': %+.3f (%.3f to %.3f)" % d_h)

# ---------------------------------------------------------------- 2. model with and without the length features
LEN = ["description_char_count", "description_word_count"]


def fit_predict(cols, seed):
    cat_idx = [cols.index(c) for c in cols if c in CATEGORICAL]
    cb = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6, eval_metric="AUC", random_seed=seed, verbose=0,
                            early_stopping_rounds=100, allow_writing_files=False)
    cb.fit(Pool(X[cols].iloc[tr], y[tr], cat_features=cat_idx), eval_set=Pool(X[cols].iloc[va], y[va], cat_features=cat_idx))
    return cb.predict_proba(Pool(X[cols].iloc[te], yt, cat_features=cat_idx))[:, 1]


with_len = np.mean([fit_predict(list(ALL), s) for s in (42, 43, 44)], axis=0)
no_len = np.mean([fit_predict([c for c in ALL if c not in LEN], s) for s in (42, 43, 44)], axis=0)
abl = []
for name, s in [("All 52 facts", with_len), ("Without the two description-length facts", no_len)]:
    a, lo, hi = bootstrap_auc(yt, s)
    abl.append({"model": name, "test_auc": a, "lo": lo, "hi": hi})
d_m = paired(with_len, no_len)
abl.append({"model": "Difference (with minus without), paired bootstrap", "test_auc": d_m[0], "lo": d_m[1], "hi": d_m[2]})
abl = pd.DataFrame(abl)
abl.to_csv(DATA / "length_ablation_model.csv", index=False)
print(abl.round(3).to_string(index=False))

# ---------------------------------------------------------------- 3. the checklist with and without the length question
fixed = (df["job_type_label"] == "Fixed Price")
hires = df["client_total_hires"].fillna(0)
dur = df["duration_label"].map({"Less than 1 week": 0, "Less than 1 month": 1, "1 to 3 months": 2, "3 to 6 months": 3, "More than 6 months": 4}).fillna(99)
q = {"fixed": fixed, "spent": df["client_total_spent"].fillna(0) > 0, "hires": hires >= 10, "dur": dur <= 2, "crypto": ~df["has_crypto_term"], "agency": ~df["agency_tone"],
     "length": df["description_word_count"] <= 400}
pts7 = sum(v.fillna(False).astype(int) for v in q.values()).to_numpy().astype(float)
pts6 = sum(v.fillna(False).astype(int) for k, v in q.items() if k != "length").to_numpy().astype(float)
a7, l7, h7 = bootstrap_auc(yt, pts7[te]); a6, l6, h6 = bootstrap_auc(yt, pts6[te]); d_c = paired(pts7[te], pts6[te])
chk = pd.DataFrame([{"checklist": "Seven questions", "test_auc": a7, "lo": l7, "hi": h7}, {"checklist": "Six questions (without the 400-word question)", "test_auc": a6, "lo": l6, "hi": h6},
                    {"checklist": "Difference (seven minus six), paired bootstrap", "test_auc": d_c[0], "lo": d_c[1], "hi": d_c[2]}])
chk.to_csv(DATA / "length_checklist_ablation.csv", index=False)
print(chk.round(3).to_string(index=False))

# ---------------------------------------------------------------- 4. robustness of the pattern
band = pd.cut(df["description_word_count"], [-1, 100, 200, 400, 800, 1e9], labels=["up to 100", "101-200", "201-400", "401-800", "over 800"])
df["band"] = band
pos = np.arange(len(df))
df["period"] = np.where(pos < tr.stop, "Oldest 70%", np.where(pos < va.stop, "Next 15%", "Latest 15%"))
df["jobtype"] = np.where(fixed, "Fixed price", "Hourly")
df["spend"] = np.where(df["client_total_spent"].fillna(0) > 0, "Client has spent money", "No spend history")
out = []
for dim in ("period", "jobtype", "spend"):
    t = df.groupby([dim, "band"], observed=True)["hired"].agg(["size", "mean"]).reset_index().rename(columns={dim: "group", "size": "postings", "mean": "hire_rate"})
    t.insert(0, "split_by", dim)
    out.append(t)
rob = pd.concat(out)
rob["hire_rate"] = rob["hire_rate"].astype(float)
rob.to_csv(DATA / "length_robustness.csv", index=False)
print(rob.pivot_table(index=["split_by", "group"], columns="band", values="hire_rate", observed=True).round(3).to_string())
print(rob.pivot_table(index=["split_by", "group"], columns="band", values="postings", observed=True).to_string())
