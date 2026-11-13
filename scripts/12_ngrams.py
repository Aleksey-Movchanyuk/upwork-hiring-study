"""Step 12: do word sequences (unigrams, bigrams, trigrams) help predict a hire?

1. Text-only models: TF-IDF of 1-grams, 1-2-grams, 1-3-grams + logistic regression, trained on the
   earliest 70%, strength chosen on the next 15%, scored on the latest 15%.
2. Does it add anything? The best n-gram model's score is fed to CatBoost as one extra feature
   (out-of-fold on the training part, so the model never sees its own answers). Compare with CatBoost alone.
3. Robustness: the same test restricted to postings whose exact description never appeared in training
   (otherwise re-posted text could flatter the n-gram model).
4. The n-grams most linked to hiring and to not hiring (from the training period only).
Text stays in memory; outputs are n-grams that appear in many postings, never job text.
"""
import hashlib

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from dotenv import load_dotenv
from plotnine import aes, coord_flip, geom_hline, geom_pointrange, ggplot, labs, scale_color_manual, scale_y_continuous
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from _common import BLUE, DATA, ORANGE, ROOT, base_theme, save
from _features import ALL, CATEGORICAL, SEED, bootstrap_auc, model_frame, top_share_stats

load_dotenv(ROOT / ".env")
from upwork_ml.db import get_connection  # noqa: E402

df, X, y, (tr, va, te) = model_frame()
with get_connection() as conn:
    t = pd.read_sql("select j.uid, j.title, j.description from data_upwork.jobs j "
                    "join data_upwork.job_outcomes o on o.uid = j.uid", conn)
t["job_key"] = t["uid"].map(lambda u: hashlib.sha1(u.encode()).hexdigest()[:12])
t = t.set_index("job_key").loc[df["job_key"]].reset_index()  # same row order as df
txt = (t["title"].fillna("") + " . " + t["description"].fillna("")).str.lower()
txt = txt.str.replace(r"https?://\S+|www\.\S+|\S+@\S+", " ", regex=True)
dhash = t["description"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip().map(
    lambda s: hashlib.md5(s.encode()).hexdigest()).to_numpy()
del t

yt, ytr, yva = y[te], y[tr], y[va]
results, fitted = [], {}
for label, ngr in (("Words (1-grams)", (1, 1)), ("Words and pairs (1-2-grams)", (1, 2)), ("Words, pairs, triples (1-3-grams)", (1, 3))):
    vec = TfidfVectorizer(ngram_range=ngr, min_df=5, max_df=0.5, sublinear_tf=True, token_pattern=r"(?u)\b[a-z][a-z\+\#\.]+\b")
    Atr = vec.fit_transform(txt.iloc[tr])
    Ava, Ate = vec.transform(txt.iloc[va]), vec.transform(txt.iloc[te])
    best = max(((roc_auc_score(yva, LogisticRegression(C=C, max_iter=2000).fit(Atr, ytr).predict_proba(Ava)[:, 1]), C)
                for C in (0.1, 0.3, 1, 3, 10)))
    m = LogisticRegression(C=best[1], max_iter=2000).fit(Atr, ytr)
    s = m.predict_proba(Ate)[:, 1]
    a, lo, hi = bootstrap_auc(yt, s)
    unseen = ~np.isin(dhash[te], dhash[tr])
    a_unseen = roc_auc_score(yt[unseen], s[unseen])
    results.append({"model": f"Text only: {label}", "roc_auc": a, "lo": lo, "hi": hi, "auc_unseen_text": a_unseen,
                    "n_features": Atr.shape[1], "C": best[1], "kind": "Text"})
    fitted[label] = (vec, m, ngr, best[1], Atr)
    print(f"{label:38s} AUC {a:.3f} [{lo:.3f},{hi:.3f}]  unseen-text AUC {a_unseen:.3f}  ({Atr.shape[1]:,} n-grams, C={best[1]})")

# Top n-grams (1-3) linked to hiring / not hiring, training period only
vec, m, _, _, Atr = fitted["Words, pairs, triples (1-3-grams)"]
terms = np.array(vec.get_feature_names_out())
df_cnt = np.asarray((Atr > 0).sum(0)).ravel()
hit = np.asarray((Atr > 0).T @ ytr).ravel()
coef = m.coef_[0]
tab = pd.DataFrame({"ngram": terms, "n_words": [len(x.split()) for x in terms], "postings": df_cnt, "hire_rate": hit / np.maximum(df_cnt, 1), "coef": coef})
tab = tab[tab["postings"] >= 30]
top = pd.concat([tab.sort_values("coef", ascending=False).head(25).assign(direction="hires"),
                 tab.sort_values("coef").head(25).assign(direction="does not hire")])
top.to_csv(DATA / "ngram_top_terms.csv", index=False)
print("\nN-grams linked to hiring:\n", tab.sort_values("coef", ascending=False).head(12)[["ngram", "postings", "hire_rate"]].round(3).to_string(index=False))
print("\nN-grams linked to NOT hiring:\n", tab.sort_values("coef").head(12)[["ngram", "postings", "hire_rate"]].round(3).to_string(index=False))

# Unigram table: strongest single words, with the hire rate in the training AND the test period
uvec, um, _, _, Utr = fitted["Words (1-grams)"]
uterms = np.array(uvec.get_feature_names_out())
Ute = uvec.transform(txt.iloc[te])
u_df_tr = np.asarray((Utr > 0).sum(0)).ravel()
u_hit_tr = np.asarray((Utr > 0).T @ ytr).ravel()
u_df_te = np.asarray((Ute > 0).sum(0)).ravel()
u_hit_te = np.asarray((Ute > 0).T @ yt).ravel()
ut = pd.DataFrame({"word": uterms, "coef": um.coef_[0], "postings_train": u_df_tr, "hire_rate_train": u_hit_tr / np.maximum(u_df_tr, 1),
                   "postings_test": u_df_te, "hire_rate_test": np.where(u_df_te > 0, u_hit_te / np.maximum(u_df_te, 1), np.nan)})
ut = ut[ut["postings_train"] >= 40]
uni = pd.concat([ut.sort_values("coef", ascending=False).head(20).assign(direction="hires"),
                 ut.sort_values("coef").head(20).assign(direction="does not hire")])
uni.to_csv(DATA / "unigram_top_terms.csv", index=False)
print("\nUNIGRAMS linked to HIRING (train postings >= 40):\n", ut.sort_values("coef", ascending=False).head(20)[["word", "postings_train", "hire_rate_train", "postings_test", "hire_rate_test"]].round(3).to_string(index=False))
print("\nUNIGRAMS linked to NOT hiring:\n", ut.sort_values("coef").head(20)[["word", "postings_train", "hire_rate_train", "postings_test", "hire_rate_test"]].round(3).to_string(index=False))

# Does the n-gram score help CatBoost? (out-of-fold on the training period)
vec13, _, ngr, C, _ = fitted["Words, pairs, triples (1-3-grams)"]
oof = np.zeros(len(ytr))
for k_tr, k_va in KFold(5, shuffle=False).split(np.arange(len(ytr))):
    v = TfidfVectorizer(ngram_range=ngr, min_df=5, max_df=0.5, sublinear_tf=True, token_pattern=r"(?u)\b[a-z][a-z\+\#\.]+\b")
    A = v.fit_transform(txt.iloc[tr].iloc[k_tr])
    oof[k_va] = LogisticRegression(C=C, max_iter=2000).fit(A, ytr[k_tr]).predict_proba(v.transform(txt.iloc[tr].iloc[k_va]))[:, 1]
full_vec, full_m = fitted["Words, pairs, triples (1-3-grams)"][0], fitted["Words, pairs, triples (1-3-grams)"][1]
ng = np.zeros(len(y))
ng[tr] = oof
ng[va] = full_m.predict_proba(full_vec.transform(txt.iloc[va]))[:, 1]
ng[te] = full_m.predict_proba(full_vec.transform(txt.iloc[te]))[:, 1]


def run_catboost(frame):
    cols = list(frame.columns)
    ci = [cols.index(c) for c in cols if c in CATEGORICAL]
    cb = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6, eval_metric="AUC", random_seed=SEED,
                            verbose=0, early_stopping_rounds=100, allow_writing_files=False)
    cb.fit(Pool(frame.iloc[tr], ytr, cat_features=ci), eval_set=Pool(frame.iloc[va], yva, cat_features=ci))
    return cb.predict_proba(Pool(frame.iloc[te], yt, cat_features=ci))[:, 1]


s_base = run_catboost(X)
s_stack = run_catboost(X.assign(ngram_score=ng))
pd.DataFrame({"y": yt, "CatBoost facts only": s_base, "CatBoost + word model": s_stack}).to_parquet(DATA / "ml_test_scores_stack.parquet", index=False)
for name, s, kind in (("Structured facts only (CatBoost)", s_base, "Facts"), ("Structured facts + n-gram score (CatBoost)", s_stack, "Facts + text")):
    a, lo, hi = bootstrap_auc(yt, s)
    unseen = ~np.isin(dhash[te], dhash[tr])
    results.append({"model": name, "roc_auc": a, "lo": lo, "hi": hi, "auc_unseen_text": roc_auc_score(yt[unseen], s[unseen]),
                    "n_features": X.shape[1] + (name != "Structured facts only (CatBoost)"), "C": np.nan, "kind": kind,
                    **top_share_stats(yt, s)})
    print(f"{name:45s} AUC {a:.3f} [{lo:.3f},{hi:.3f}]")
rng = np.random.default_rng(SEED)
d = []
for _ in range(1000):
    i = rng.integers(0, len(yt), len(yt))
    d.append(roc_auc_score(yt[i], s_stack[i]) - roc_auc_score(yt[i], s_base[i]))
print(f"AUC gain from adding n-grams: {np.mean(d):+.4f}  95% range [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
res = pd.DataFrame(results)
res["gain_ci"] = ""
res.loc[res["model"].str.startswith("Structured facts + n-gram"), "gain_ci"] = f"{np.mean(d):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
res.to_csv(DATA / "ml_ngram_results.csv", index=False)

plot = res.sort_values("roc_auc").copy()
plot["model"] = pd.Categorical(plot["model"], categories=plot["model"], ordered=True)
p = (
    ggplot(plot, aes("model", "roc_auc", color="kind"))
    + geom_pointrange(aes(ymin="lo", ymax="hi"))
    + geom_hline(yintercept=0.5, linetype="dashed", color="#444444")
    + coord_flip()
    + scale_color_manual({"Text": ORANGE, "Facts": BLUE, "Facts + text": "#2A9D5C"})
    + scale_y_continuous(limits=(0.45, 0.8))
    + labs(title="Do word sequences help?",
           subtitle="Test ROC AUC (latest 15% of postings), 95% bootstrap range", x="", y="ROC AUC", color="")
    + base_theme()
)
save(p, "29_ngram_models", 9.5, 4.5)
