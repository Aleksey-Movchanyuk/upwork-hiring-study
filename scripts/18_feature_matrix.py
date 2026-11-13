"""Step 18: a matrix of small charts, one per feature group (ordered by importance), to see what is signal and what is noise.

Each cell: X = values of one representative feature from the group, bar height = number of postings, orange part = hired.
The short black tick on each bar is the number of hires you would see if the feature meant nothing (postings x overall hire rate).
Orange clearly above or below the tick = the feature separates hired from not hired. Orange on the tick = noise.
Panel titles are coloured by the permutation importance test (10_importance.py): green = clearly above zero.
Output: figures/35_feature_matrix.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import DATA, FIG, ORANGE

df = pd.read_parquet(DATA / "features.parquet")
df = df[df["hired"].notna()].copy()
df["hired"] = df["hired"].astype(int)
base = df["hired"].mean()
imp = pd.read_csv(DATA / "feature_group_importance.csv").set_index("group")


def cut(s, edges, labels):
    return pd.cut(s, edges, labels=labels, right=True, include_lowest=True).astype(object)


def flag(*pairs):
    """Several yes/no columns shown side by side: bar = postings where the flag is on."""
    return pairs


df["tags_listed"] = df["tags_count"] > 0
df["skills_listed"] = df["skills_count"] > 0
top_countries = df["client_country"].value_counts().head(6).index.tolist()
dur_order = ["Less than 1 month", "1 to 3 months", "3 to 6 months", "More than 6 months"]
dur = df["duration_label"].where(df["duration_label"].isin(dur_order))
spent = df["client_total_spent"].fillna(0)
hires = df["client_total_hires"]
panels = {  # group name -> (short title, x series (categories), order)
    "Duration, engagement, tier": ("Project length", dur.map({"Less than 1 month": "<1 month", "1 to 3 months": "1-3 months", "3 to 6 months": "3-6 months", "More than 6 months": "6+ months"}), ["<1 month", "1-3 months", "3-6 months", "6+ months"]),
    "Budget": ("Fixed budget ($)", cut(df["budget_fixed"].where(df["budget_fixed"] > 0), [0, 20, 80, 220, 650, 1e9], ["<=20", "20-80", "80-220", "220-650", "650+"]), ["<=20", "20-80", "80-220", "220-650", "650+"]),
    "Job type (fixed price or hourly)": ("Job type", np.where(df["job_type_label"] == "Fixed Price", "Fixed", "Hourly"), ["Fixed", "Hourly"]),
    "Client spend history": ("Client total spent ($)", cut(spent, [-1, 0, 1000, 10000, 100000, 1e12], ["none", "<1k", "1-10k", "10-100k", "100k+"]), ["none", "<1k", "1-10k", "10-100k", "100k+"]),
    "Description length": ("Words in description", cut(df["description_word_count"], [-1, 100, 200, 400, 800, 1e9], ["<=100", "101-200", "201-400", "401-800", "800+"]), ["<=100", "101-200", "201-400", "401-800", "800+"]),
    "Client past hires": ("Client past hires", cut(hires, [-1, 0, 9, 49, 1e9], ["0", "1-9", "10-49", "50+"]), ["0", "1-9", "10-49", "50+"]),
    "Writing style (sentences, lines, bullets, !)": ("Bullet points in text", cut(df["bullet_count"], [-1, 0, 4, 1e9], ["0", "1-4", "5+"]), ["0", "1-4", "5+"]),
    "Title length": ("Characters in title", cut(df["title_char_count"], [-1, 37, 55, 72, 1e9], ["<=37", "38-55", "56-72", "73+"]), ["<=37", "38-55", "56-72", "73+"]),
    "Client country": ("Client country", df["client_country"].where(df["client_country"].isin(top_countries), "Other").replace({"United States": "US", "United Kingdom": "UK", "United Arab Emirates": "UAE"}), None),
    "Text copied by others": ("Earlier copies of the text", cut(df["desc_prior_copies"], [-1, 0, 1, 1e9], ["0", "1", "2+"]), ["0", "1", "2+"]),
    "Other request wording (fix, long-term, quick, screening)": ("Wording flag on", flag(("fix", "fix_title"), ("long-term", "long_term_wording"), ("quick", "quick_task_wording"), ("screening", "screening_prompt")), None),
    "Crypto and wallet words": ("Crypto / wallet", flag(("crypto", "has_crypto_term"), ("wallet", "has_wallet_term")), None),
    "Agency wording": ("Agency wording", np.where(df["agency_tone"].fillna(False), "yes", "no"), ["no", "yes"]),
    "Asks to build something new": ("Asks for something new", np.where(df["new_build"].fillna(False), "yes", "no"), ["no", "yes"]),
    "Client badge and payment check": ("Payment check level", df["client_payment_verification_status"].map({1.0: "level 1", 2.0: "level 2"}).fillna("unknown"), ["level 1", "level 2", "unknown"]),
    "Other job settings (tags, skills, premium...)": ("Setting filled in (rare)", flag(("skills listed", "skills_listed"), ("tags listed", "tags_listed"), ("premium", "premium")), None),
    "Links and emails in the text": ("Link / email in text", flag(("link", "desc_has_url"), ("email", "desc_has_email")), None),
    "Posting time (hour, weekday)": ("Weekday posted", df["published_dow"].map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
    "AI-style marks (dashes, quotes, markdown, dictionary score)": ("AI-style signals", cut(df["ai_style_signals"], [-1, 0, 1, 2, 99], ["0", "1", "2", "3+"]), ["0", "1", "2", "3+"]),
    "Asks to leave the platform": ("Asks to leave Upwork", np.where(df["has_offplatform_contact"].fillna(False), "yes", "no"), ["no", "yes"]),
}


def table_for(x, order):
    if isinstance(x, tuple):  # flag panel
        rows = []
        for label, col in x:
            m = df[col].fillna(False).astype(bool)
            rows.append((label, int(m.sum()), int(df.loc[m, "hired"].sum())))
        return pd.DataFrame(rows, columns=["x", "n", "hired"])
    s = pd.Series(np.asarray(x, dtype=object), index=df.index)
    t = df.assign(_x=s).dropna(subset=["_x"]).groupby("_x")["hired"].agg(n="size", hired="sum").reset_index().rename(columns={"_x": "x"})
    if order is not None:
        t = t.set_index("x").reindex(order).dropna().reset_index()
    else:
        t = t.sort_values("n", ascending=False)
    return t


groups = list(imp.sort_values("rank").index)
ncol, nrow = 5, 4
fig, axes = plt.subplots(nrow, ncol, figsize=(19, 14.5))
for ax, g in zip(axes.flat, groups):
    title, x, order = panels[g]
    t = table_for(x, order)
    t = t[t["n"] > 0].reset_index(drop=True)
    pos = np.arange(len(t))
    ax.bar(pos, t["n"], color="#D9D9D9", width=0.72)
    ax.bar(pos, t["hired"], color=ORANGE, width=0.72)
    expected = t["n"] * base
    for p_, e_ in zip(pos, expected):
        ax.plot([p_ - 0.36, p_ + 0.36], [e_, e_], color="black", lw=1.6)
    for p_, n_, h_ in zip(pos, t["n"], t["hired"]):
        ax.text(p_, n_, f"{100 * h_ / n_:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(t["x"], fontsize=7.5, rotation=0 if len(t) <= 5 else 45, ha="center" if len(t) <= 5 else "right")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, t["n"].max() * 1.15)
    r = imp.loc[g]
    matters = r["matters"].startswith("Matters")
    ax.set_title(f"#{int(r['rank'])} {title}", fontsize=10, fontweight="bold", loc="left", pad=16, color="#1B7F3B" if matters else "#555555")
    ax.text(0, 1.02, f"AUC drop {r['auc_drop']:.3f}, {'matters' if matters else 'noise / not clear'}", transform=ax.transAxes, fontsize=8, ha="left", va="bottom", color="#1B7F3B" if matters else "#555555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("Which features are signal and which are noise: postings (grey) and hires (orange) for each feature group", fontsize=15, fontweight="bold", x=0.01, ha="left")
fig.text(0.01, 0.947, f"Black tick = hires expected if the feature meant nothing (postings x overall hire rate of {100 * base:.1f}%). Percentage = actual hire rate. Panels ordered by importance; green = clearly matters.", fontsize=10, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "35_feature_matrix.png", dpi=130)
print("saved", FIG / "35_feature_matrix.png", "| base", round(base, 3), "| rows", len(df))
