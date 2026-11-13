"""Shared paths, theme and helpers for the article scripts.

Plots use plotnine (a Python port of ggplot2) with the default grey theme,
which reproduces the look of the JHU Data Science Specialization slides
(grey panel, white gridlines, lm line with a grey confidence band).
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
from plotnine import element_text, theme, theme_gray
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "article"
DATA = ART / "data"
FIG = ART / "figures"
FEATURES = DATA / "features.parquet"

sys.path.insert(0, str(ROOT / "src"))

# Upwork sells Connects at about $0.15 each (assumption, adjust if it changes).
CONNECT_USD = 0.15

BLUE = "#3366CC"
ORANGE = "#E8833A"
HIRED_COLORS = {"not hired": "#7A7A7A", "hired": ORANGE}


def base_theme():
    return theme_gray(base_size=11) + theme(
        plot_title=element_text(weight="bold", size=13),
        plot_subtitle=element_text(size=9),
        figure_size=(8, 5),
    )


def pct_labels(values):
    return [f"{v:.0%}" for v in values]


def save(plot, name, width=8, height=5):
    FIG.mkdir(parents=True, exist_ok=True)
    plot.save(FIG / f"{name}.png", width=width, height=height, dpi=200, verbose=False)
    print(f"saved figures/{name}.png")


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES)
    df["published_on"] = pd.to_datetime(df["published_on"], utc=True)
    df["first_captured_at"] = pd.to_datetime(df["first_captured_at"], utc=True)
    df["hired"] = df["hired"].astype(int)
    df["outcome"] = df["hired"].map({0: "not hired", 1: "hired"})
    return df.sort_values("first_captured_at").reset_index(drop=True)


def rate_table(df: pd.DataFrame, by) -> pd.DataFrame:
    """Hire rate per group with a 95% Wilson interval."""
    by = [by] if isinstance(by, str) else list(by)
    g = df.groupby(by, observed=True)["hired"].agg(hired_n="sum", n="count").reset_index()
    lo, hi = proportion_confint(g["hired_n"], g["n"], alpha=0.05, method="wilson")
    g["rate"] = g["hired_n"] / g["n"]
    g["lo"], g["hi"] = lo, hi
    return g
