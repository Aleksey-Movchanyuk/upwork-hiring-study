"""Redraw results/benchmark_top10.png from results/benchmark_all_models.csv (no retraining needed)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
COL = {"Boosting": "#3366CC", "Combination": "#E8833A", "Neural net": "#7B4FA3", "Linear": "#2E8B57", "Trees": "#999999",
       "Checklist+": "#B22222", "Distance": "#555555", "Kernel": "#555555"}


def nice(name):
    """Plain-English names for the chart (the CSV keeps the technical names)."""
    rules = {"Blend: best boosting + best neural net + best linear": "Blend: best boosting + neural net + linear model",
             "Blend: best 10 on the tuning slice": "Blend of the 10 best models", "Blend: best 5 on the tuning slice": "Blend of the 5 best models",
             "Blend: best 3 on the tuning slice": "Blend of the 3 best models", "Blend: CatBoost depth 6 + logistic regression C=0.1": "Blend: CatBoost + logistic regression",
             "MLP (256, 128), alpha=0.01": "Neural net, 2 layers (256, 128 units)", "MLP (64, 32), alpha=0.001": "Neural net, 2 layers (64, 32), light regularisation",
             "MLP (64, 32), alpha=0.1": "Neural net, 2 layers (64, 32), strong regularisation", "MLP (64, 32), alpha=0.01": "Neural net, 2 layers (64, 32)",
             "MLP (256, 128, 64, 32, 16), alpha=0.01": "Neural net, 5 layers (256 down to 16 units)"}
    return rules.get(name, name)


def make_figure():
    tab = pd.read_csv(OUT / "benchmark_all_models.csv")
    chk = pd.read_csv(OUT / "benchmark_checklist.csv").iloc[0]
    top = tab.head(10).iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [1.3, 1]})
    a1.barh([nice(m) for m in top["model"]], top["test_auc"], xerr=[top["test_auc"] - top["test_lo"], top["test_hi"] - top["test_auc"]],
            color=[COL[f] for f in top["family"]], ecolor="black", capsize=2)
    a1.axvline(chk["test_auc"], color="black", ls="--")
    a1.text(chk["test_auc"] - 0.001, 10.55, f"Seven-question checklist ({chk['test_auc']:.3f})", fontsize=9, va="bottom", ha="right")
    a1.set_xlim(0.66, 0.8); a1.set_ylim(-0.6, 11.2)
    a1.set_xlabel("ROC AUC on the test period (bars: 95% range)")
    a1.set_title("Top 10 of 64 models, picked before looking at the test period", fontsize=12, loc="left")
    a1.tick_params(axis="y", labelsize=9)
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=COL["Combination"], label="Blend of models"), Patch(color=COL["Neural net"], label="Neural net")], fontsize=9, loc="lower right")
    a2.scatter(tab["val_auc"], tab["test_auc"], c=[COL[f] for f in tab["family"]], s=24)
    a2.axhline(chk["test_auc"], color="black", ls="--", lw=1); a2.axvline(chk["val_auc"], color="black", ls="--", lw=1)
    a2.set_xlim(0.62, 0.76); a2.set_ylim(0.63, 0.76)
    for lab, c in [("Boosting", COL["Boosting"]), ("Blend of models", COL["Combination"]), ("Neural net", COL["Neural net"]), ("Linear", COL["Linear"]),
                   ("Trees", COL["Trees"]), ("Checklist-based", COL["Checklist+"]), ("Nearest neighbours, SVM", COL["Distance"])]:
        a2.scatter([], [], c=c, s=24, label=lab)
    a2.plot([], [], "k--", lw=1, label="Seven-question checklist")
    a2.legend(fontsize=8, loc="lower right")
    a2.set_xlabel("ROC AUC on the tuning period"); a2.set_ylabel("ROC AUC on the test period")
    a2.set_title("All models: tuning period against test period", fontsize=12, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "benchmark_top10.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    make_figure()
