"""
Figures of the preliminary benchmark (run of 2026-10-22, CNNs trained 1 epoch).
The values are the output of train.py on ourdataset (see results/report.md).
The full run uses generate_report.py, which reads everything from MLflow.

    python plot_preliminary.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).parent / "results"

# model: (color, {metric: (mean, std)}, MAcc of folds 1-3) - best first
RESULTS = {
    "Random Forest": ("#2a78d6", {"macc": (0.797, 0.013), "sensitivity": (0.713, 0.035), "specificity": (0.881, 0.013), "accuracy": (0.843, 0.006), "f1": (0.673, 0.013), "roc_auc": (0.913, 0.005)}, [0.800, 0.781, 0.811]),
    "XGBoost": ("#1baf7a", {"macc": (0.793, 0.019), "sensitivity": (0.739, 0.044), "specificity": (0.847, 0.006), "accuracy": (0.822, 0.006), "f1": (0.653, 0.021), "roc_auc": (0.902, 0.008)}, [0.811, 0.766, 0.801]),
    "2D-CNN": ("#e87ba4", {"macc": (0.757, 0.021), "sensitivity": (0.770, 0.080), "specificity": (0.743, 0.040), "accuracy": (0.749, 0.014), "f1": (0.581, 0.016), "roc_auc": (0.833, 0.031)}, [0.777, 0.765, 0.728]),
    "SVM": ("#eb6834", {"macc": (0.753, 0.013), "sensitivity": (0.580, 0.030), "specificity": (0.926, 0.006), "accuracy": (0.848, 0.004), "f1": (0.633, 0.017), "roc_auc": (0.899, 0.005)}, [0.771, 0.743, 0.745]),
    "1D-CNN": ("#eda100", {"macc": (0.718, 0.012), "sensitivity": (0.788, 0.197), "specificity": (0.649, 0.184), "accuracy": (0.681, 0.098), "f1": (0.533, 0.021), "roc_auc": (0.850, 0.024)}, [0.736, 0.710, 0.710])
}
FAMILY = {"Random Forest": "ML", "XGBoost": "ML", "SVM": "ML", "2D-CNN": "DL", "1D-CNN": "DL"}
METRICS = {"macc": "MAcc", "sensitivity": "Sensitivity", "specificity": "Specificity", "accuracy": "Accuracy", "f1": "F1", "roc_auc": "ROC-AUC"}

SURFACE, TEXT, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SUBTITLE = "3-fold CV grouped by patient · 3784 recordings · preliminary run (CNNs: 1 epoch)"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT,
    "font.size": 10, "savefig.dpi": 160, "savefig.bbox": "tight"
})


def plot_table():
    columns = ["Rank", "Model", "Family"] + list(METRICS.values())
    cells = [
        [str(rank), name, FAMILY[name]] + [f"{values[m][0]:.3f} ± {values[m][1]:.3f}" for m in METRICS]
        for rank, (name, (_, values, _)) in enumerate(RESULTS.items(), start=1)
    ]

    figure, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    table.auto_set_column_width([0, 1, 2])

    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor("#f1f0ec" if row == 0 else ("#e6f0fc" if row == 1 else SURFACE))
        if row in (0, 1):
            cell.set_text_props(weight="bold")

    ax.set_title("Heart-sound classification - benchmark of 5 models (mean ± std over 3 folds)", pad=14)
    figure.text(0.5, 0.02, SUBTITLE + " · selected: Random Forest", ha="center", color=TEXT_SECONDARY, fontsize=9)
    figure.savefig(RESULTS_DIR / "benchmark_table.png")
    plt.close(figure)


def plot_metrics():
    shown = ["macc", "sensitivity", "specificity", "roc_auc"]
    names = list(RESULTS)[::-1]
    colors = [RESULTS[n][0] for n in names]

    figure, axes = plt.subplots(1, len(shown), figsize=(15, 3.8), sharey=True)

    for ax, metric in zip(axes, shown):
        values = [RESULTS[n][1][metric][0] for n in names]
        errors = [RESULTS[n][1][metric][1] for n in names]

        ax.barh(names, values, xerr=errors, color=colors, height=0.6,
                error_kw={"ecolor": TEXT_SECONDARY, "capsize": 3, "elinewidth": 1})

        # Label after the whisker so they never overlap
        for y, (value, error) in enumerate(zip(values, errors)):
            ax.text(value + error + 0.02, y, f"{value:.3f}", va="center", fontsize=9)

        ax.set_xlim(0, 1.25)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_title(METRICS[metric])
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    figure.suptitle("Model comparison - mean over 3 folds (whiskers: std)", fontweight="bold", y=1.03)
    figure.text(0.5, -0.04, SUBTITLE, ha="center", color=TEXT_SECONDARY, fontsize=9)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "metrics_comparison.png")
    plt.close(figure)


def plot_folds():
    figure, ax = plt.subplots(figsize=(8, 4.2))

    for x, (name, (color, values, folds)) in enumerate(RESULTS.items()):
        ax.scatter([x] * len(folds), folds, s=80, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.hlines(np.mean(folds), x - 0.25, x + 0.25, color=TEXT, linewidth=2, zorder=4)
        ax.text(x + 0.3, np.mean(folds), f"{np.mean(folds):.3f}", va="center", fontsize=9)

    ax.set_xticks(range(len(RESULTS)), list(RESULTS))
    ax.set_ylabel("MAcc = (sensitivity + specificity) / 2")
    ax.set_title("Stability across the 3 folds: MAcc of each fold (dots) and mean (bar)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    figure.text(0.5, -0.02, SUBTITLE, ha="center", color=TEXT_SECONDARY, fontsize=9)
    figure.savefig(RESULTS_DIR / "folds_macc.png")
    plt.close(figure)


def plot_sensitivity_specificity():
    """
    Screening trade-off: the ideal model is in the top-right corner.
    """

    figure, ax = plt.subplots(figsize=(6.5, 5.5))

    for name, (color, values, _) in RESULTS.items():
        (sp, sp_std), (se, se_std) = values["specificity"], values["sensitivity"]
        ax.errorbar(sp, se, xerr=sp_std, yerr=se_std, fmt="o", color=color, markersize=10,
                    markeredgecolor=SURFACE, markeredgewidth=2, ecolor=color, elinewidth=1, capsize=3)
        ax.annotate(name, (sp, se), xytext=(8, 8), textcoords="offset points", fontsize=9)

    ax.set_xlim(0.4, 1)
    ax.set_ylim(0.4, 1)
    ax.set_xlabel("Specificity (normal hearts recognized)")
    ax.set_ylabel("Sensitivity (abnormal hearts detected)")
    ax.set_title("Sensitivity / specificity trade-off")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    figure.text(0.5, -0.02, "mean ± std over 3 folds · preliminary run", ha="center", color=TEXT_SECONDARY, fontsize=9)
    figure.savefig(RESULTS_DIR / "sensitivity_specificity.png")
    plt.close(figure)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_table()
    plot_metrics()
    plot_folds()
    plot_sensitivity_specificity()
    print("Figures written to", RESULTS_DIR)
