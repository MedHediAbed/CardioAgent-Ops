"""
Benchmark report of the 5 heart-sound models, generated from MLflow:

    python generate_report.py                  # last finished "comparison" run
    python generate_report.py --run-id <id>    # a specific comparison run

Reads the runs logged by ../train.py and writes into results/:

    benchmark_table.csv / .md / .png   mean ± std over the 3 folds
    metrics_comparison.png             MAcc, sensitivity, specificity, ROC-AUC
    confusion_matrices.png             out-of-fold predictions
    roc_curves.png                     out-of-fold predictions
    folds_macc.png                     stability across the 3 folds
    training_loss.png                  training curves of the 2 CNNs
    report.md                          everything above on one page

"Out-of-fold": every recording is scored by the model of the fold where it was
in the test set, i.e. by a model that never saw it (nor its patient).

The results folder is also logged to the MLflow comparison run (artifacts/benchmark/).
MLflow server: MLFLOW_TRACKING_URI (default: http://localhost:5000).
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix, roc_curve


EXPERIMENT_NAME = "heart-sound-classification"
MODEL_NAME = os.getenv("MODEL_NAME", "heart-sound-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
RESULTS_DIR = Path(__file__).parent / "results"

MODELS = {
    # run name: (display name, family, color) - one fixed color per model in every figure
    "random_forest": ("Random Forest", "ML", "#2a78d6"),
    "svm": ("SVM", "ML", "#eb6834"),
    "xgboost": ("XGBoost", "ML", "#1baf7a"),
    "cnn_1d": ("1D-CNN", "DL", "#eda100"),
    "cnn_2d": ("2D-CNN", "DL", "#e87ba4")
}

METRICS = {
    "macc": "MAcc",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "accuracy": "Accuracy",
    "f1": "F1",
    "roc_auc": "ROC-AUC"
}

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "text.color": TEXT,
    "font.size": 10,
    "savefig.dpi": 160,
    "savefig.bbox": "tight"
})


# ---------------------------------------------------------------- read MLflow

def find_comparison_run(client: MlflowClient, run_id: str | None):
    if run_id:
        return client.get_run(run_id)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise SystemExit(f"No experiment '{EXPERIMENT_NAME}': run ../train.py first")

    runs = client.search_runs(
        [experiment.experiment_id],
        "attributes.run_name = 'comparison' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        raise SystemExit("No finished 'comparison' run: run ../train.py first")

    return runs[0]


def load_model_runs(client: MlflowClient, parent):
    """
    One entry per child run: metrics, params, per-fold values,
    out-of-fold predictions and training loss.
    """

    children = client.search_runs(
        [parent.info.experiment_id],
        f"tags.mlflow.parentRunId = '{parent.info.run_id}'"
    )
    results = []

    for run in children:
        name = run.info.run_name

        if name not in MODELS:
            continue

        def history(metric):
            points = sorted(client.get_metric_history(run.info.run_id, metric), key=lambda m: m.step)
            return [point.value for point in points]

        try:
            oof_path = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="oof_predictions.csv")
            oof = pd.read_csv(oof_path)
        except Exception:
            oof = None   # runs logged before train.py saved the predictions

        results.append({
            "name": name,
            "run_id": run.info.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "fold_macc": history("fold_macc"),
            "train_loss": history("final_train_loss"),
            "oof": oof
        })

    # Best first (selection metric of train.py)
    return sorted(results, key=lambda r: r["metrics"]["macc"], reverse=True)


# ---------------------------------------------------------------- table

def build_table(results: list[dict]):
    rows = []

    for result in results:
        display, family, _ = MODELS[result["name"]]
        row = {"model": display, "family": family}

        for metric in METRICS:
            row[metric] = result["metrics"][metric]
            row[f"{metric}_std"] = result["metrics"].get(f"{metric}_std", 0.0)

        row["cv_time_min"] = result["metrics"].get("cv_time_s", float("nan")) / 60
        rows.append(row)

    return pd.DataFrame(rows)


def table_markdown(table: pd.DataFrame):
    header = "| Rank | Model | Family | " + " | ".join(METRICS.values()) + " | CV time (min) |"
    lines = [header, "|---" * (len(METRICS) + 4) + "|"]

    for rank, row in enumerate(table.itertuples(), start=1):
        cells = [f"{getattr(row, m):.3f} ± {getattr(row, m + '_std'):.3f}" for m in METRICS]
        model = f"**{row.model}**" if rank == 1 else row.model
        lines.append(f"| {rank} | {model} | {row.family} | " + " | ".join(cells) + f" | {row.cv_time_min:.1f} |")

    return "\n".join(lines)


def plot_table(table: pd.DataFrame, path: Path):
    columns = ["Rank", "Model", "Family"] + list(METRICS.values())
    cells = [
        [str(rank), row.model, row.family] + [f"{getattr(row, m):.3f} ± {getattr(row, m + '_std'):.3f}" for m in METRICS]
        for rank, row in enumerate(table.itertuples(), start=1)
    ]

    figure, axes = plt.subplots(figsize=(13, 0.5 + 0.45 * len(cells)))
    axes.axis("off")

    chart = axes.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    chart.auto_set_font_size(False)
    chart.set_fontsize(10)
    chart.scale(1, 1.6)

    for (row, _), cell in chart.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)

        if row == 0:
            cell.set_text_props(weight="bold", color=TEXT)
            cell.set_facecolor("#f1f0ec")
        elif row == 1:
            cell.set_text_props(weight="bold")   # best model

    axes.set_title("Heart-sound classification - 3-fold cross-validation (mean ± std)", pad=12)
    figure.savefig(path)
    plt.close(figure)


# ---------------------------------------------------------------- figures

def plot_metrics(results: list[dict], path: Path):
    """
    One panel per metric, same model order and colors in every panel.
    """

    shown = ["macc", "sensitivity", "specificity", "roc_auc"]
    names = [MODELS[r["name"]][0] for r in results][::-1]
    colors = [MODELS[r["name"]][2] for r in results][::-1]

    figure, axes = plt.subplots(1, len(shown), figsize=(15, 3.6), sharey=True)

    for ax, metric in zip(axes, shown):
        values = [r["metrics"][metric] for r in results][::-1]
        errors = [r["metrics"].get(f"{metric}_std", 0) for r in results][::-1]

        ax.barh(names, values, xerr=errors, color=colors, height=0.6,
                error_kw={"ecolor": TEXT_SECONDARY, "capsize": 3, "elinewidth": 1})

        for y, value in enumerate(values):
            ax.text(min(value + 0.02, 1.0), y, f"{value:.3f}", va="center", fontsize=9, color=TEXT)

        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_title(METRICS[metric])
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    figure.suptitle("Mean over the 3 folds (whiskers: std)", color=TEXT_SECONDARY, fontsize=10, y=0.02)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_confusion_matrices(results: list[dict], path: Path):
    with_oof = [r for r in results if r["oof"] is not None]

    if not with_oof:
        return False

    figure, axes = plt.subplots(1, len(with_oof), figsize=(3.4 * len(with_oof), 3.6))
    axes = np.atleast_1d(axes)

    for ax, result in zip(axes, with_oof):
        oof = result["oof"]
        predictions = (oof["probability_abnormal"] >= 0.5).astype(int)
        matrix = confusion_matrix(oof["label"], predictions, labels=[0, 1])
        rates = matrix / matrix.sum(axis=1, keepdims=True)

        ax.imshow(rates, cmap=SEQUENTIAL, vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                color = "white" if rates[i, j] > 0.55 else TEXT
                ax.text(j, i, f"{matrix[i, j]}\n{rates[i, j]:.0%}", ha="center", va="center", color=color, fontsize=10)

        ax.set_xticks([0, 1], ["normal", "abnormal"])
        ax.set_yticks([0, 1], ["normal", "abnormal"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{MODELS[result['name']][0]}\nMAcc {result['metrics']['macc']:.3f}", fontsize=10)

        for spine in ax.spines.values():
            spine.set_visible(False)

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)

    return True


def plot_roc(results: list[dict], path: Path):
    with_oof = [r for r in results if r["oof"] is not None]

    if not with_oof:
        return False

    figure, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_SECONDARY, linewidth=1, label="Chance")

    for result in with_oof:
        display, _, color = MODELS[result["name"]]
        fpr, tpr, _ = roc_curve(result["oof"]["label"], result["oof"]["probability_abnormal"])

        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{display} (AUC {result['metrics']['roc_auc']:.3f})")

    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curves - out-of-fold predictions")
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)

    figure.savefig(path)
    plt.close(figure)

    return True


def plot_folds(results: list[dict], path: Path):
    figure, ax = plt.subplots(figsize=(7.5, 4))

    for x, result in enumerate(results):
        display, _, color = MODELS[result["name"]]
        values = result["fold_macc"]

        ax.scatter([x] * len(values), values, s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.hlines(np.mean(values), x - 0.25, x + 0.25, color=TEXT, linewidth=2, zorder=4)

    ax.set_xticks(range(len(results)), [MODELS[r["name"]][0] for r in results])
    ax.set_ylabel("MAcc")
    ax.set_title("MAcc of each fold (dots) and mean (bar)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    figure.savefig(path)
    plt.close(figure)


def plot_training_loss(results: list[dict], path: Path):
    with_loss = [r for r in results if r["train_loss"]]

    if not with_loss:
        return False

    figure, ax = plt.subplots(figsize=(7, 4))

    for result in with_loss:
        display, _, color = MODELS[result["name"]]
        epochs = range(1, len(result["train_loss"]) + 1)

        ax.plot(epochs, result["train_loss"], color=color, linewidth=2, marker="o", markersize=4, label=display)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE loss")
    ax.set_title("CNN training loss (final model, whole dataset)")
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False)

    figure.savefig(path)
    plt.close(figure)

    return True


# ---------------------------------------------------------------- report

def production_version(client: MlflowClient):
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        return f"version {version.version} ({version.tags.get('model_type')})"
    except Exception:
        return "none"


def write_report(parent, results: list[dict], table: pd.DataFrame, figures: dict, production: str):
    params = parent.data.params
    best = results[0]
    best_name = MODELS[best["name"]][0]

    lines = [
        "# Heart-sound classification - model benchmark",
        "",
        f"Generated on {datetime.now():%Y-%m-%d %H:%M} from MLflow run `{parent.info.run_id}` "
        f"(experiment `{EXPERIMENT_NAME}`) by `benchmark/generate_report.py`.",
        "",
        "## Protocol",
        "",
        f"- **Data:** {params.get('n_recordings')} PCG recordings, {params.get('n_patients')} patients, "
        f"{params.get('n_abnormal')} abnormal ({int(params.get('n_abnormal', 0)) / max(int(params.get('n_recordings', 1)), 1):.0%}).",
        f"- **Preprocessing:** resampled to {params.get('sample_rate')} Hz, normalized, cut/padded to "
        f"{params.get('duration_s')} s, {params.get('n_mfcc')} MFCCs.",
        f"- **Evaluation:** {params.get('n_folds')}-fold cross-validation grouped by patient "
        "(each fold: ~66% train / ~34% test, a patient is never in both).",
        "- **Selection metric:** MAcc = (sensitivity + specificity) / 2, the PhysioNet/CinC 2016 score. "
        "Accuracy is misleading here: always answering \"normal\" already scores ~77%.",
        "",
        "| Model | Family | Input |",
        "|---|---|---|",
        "| Random Forest, SVM, XGBoost | ML | 26 features: mean + std of the 13 MFCCs |",
        "| 1D-CNN | DL | raw signal (10 000 samples) |",
        "| 2D-CNN | DL | MFCC matrix (13 × 20) |",
        "",
        "## Results",
        "",
        table_markdown(table),
        "",
        f"**Best model: {best_name}** (MAcc {best['metrics']['macc']:.3f} ± {best['metrics'].get('macc_std', 0):.3f}). "
        f"Registry `{MODEL_NAME}@{MODEL_ALIAS}`: {production}.",
        "",
        "![Metrics](metrics_comparison.png)",
        "",
        "![Stability across folds](folds_macc.png)",
        ""
    ]

    if figures.get("confusion"):
        lines += ["## Confusion matrices (out-of-fold)", "", "![Confusion matrices](confusion_matrices.png)", ""]

    if figures.get("roc"):
        lines += ["## ROC curves (out-of-fold)", "", "![ROC curves](roc_curves.png)", ""]

    if figures.get("loss"):
        lines += ["## CNN training", "", "![Training loss](training_loss.png)", ""]

    lines += ["## Hyper-parameters", "", "| Model | Parameters | MLflow run |", "|---|---|---|"]

    for result in results:
        parameters = ", ".join(f"{k}={v}" for k, v in sorted(result["params"].items()))
        lines.append(f"| {MODELS[result['name']][0]} | {parameters} | `{result['run_id']}` |")

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd audio-engine",
        "python train.py ../ourdataset/ourdataset/train --promote   # trains + logs to MLflow",
        "python benchmark/generate_report.py                        # this report",
        "```",
        ""
    ]

    (RESULTS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(run_id: str | None):
    client = MlflowClient()
    parent = find_comparison_run(client, run_id)
    results = load_model_runs(client, parent)

    if not results:
        raise SystemExit(f"Comparison run {parent.info.run_id} has no model run")

    RESULTS_DIR.mkdir(exist_ok=True)

    table = build_table(results)
    table.round(4).to_csv(RESULTS_DIR / "benchmark_table.csv", index=False)
    (RESULTS_DIR / "benchmark_table.md").write_text(table_markdown(table) + "\n", encoding="utf-8")
    plot_table(table, RESULTS_DIR / "benchmark_table.png")

    plot_metrics(results, RESULTS_DIR / "metrics_comparison.png")
    plot_folds(results, RESULTS_DIR / "folds_macc.png")

    figures = {
        "confusion": plot_confusion_matrices(results, RESULTS_DIR / "confusion_matrices.png"),
        "roc": plot_roc(results, RESULTS_DIR / "roc_curves.png"),
        "loss": plot_training_loss(results, RESULTS_DIR / "training_loss.png")
    }

    write_report(parent, results, table, figures, production_version(client))

    # Keep the report next to the runs it describes
    client.log_artifacts(parent.info.run_id, str(RESULTS_DIR), "benchmark")

    print(table_markdown(table))
    print(f"\nReport: {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark report of the heart-sound models")
    parser.add_argument("--run-id", help="MLflow comparison run (default: the latest)")
    args = parser.parse_args()

    main(args.run_id)
