"""
Compare the 5 heart-sound classifiers with 3-fold cross-validation and track
everything in MLflow:

    python train.py ../ourdataset/ourdataset/train
    python train.py ../ourdataset/ourdataset/train --promote
    python train.py ../ourdataset/ourdataset/train --models svm cnn_1d --epochs 5

`data_dir` contains one folder per class: normal/ and abnormal/ (.wav files).

Each fold trains on 2/3 of the patients (66%) and tests on the other 1/3 (34%).
All recordings of one patient (112_..._A.wav, 112_..._B.wav, ...) stay in the
same fold, otherwise the model is tested on patients it has already heard.

MLflow (MLFLOW_TRACKING_URI, default: http://localhost:5000):
    experiment "heart-sound-classification"
      run "comparison"
        └─ 1 child run per model: params, metrics of every fold + mean/std,
           final model trained on the whole dataset
    the best model (mean MAcc) is registered as "heart-sound-classifier"
"""
import argparse
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from audio_processing import N_MFCC, TARGET_DURATION, TARGET_SAMPLE_RATE, fix_audio_length, load_audio, normalize_audio
from heart_models import ALL_MODELS, DEFAULT_PARAMS, DL_MODELS, ML_MODELS, train_model
from model_registry import MODEL_ALIAS, MODEL_NAME


EXPERIMENT_NAME = "heart-sound-classification"
N_FOLDS = 3
CLASSES = {"normal": 0, "abnormal": 1}

# PhysioNet/CinC 2016 score: mean of sensitivity and specificity.
# Accuracy alone is misleading here: predicting "normal" for everyone already gives 77%.
SELECTION_METRIC = "macc"

# Files shipped with the logged model, so MLflow can unpickle HeartSoundModel anywhere
CODE_PATHS = [
    str(Path(__file__).parent / "heart_models.py"),
    str(Path(__file__).parent / "audio_processing.py")
]


def patient_id(wav_path: Path):
    """
    "112_1306243000964_A.wav" -> "112_1306243000964" (A, B, C, D = auscultation sites).
    """

    return wav_path.stem.rsplit("_", 1)[0]


def load_dataset(data_dir: str):
    """
    Read every .wav of normal/ and abnormal/ and preprocess it
    exactly like the audio-engine does at inference time.
    """

    signals, labels, groups = [], [], []

    for class_name, label in CLASSES.items():
        class_dir = Path(data_dir) / class_name

        if not class_dir.is_dir():
            raise SystemExit(f"Missing folder: {class_dir}")

        files = sorted(class_dir.glob("*.wav"))
        print(f"Loading {len(files)} {class_name} recordings...")

        for wav_path in files:
            signal, _ = load_audio(str(wav_path))

            signals.append(fix_audio_length(normalize_audio(signal)))
            labels.append(label)
            groups.append(patient_id(wav_path))

    return np.array(signals, dtype=np.float32), np.array(labels), np.array(groups)


def evaluate(labels: np.ndarray, probabilities: np.ndarray):
    """
    Screening metrics: missing an abnormal heart (sensitivity) is worse
    than a false alarm (specificity).
    """

    predictions = (probabilities >= 0.5).astype(int)

    sensitivity = recall_score(labels, predictions, pos_label=1, zero_division=0)
    specificity = recall_score(labels, predictions, pos_label=0, zero_division=0)

    return {
        "macc": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": float((predictions == labels).mean()),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities)
    }


def model_requirements():
    """
    Pin the libraries the logged model needs (faster and more reliable
    than letting MLflow guess them).
    """

    requirements = []

    for package in ("numpy", "librosa", "scikit-learn", "xgboost", "xgboost-cpu", "torch"):
        try:
            requirements.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            pass

    return requirements


def run_model(name: str, params: dict, signals: np.ndarray, labels: np.ndarray, folds: list):
    """
    One MLflow child run: cross-validation, then the final model trained on all the data.
    """

    with mlflow.start_run(run_name=name, nested=True) as run:
        mlflow.set_tag("model_family", "ml" if name in ML_MODELS else "dl")
        mlflow.log_params(params)

        fold_metrics = []
        oof_probabilities = np.zeros(len(labels))
        oof_folds = np.zeros(len(labels), dtype=int)
        start_time = time.perf_counter()

        for fold, (train_index, test_index) in enumerate(folds, start=1):
            model = train_model(name, params, signals[train_index], labels[train_index], log_prefix=f"fold{fold}_")
            probabilities = model.predict_proba(signals[test_index])
            metrics = evaluate(labels[test_index], probabilities)

            oof_probabilities[test_index] = probabilities
            oof_folds[test_index] = fold

            # step = fold number -> one curve per metric in the MLflow UI
            for metric, value in metrics.items():
                mlflow.log_metric(f"fold_{metric}", value, step=fold)

            fold_metrics.append(metrics)
            print(f"  {name} fold {fold}: macc={metrics['macc']:.3f} auc={metrics['roc_auc']:.3f}")

        summary = {}

        for metric in fold_metrics[0]:
            values = [metrics[metric] for metrics in fold_metrics]
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))

        summary["cv_time_s"] = time.perf_counter() - start_time
        mlflow.log_metrics(summary)

        # Out-of-fold predictions: every recording scored by the model that never saw it.
        # Used by benchmark/ for the confusion matrices and ROC curves.
        rows = ["fold,label,probability_abnormal"] + [
            f"{fold},{label},{probability:.6f}"
            for fold, label, probability in zip(oof_folds, labels, oof_probabilities)
        ]
        mlflow.log_text("\n".join(rows), "oof_predictions.csv")

        # Model to deploy: same recipe, trained on the whole dataset
        final_model = train_model(name, params, signals, labels, log_prefix="final_")
        example = signals[:2]

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=final_model,
            code_paths=CODE_PATHS,
            signature=infer_signature(example, final_model.predict_proba(example)),
            pip_requirements=model_requirements()
        )

    return {"name": name, "run_id": run.info.run_id, "model_uri": model_info.model_uri, **summary}


def print_comparison(results: list[dict]):
    """
    Markdown table, printed and saved in the parent run.
    """

    columns = ["macc", "sensitivity", "specificity", "accuracy", "f1", "roc_auc"]
    lines = [
        "| model | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns)
    ]

    for result in sorted(results, key=lambda r: r[SELECTION_METRIC], reverse=True):
        cells = [f"{result[c]:.3f} ± {result[c + '_std']:.3f}" for c in columns]
        lines.append(f"| {result['name']} | " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    print(table)

    return table


def compare(data_dir: str, models: list[str], epochs: int | None, promote: bool):
    signals, labels, groups = load_dataset(data_dir)

    print(f"{len(signals)} recordings, {len(set(groups))} patients, {labels.sum()} abnormal")

    # Same 3 folds for every model -> fair comparison
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(splitter.split(signals, labels, groups))

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="comparison"):
        mlflow.log_params({
            "data_dir": Path(data_dir).name,
            "n_recordings": len(signals),
            "n_patients": len(set(groups)),
            "n_abnormal": int(labels.sum()),
            "n_folds": N_FOLDS,
            "sample_rate": TARGET_SAMPLE_RATE,
            "duration_s": TARGET_DURATION,
            "n_mfcc": N_MFCC,
            "selection_metric": SELECTION_METRIC
        })

        results = []

        for name in models:
            params = dict(DEFAULT_PARAMS[name])

            if epochs and name in DL_MODELS:
                params["epochs"] = epochs

            print(f"Training {name}...")
            results.append(run_model(name, params, signals, labels, folds))

        table = print_comparison(results)
        mlflow.log_text(table, "comparison.md")

        best = max(results, key=lambda r: r[SELECTION_METRIC])

        mlflow.set_tag("best_model", best["name"])
        mlflow.log_metric(f"best_{SELECTION_METRIC}", best[SELECTION_METRIC])

    # Only the winner goes to the model registry
    model_version = mlflow.register_model(best["model_uri"], MODEL_NAME)
    client = MlflowClient()

    client.set_model_version_tag(MODEL_NAME, model_version.version, "model_type", best["name"])
    client.set_model_version_tag(MODEL_NAME, model_version.version, SELECTION_METRIC, f"{best[SELECTION_METRIC]:.4f}")

    print(f"Best: {best['name']} ({SELECTION_METRIC}={best[SELECTION_METRIC]:.3f})")
    print(f"Registered '{MODEL_NAME}' version {model_version.version}")

    if promote:
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, model_version.version)
        print(f"Alias '{MODEL_ALIAS}' -> version {model_version.version}")


if __name__ == "__main__":
    # MLflow prints emojis ("🏃 View run ..."): crashes a Windows cp1252 console otherwise
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Compare the 5 heart-sound classifiers")
    parser.add_argument("data_dir", help="Folder with normal/ and abnormal/ sub-folders of .wav files")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs of the CNNs")
    parser.add_argument("--promote", action="store_true", help=f"Set the '{MODEL_ALIAS}' alias on the best model")
    args = parser.parse_args()

    compare(args.data_dir, args.models, args.epochs, args.promote)
