import os

import numpy as np


MODEL_NAME = os.getenv("MODEL_NAME", "heart-sound-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

THRESHOLD = 0.5

# Loaded lazily from the MLflow model registry, once.
# If loading fails we keep the error and do not retry on every request
# (the MLflow client retries for a long time) -> call reload_model() instead.
_model = None
_version = None
_load_error = None


def get_model():
    """
    Load the model the "production" alias points to (any of the 5 models:
    they all take the preprocessed signal as input).
    Returns None if no model is available.
    """

    global _model, _version, _load_error

    if _model is None and _load_error is None:
        try:
            import mlflow.pyfunc
            from mlflow.tracking import MlflowClient

            version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

            _model = mlflow.pyfunc.load_model(MODEL_URI)
            _version = {
                "version": version.version,
                "model_type": version.tags.get("model_type"),
                "run_id": version.run_id
            }
        except Exception as error:
            _load_error = str(error)

    return _model


def reload_model():
    """
    Forget the cached model (e.g. after promoting a new version) and load it again.
    """

    global _model, _version, _load_error

    _model = None
    _version = None
    _load_error = None

    return get_model()


def model_status():
    return {
        "model_uri": MODEL_URI,
        "loaded": _model is not None,
        **(_version or {}),
        "error": _load_error
    }


def predict(signal: np.ndarray):
    """
    Classify one preprocessed recording.
    Returns None if no model is available.
    """

    model = get_model()

    if model is None:
        return None

    probability = float(model.predict(signal[np.newaxis, :].astype(np.float32))[0])

    return {
        "label": "abnormal" if probability >= THRESHOLD else "normal",
        "probability_abnormal": round(probability, 4),
        "model_name": MODEL_NAME,
        **_version
    }
