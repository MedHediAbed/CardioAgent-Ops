"""
The 5 heart-sound classifiers compared in train.py:

    ML (on the MFCC feature vector):  random_forest, svm, xgboost
    DL (on the signal itself):        cnn_1d (raw signal), cnn_2d (MFCC matrix)

Every model is wrapped in HeartSoundModel, so they all take the same input
(preprocessed signals, shape (n, 10000)) and return the probability of
"abnormal". The audio-engine can serve any of them without knowing which one it is.
"""
import mlflow.pyfunc
import numpy as np

from audio_processing import extract_features, extract_mfcc


ML_MODELS = ("random_forest", "svm", "xgboost")
DL_MODELS = ("cnn_1d", "cnn_2d")
ALL_MODELS = ML_MODELS + DL_MODELS

DEFAULT_PARAMS = {
    "random_forest": {"n_estimators": 300, "max_depth": None},
    "svm": {"C": 10.0, "gamma": "scale"},
    "xgboost": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    "cnn_1d": {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3},
    "cnn_2d": {"epochs": 30, "batch_size": 32, "learning_rate": 1e-3}
}


def signals_to_features(signals: np.ndarray):
    return np.stack([extract_features(extract_mfcc(signal)) for signal in signals])


def signals_to_mfcc(signals: np.ndarray):
    return np.stack([extract_mfcc(signal) for signal in signals])


# ---------------------------------------------------------------- ML models

def build_ml_model(name: str, params: dict, labels: np.ndarray):
    """
    Scaler + classifier. Every classifier compensates for the class imbalance
    (abnormal recordings are the minority).
    """

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "random_forest":
        classifier = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **params)

    elif name == "svm":
        classifier = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42, **params)

    elif name == "xgboost":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            scale_pos_weight=(labels == 0).sum() / max((labels == 1).sum(), 1),
            eval_metric="logloss",
            random_state=42,
            **params
        )

    else:
        raise ValueError(f"Unknown ML model: {name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])


def train_ml_model(name: str, params: dict, signals: np.ndarray, labels: np.ndarray):
    model = build_ml_model(name, params, labels)
    model.fit(signals_to_features(signals), labels)

    return HeartSoundModel(name, model)


# ---------------------------------------------------------------- DL models

def build_cnn(name: str):
    import torch.nn as nn

    if name == "cnn_1d":
        # (n, 1, 10000) raw signal
        def block(c_in, c_out):
            return [nn.Conv1d(c_in, c_out, kernel_size=9, padding=4), nn.BatchNorm1d(c_out), nn.ReLU()]

        return nn.Sequential(
            *block(1, 16), nn.MaxPool1d(4),
            *block(16, 32), nn.MaxPool1d(4),
            *block(32, 64), nn.MaxPool1d(4),
            *block(64, 64), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    if name == "cnn_2d":
        # (n, 1, N_MFCC, frames) MFCC matrix
        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]

        return nn.Sequential(
            *block(1, 16), nn.MaxPool2d(2),
            *block(16, 32), nn.MaxPool2d(2),
            *block(32, 64), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    raise ValueError(f"Unknown DL model: {name}")


def train_dl_model(name: str, params: dict, signals: np.ndarray, labels: np.ndarray, log_prefix: str = ""):
    """
    Train a CNN with a weighted loss (class imbalance).
    The training loss of every epoch is logged to the active MLflow run.
    """

    import torch

    torch.manual_seed(42)

    wrapper = HeartSoundModel(name, build_cnn(name))

    if name == "cnn_2d":
        # Standardize each MFCC coefficient with the training-set statistics
        mfcc = signals_to_mfcc(signals)
        wrapper.mfcc_mean = mfcc.mean(axis=(0, 2), keepdims=True)[0]
        wrapper.mfcc_std = mfcc.std(axis=(0, 2), keepdims=True)[0] + 1e-6

    X = wrapper.to_tensor(signals)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    network = wrapper.estimator
    optimizer = torch.optim.Adam(network.parameters(), lr=params["learning_rate"])
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)], dtype=torch.float32)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(params["epochs"]):
        network.train()
        permutation = torch.randperm(len(X))
        total_loss = 0.0

        for start in range(0, len(X), params["batch_size"]):
            batch = permutation[start:start + params["batch_size"]]

            optimizer.zero_grad()
            loss = loss_function(network(X[batch]), y[batch])
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        mlflow.log_metric(f"{log_prefix}train_loss", total_loss / len(X), step=epoch)

    network.eval()

    return wrapper


def train_model(name: str, params: dict, signals: np.ndarray, labels: np.ndarray, log_prefix: str = ""):
    if name in ML_MODELS:
        return train_ml_model(name, params, signals, labels)

    return train_dl_model(name, params, signals, labels, log_prefix)


# ---------------------------------------------------------------- MLflow wrapper

class HeartSoundModel(mlflow.pyfunc.PythonModel):
    """
    Common interface of the 5 models, saved to MLflow as a pyfunc model.
    Input: preprocessed signals (n, 10000). Output: probability of "abnormal".
    """

    def __init__(self, name: str, estimator):
        self.name = name
        self.estimator = estimator
        self.mfcc_mean = None
        self.mfcc_std = None

    def to_tensor(self, signals: np.ndarray):
        import torch

        if self.name == "cnn_1d":
            x = signals[:, np.newaxis, :]
        else:
            x = ((signals_to_mfcc(signals) - self.mfcc_mean) / self.mfcc_std)[:, np.newaxis, :, :]

        return torch.tensor(x, dtype=torch.float32)

    def predict_proba(self, signals: np.ndarray):
        signals = np.asarray(signals, dtype=np.float32)

        if self.name in ML_MODELS:
            return self.estimator.predict_proba(signals_to_features(signals))[:, 1]

        import torch

        with torch.no_grad():
            return torch.sigmoid(self.estimator(self.to_tensor(signals))).squeeze(1).numpy()

    def predict(self, context, model_input, params=None):
        if hasattr(model_input, "to_numpy"):
            model_input = model_input.to_numpy()

        return self.predict_proba(model_input)
