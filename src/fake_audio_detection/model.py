import os
from typing import Any, Dict
import joblib
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .normalization import CustomNormalizer

from .extract_features import extract_all_dataset, extract_features_from_array


def predict_audio_blocks(
    model_path: str, audio_path: str, audio_bytes=None, block_duration: float = 1.0
):
    model_t = joblib.load(model_path)

    y, sr = (
        librosa.load(audio_bytes, sr=22050)
        if audio_bytes is not None
        else librosa.load(audio_path, sr=22050)
    )
    block_size = int(block_duration * sr)
    n_blocks = len(y) // block_size

    times = []
    probas = []

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block = y[start:end]

        features = extract_features_from_array(block, sr).reshape(1, -1)
        proba = model_t["pipeline"].predict_proba(features)[0]

        times.append(i * block_duration)
        probas.append(proba)

    return times, np.array(probas)


def train_svm(
    dataset_path: str,
    save_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    normalisation_type: str = "l2",
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    probability: bool = True,
    class_weight=None,
) -> Dict[str, Any]:
    X, y = extract_all_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = Pipeline(
        [
            ("normalizer", CustomNormalizer(method=normalisation_type)),
            (
                "svm",
                SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    probability=probability,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    model = {"pipeline": pipeline}

    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/my_model.joblib", "wb") as f:
        joblib.dump(model, f)
    print(f"Model saved to: {save_path}/my_model.joblib")

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }

    return metrics
