"""
Inférence RNN : le pickle d’entraînement contient scaler + lookback + feature_columns
(voir notebook), le réseau est dans rnn_direction_model.keras.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

import __main__
from app.ml_logic.features import build_technical_features
import tensorflow.keras.models

KERAS_MODEL_FILENAME = "rnn.keras"
RNN_ARTIFACTS_FILENAME = "rnn_preprocessing.pkl"

__all__ = ["rnn_predict_from_artifacts", "KERAS_MODEL_FILENAME", "RNN_ARTIFACTS_FILENAME"]


def _load_keras_model(path: str):
    try:
        from keras.saving import load_model
    except ImportError:  # pragma: no cover
        from tensorflow.keras.models import load_model
    return load_model(path, compile=False)


def _make_sequences(features_array: np.ndarray, lookback: int) -> np.ndarray:
    """
    X_seq[k] = features_array[k : k+lookback] for k in 0..len-lookback
    (dernier pas utilisé ici, on s’appuie sur l’itération explicite du notebook).
    """
    n = features_array.shape[0]
    if n < lookback:
        return np.empty((0, lookback, features_array.shape[1]), dtype=np.float32)
    out = np.zeros((n - lookback, lookback, features_array.shape[1]), dtype=np.float32)
    for j, i in enumerate(range(lookback, n)):
        out[j] = features_array[i - lookback : i]
    return out


def rnn_predict_from_artifacts(
    df: pd.DataFrame,
    root_path: str,
    *,
    build_features=build_technical_features,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retourne (predictions, probabilités) de même longueur que df.
    Les `lookback` premiers points sont 0.5 / 0 (pas de séquence complète).
    """
    artifacts_path = os.path.join(root_path, "models", RNN_ARTIFACTS_FILENAME)
    keras_path = os.path.join(root_path, "models", KERAS_MODEL_FILENAME)

    __main__.build_technical_features = build_features

    with open(artifacts_path, "rb") as f:
        rnn_artifacts = pickle.load(f)

    if not isinstance(rnn_artifacts, dict):
        raise TypeError(
            f"Fichier RNN inattendu: attendu un dict (scaler, lookback, feature_columns), "
            f"reçu {type(rnn_artifacts)}. Utiliser le pickle produit par le notebook."
        )

    if not os.path.isfile(keras_path):
        raise FileNotFoundError(
            f"Modèle Keras RNN manquant: {keras_path} — exporter le .keras comme dans le notebook."
        )

    scaler = rnn_artifacts["scaler"]
    lookback = int(rnn_artifacts["lookback"])
    feature_columns = list(rnn_artifacts["feature_columns"])
    n_rows = len(df)
    if n_rows == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    sorter: np.ndarray | None = None
    dfo = df.copy()
    if "Date" in dfo.columns:
        sorter = np.argsort(pd.to_datetime(dfo["Date"], utc=False), kind="mergesort")
        dfo = dfo.iloc[sorter].reset_index(drop=True)

    X_raw = build_features(dfo)
    for c in feature_columns:
        if c not in X_raw.columns:
            X_raw[c] = np.nan
    X_raw = X_raw.reindex(columns=feature_columns)
    X_raw = X_raw.ffill().bfill()
    if X_raw.isna().any().any():
        X_raw = X_raw.fillna(0.0)

    X_scaled = scaler.transform(X_raw.values.astype(np.float64))
    n = X_scaled.shape[0]
    proba = np.full(n, 0.5, dtype=np.float64)
    pred = np.zeros(n, dtype=np.int64)

    if n <= lookback:
        if sorter is not None:
            out_p = np.empty(n_rows, dtype=np.float64)
            out_d = np.empty(n_rows, dtype=np.int64)
            out_p[sorter] = proba
            out_d[sorter] = pred
            return out_d, out_p
        return pred, proba

    model = _load_keras_model(keras_path)
    X_seq = _make_sequences(X_scaled, lookback)
    raw = model.predict(X_seq, verbose=0)
    p = np.asarray(raw, dtype=np.float64).reshape(-1)
    proba[lookback:n] = p
    pred[lookback:n] = (p >= 0.5).astype(np.int64)

    if sorter is not None:
        out_proba = np.empty(n_rows, dtype=np.float64)
        out_pred = np.empty(n_rows, dtype=np.int64)
        out_proba[sorter] = proba
        out_pred[sorter] = pred
        return out_pred, out_proba

    return pred, proba
