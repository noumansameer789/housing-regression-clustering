"""Reproducible regression pipeline and honest cluster-feature ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    r2: float


def make_model(numeric: list[str], categorical: list[str], seed: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)),
    ])


def evaluate(model, x_train, y_train, x_test, y_test) -> RegressionMetrics:
    fitted = clone(model).fit(x_train, y_train)
    predictions = fitted.predict(x_test)
    return RegressionMetrics(float(mean_squared_error(y_test, predictions) ** 0.5),
                             float(r2_score(y_test, predictions)))


def choose_k(matrix: np.ndarray, candidates=range(2, 11), seed: int = 42) -> tuple[int, dict[int, float]]:
    scores = {}
    for k in candidates:
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(matrix)
        scores[k] = float(silhouette_score(matrix, labels))
    return max(scores, key=scores.get), scores


def ablation_delta(baseline: RegressionMetrics, augmented: RegressionMetrics) -> float:
    """Positive means the cluster feature reduced RMSE."""
    return baseline.rmse - augmented.rmse
