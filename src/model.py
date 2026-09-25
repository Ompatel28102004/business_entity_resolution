"""
Pair-classification models.

Implements the four model families the challenge methodology calls for, all
CPU-friendly and scikit-learn-native (no GPU, no external services):

  Model A: rule-based / weighted-similarity baseline (no training; a fixed
           linear combination of a few strong similarity features).
  Model B: Logistic Regression (scikit-learn), with feature standardization.
  Model C: Random Forest / Extra Trees (scikit-learn ensemble).
  Model D: HistGradientBoostingClassifier (scikit-learn) -- a strong,
           natively CPU-optimized gradient boosting implementation
           (histogram-based, similar algorithm family to LightGBM/XGBoost)
           that ships inside scikit-learn's BSD-3 license, avoiding an extra
           GPU-oriented dependency while remaining well within the
           MIT/Apache-2.0-compatible, <=8B-parameter licensing constraint
           (this is a small tree ensemble, not a parameterized neural net).

All models expose a common ``.predict_proba_pair(X) -> np.ndarray`` so the
rest of the pipeline (thresholding, inference) is model-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class RuleBasedModel:
    """Model A: a fixed, hand-weighted linear combination of similarity features.

    Not fit to data (no parameters learned) -- serves as the "does a
    trained model actually help" baseline required by the ablation study.
    Weights were chosen from domain reasoning (name similarity matters most,
    address corroborates, exact/rare-token agreement is a strong bonus) and
    are NOT tuned on validation data, to keep it an honest zero-training
    baseline.
    """

    name_weight: float = 0.5
    addr_weight: float = 0.3
    exact_bonus_weight: float = 0.2

    def predict_proba_pair(self, X: pd.DataFrame) -> np.ndarray:
        name_sim = X.get("name_token_sort_ratio", X.get("name_ratio"))
        addr_sim = X.get("addr_token_sort_ratio", X.get("addr_ratio"))
        exact = X.get("name_exact_compact_match", 0) * 0.6 + X.get("addr_number_exact_seq", 0) * 0.4
        score = self.name_weight * name_sim + self.addr_weight * addr_sim + self.exact_bonus_weight * exact
        return np.clip(score.to_numpy() if hasattr(score, "to_numpy") else np.asarray(score), 0, 1)

    def fit(self, X, y):  # no-op, kept for interface parity
        return self


def build_logistic_regression() -> Pipeline:
    """Model B: scaled Logistic Regression -- fast, interpretable linear baseline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def build_random_forest() -> RandomForestClassifier:
    """Model C: Random Forest -- non-linear, robust to feature scale, CPU-parallel."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
    )


def build_extra_trees() -> ExtraTreesClassifier:
    """Model C (variant): Extra Trees -- usually a bit faster than RF, similar accuracy."""
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
    )


def build_hist_gradient_boosting() -> HistGradientBoostingClassifier:
    """Model D: HistGradientBoostingClassifier -- the strongest CPU-friendly model tried."""
    return HistGradientBoostingClassifier(
        max_depth=8,
        max_iter=300,
        learning_rate=0.08,
        l2_regularization=1.0,
        random_state=config.RANDOM_SEED,
    )


class SklearnModelWrapper:
    """Thin wrapper giving any fitted scikit-learn classifier a ``predict_proba_pair`` method."""

    def __init__(self, estimator, feature_names: Sequence[str]):
        self.estimator = estimator
        self.feature_names = list(feature_names)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnModelWrapper":
        self.estimator.fit(X[self.feature_names], y)
        return self

    def predict_proba_pair(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.feature_names])[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: Path) -> "SklearnModelWrapper":
        obj = joblib.load(path)
        return cls(obj["estimator"], obj["feature_names"])


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "extra_trees": build_extra_trees,
    "hist_gradient_boosting": build_hist_gradient_boosting,
}


def train_model(name: str, X: pd.DataFrame, y: pd.Series, feature_names: Sequence[str]) -> SklearnModelWrapper:
    """Train one of the named scikit-learn models on (X[feature_names], y)."""
    estimator = MODEL_BUILDERS[name]()
    wrapper = SklearnModelWrapper(estimator, feature_names)
    wrapper.fit(X, y)
    return wrapper
