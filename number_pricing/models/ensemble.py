"""Custom weighted ensemble regressor for combining multiple pipelines."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from number_pricing.features.feature_extractor import NumberFeatureTransformer
from number_pricing.models.model_factory import instantiate_estimator


class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Train multiple estimators and blend their predictions with configured weights."""

    def __init__(
        self,
        members: Tuple[Dict[str, object], ...],
        feature_scaling: str = "none",
        primary_overrides: Dict[str, float] | None = None,
    ) -> None:
        self.members = members
        self.feature_scaling = feature_scaling
        self.primary_overrides = primary_overrides or {}
        self._pipelines: List[Tuple[float, Pipeline]] = []
        self._total_weight: float = 0.0

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        params = {
            "members": self.members,
            "feature_scaling": self.feature_scaling,
            "primary_overrides": self.primary_overrides,
        }
        if not deep:
            return params
        deep_params: Dict[str, object] = {}
        for key, value in params.items():
            deep_params[key] = value
        return deep_params

    def set_params(self, **params: object) -> "WeightedEnsembleRegressor":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        self._pipelines = []
        self._total_weight = 0.0
        for idx, member in enumerate(self.members):
            name = member.get("name")
            weight = float(member.get("weight", 1.0))
            hyperparameters = dict(member.get("hyperparameters", {}))
            if idx == 0 and self.primary_overrides:
                hyperparameters.update(self.primary_overrides)

            estimator = instantiate_estimator(name, hyperparameters)
            steps = [("features", NumberFeatureTransformer())]
            if self.feature_scaling.lower() == "standard":
                steps.append(("scaler", StandardScaler()))
            steps.append(("regressor", estimator))
            pipeline = Pipeline(steps)
            pipeline.fit(X, y)
            self._pipelines.append((weight, pipeline))
            self._total_weight += weight

        if self._total_weight == 0:
            self._total_weight = float(len(self._pipelines))
        return self

    def predict(self, X):
        if not self._pipelines:
            raise RuntimeError("WeightedEnsembleRegressor has not been fitted yet.")

        blended = None
        for weight, pipeline in self._pipelines:
            preds = pipeline.predict(X)
            if blended is None:
                blended = weight * preds
            else:
                blended += weight * preds
        return blended / self._total_weight
