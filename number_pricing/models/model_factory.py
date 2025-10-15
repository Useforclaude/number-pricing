"""Factory helpers to assemble estimators and pipelines."""

from __future__ import annotations

from typing import Dict, Type

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from number_pricing.config import CONFIG
from number_pricing.features.feature_extractor import NumberFeatureTransformer
from number_pricing.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


ESTIMATOR_REGISTRY: Dict[str, Type] = {
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "ridge": Ridge,
}


def instantiate_estimator(name: str, params: Dict[str, object] | None = None):
    estimator_cls = ESTIMATOR_REGISTRY.get(name)
    if estimator_cls is None:
        raise ValueError(
            f"Unknown estimator '{name}'. "
            f"Available options: {', '.join(sorted(ESTIMATOR_REGISTRY))}"
        )

    params = params or {}
    LOGGER.info("Initialising estimator %s with params %s", name, params)
    estimator = estimator_cls(**params)
    early_rounds = getattr(CONFIG.training, "early_stopping_rounds", None)
    if early_rounds and hasattr(estimator, "set_params"):
        try:
            estimator.set_params(n_iter_no_change=early_rounds)
        except ValueError:
            LOGGER.debug(
                "Estimator %s does not support n_iter_no_change adjustment.",
                name,
            )
    return estimator


def build_estimator(overrides: dict | None = None):
    settings = CONFIG.model
    params = settings.hyperparameters.copy()
    if overrides:
        params.update(overrides)
    return instantiate_estimator(settings.estimator_name, params)


def build_model_pipeline(overrides: dict | None = None):
    from number_pricing.models.ensemble import WeightedEnsembleRegressor

    if CONFIG.model.use_ensemble:
        return WeightedEnsembleRegressor(
            members=CONFIG.model.ensemble_members,
            feature_scaling=CONFIG.model.feature_scaling,
            primary_overrides=overrides,
        )

    steps = [("features", NumberFeatureTransformer())]

    if CONFIG.model.feature_scaling.lower() == "standard":
        steps.append(("scaler", StandardScaler()))

    steps.append(("regressor", build_estimator(overrides=overrides)))
    return Pipeline(steps=steps)
