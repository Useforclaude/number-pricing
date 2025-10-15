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


def build_estimator(overrides: dict | None = None):
    settings = CONFIG.model
    estimator_cls = ESTIMATOR_REGISTRY.get(settings.estimator_name)
    if estimator_cls is None:
        raise ValueError(
            f"Unknown estimator '{settings.estimator_name}'. "
            f"Available options: {', '.join(sorted(ESTIMATOR_REGISTRY))}"
        )

    params = settings.hyperparameters.copy()
    if overrides:
        params.update(overrides)

    LOGGER.info(
        "Initialising estimator %s with params %s",
        settings.estimator_name,
        params,
    )
    estimator = estimator_cls(**params)

    early_rounds = getattr(CONFIG.training, "early_stopping_rounds", None)
    if early_rounds and hasattr(estimator, "set_params"):
        try:
            estimator.set_params(n_iter_no_change=early_rounds)
        except ValueError:
            LOGGER.debug(
                "Estimator %s does not support n_iter_no_change adjustment.",
                settings.estimator_name,
            )

    return estimator


def build_model_pipeline(overrides: dict | None = None) -> Pipeline:
    steps = [("features", NumberFeatureTransformer())]

    if CONFIG.model.feature_scaling.lower() == "standard":
        steps.append(("scaler", StandardScaler()))

    steps.append(("regressor", build_estimator(overrides=overrides)))
    pipeline = Pipeline(steps=steps)
    return pipeline
