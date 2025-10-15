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
from number_pricing.models._registry import instantiate_estimator
from number_pricing.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_estimator(overrides: dict | None = None):
    settings = CONFIG.model
    params = settings.hyperparameters.copy()
    if overrides:
        params.update(overrides)
    return instantiate_estimator(settings.estimator_name, params)


def build_model_pipeline(overrides: dict | None = None):
    from number_pricing.models.ensemble import WeightedEnsembleRegressor
    from sklearn.ensemble import StackingRegressor

    steps = [("features", NumberFeatureTransformer())]

    if CONFIG.model.feature_scaling.lower() == "standard":
        steps.append(("scaler", StandardScaler()))

    if CONFIG.model.use_ensemble:
        strategy = CONFIG.model.ensemble_strategy.lower()
        if strategy == "weighted":
            return WeightedEnsembleRegressor(
                members=CONFIG.model.ensemble_members,
                feature_scaling=CONFIG.model.feature_scaling,
                primary_overrides=overrides,
            )
        if strategy == "stacking":
            estimators = []
            for idx, member in enumerate(CONFIG.model.ensemble_members):
                name = member.get("name")
                weight = member.get("weight", 1.0)  # for reference
                hyperparams = dict(member.get("hyperparameters", {}))
                if idx == 0 and overrides:
                    hyperparams.update(overrides)
                pipeline_steps = steps.copy()
                pipeline_steps.append(
                    ("regressor", instantiate_estimator(name, hyperparams))
                )
                estimators.append((f"{name}_{idx}", Pipeline(pipeline_steps)))

            meta_cfg = CONFIG.model.stacking_meta
            meta_estimator = instantiate_estimator(
                meta_cfg["name"], dict(meta_cfg.get("params", {}))
            )
            return StackingRegressor(
                estimators=estimators,
                final_estimator=meta_estimator,
                passthrough=meta_cfg.get("passthrough", False),
                cv=CONFIG.training.validation_folds,
                n_jobs=-1,
            )

    steps.append(("regressor", build_estimator(overrides=overrides)))
    return Pipeline(steps=steps)
