"""Estimator registry shared between model factories and ensembles."""

from __future__ import annotations

from typing import Dict, Type

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

ESTIMATOR_REGISTRY: Dict[str, Type] = {
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "ridge": Ridge,
}

try:  # Optional dependencies
    from catboost import CatBoostRegressor  # type: ignore

    ESTIMATOR_REGISTRY["catboost"] = CatBoostRegressor
except ImportError:  # pragma: no cover - optional
    pass

try:
    from lightgbm import LGBMRegressor  # type: ignore

    ESTIMATOR_REGISTRY["lightgbm"] = LGBMRegressor
except ImportError:  # pragma: no cover - optional
    pass


def instantiate_estimator(name: str, params: Dict[str, object] | None = None):
    estimator_cls = ESTIMATOR_REGISTRY.get(name)
    if estimator_cls is None:
        raise ValueError(
            f"Unknown estimator '{name}'. "
            f"Available options: {', '.join(sorted(ESTIMATOR_REGISTRY))}"
        )
    params = params or {}
    if name == "catboost":
        params.setdefault("verbose", 0)
        params.setdefault("loss_function", "RMSE")
    if name == "lightgbm":
        params.setdefault("verbosity", -1)
    return estimator_cls(**params)
