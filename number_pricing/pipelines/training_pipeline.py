"""End-to-end training pipeline orchestrated through the central config."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, StratifiedKFold

from number_pricing.config import CONFIG
from number_pricing.data.dataset_loader import DatasetLoader
from number_pricing.evaluation.metrics import compute_regression_metrics
from number_pricing.models.model_factory import build_model_pipeline
from number_pricing.utils.io import save_dataframe, save_json, save_model
from number_pricing.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TrainingArtifacts:
    model_path: str
    metrics_path: str
    cv_metrics_path: str
    holdout_predictions_path: str
    oof_predictions_path: str


class TrainingPipeline:
    """Coordinate data loading, model fitting, validation, and persistence."""

    def __init__(self) -> None:
        self.config = CONFIG
        self.loader = DatasetLoader()

    def _initialise_estimator(self):
        estimator = build_model_pipeline()
        if self.config.training.target_transform == "log1p":
            estimator = TransformedTargetRegressor(
                regressor=estimator,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )
        return estimator

    def _build_cv_splitter(
        self, y: pd.Series
    ) -> Tuple[KFold | StratifiedKFold, Optional[pd.Series]]:
        folds = self.config.training.validation_folds
        shuffle = self.config.training.validation_shuffle
        seed = self.config.training.random_seed

        if "stratified" in self.config.training.validation_strategy.lower():
            # Generate quantile bins to retain price distribution
            unique_targets = y.nunique()
            if unique_targets < 2:
                LOGGER.warning("Target has <2 unique values; falling back to standard KFold.")
                return KFold(n_splits=folds, shuffle=shuffle, random_state=seed), None

            bins = min(folds, unique_targets)
            try:
                quantiles = pd.qcut(y, q=bins, duplicates="drop")
                splitter = StratifiedKFold(
                    n_splits=len(quantiles.cat.categories),
                    shuffle=shuffle,
                    random_state=seed,
                )
                return splitter, quantiles.cat.codes
            except ValueError:
                LOGGER.warning("Unable to create quantile bins; using plain KFold.")
                return KFold(n_splits=folds, shuffle=shuffle, random_state=seed), None

        return KFold(n_splits=folds, shuffle=shuffle, random_state=seed), None

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        splitter, strat_labels = self._build_cv_splitter(y)
        fold_metrics: List[Dict[str, float]] = []
        store_oof = self.config.training.store_train_predictions
        oof_predictions = np.full(shape=len(X), fill_value=np.nan) if store_oof else None

        split_iterator = (
            splitter.split(X, strat_labels) if strat_labels is not None else splitter.split(X)
        )

        for fold_index, (train_idx, val_idx) in enumerate(split_iterator, start=1):
            estimator = clone(self._initialise_estimator())
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            start = time.time()
            estimator.fit(X_train, y_train)
            elapsed = time.time() - start

            predictions = estimator.predict(X_val)
            metrics = compute_regression_metrics(
                y_val.reset_index(drop=True), pd.Series(predictions)
            )
            metrics["fold"] = fold_index
            metrics["fit_time_seconds"] = elapsed
            fold_metrics.append(metrics)
            LOGGER.info("Fold %s metrics: %s", fold_index, metrics)

            if store_oof and oof_predictions is not None:
                oof_predictions[val_idx] = predictions

        if not fold_metrics:
            raise RuntimeError("Cross-validation did not yield any folds.")

        aggregated: Dict[str, float] = {}
        metric_keys = {
            key for key in fold_metrics[0] if key not in {"fold", "fit_time_seconds"}
        }
        for key in metric_keys:
            values = [fold[key] for fold in fold_metrics]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values, ddof=0))

        aggregated["fit_time_seconds_mean"] = float(
            np.mean([fold["fit_time_seconds"] for fold in fold_metrics])
        )

        report_path = self.config.paths.reports_dir / self.config.training.cv_report_name
        save_json({"fold_metrics": fold_metrics, "aggregated": aggregated}, report_path)
        oof_frame: Optional[pd.DataFrame] = None
        if store_oof and oof_predictions is not None:
            oof_frame = pd.DataFrame(
                {
                    self.config.data.id_column: X[self.config.data.id_column].reset_index(
                        drop=True
                    ),
                    "actual_price": y.reset_index(drop=True),
                    "oof_predicted_price": oof_predictions,
                }
            )
        return aggregated, oof_frame

    def run(self) -> TrainingArtifacts:
        loaded = self.loader.load()
        frame = loaded.frame

        X_train_df, X_holdout_df, y_train, y_holdout = self.loader.split(frame)

        if (
            self.config.training.target_transform == "log1p"
            and (y_train <= -1).any()
        ):
            raise ValueError(
                "Target values must be greater than -1 to use log1p transformation."
            )

        cv_summary, oof_frame = self._cross_validate(X_train_df, y_train)
        LOGGER.info("Cross-validation summary: %s", cv_summary)

        final_estimator = self._initialise_estimator()
        LOGGER.info("Fitting final model on %s rows", len(X_train_df))
        final_estimator.fit(X_train_df, y_train)

        holdout_predictions = final_estimator.predict(X_holdout_df)
        holdout_metrics = compute_regression_metrics(
            y_holdout.reset_index(drop=True), pd.Series(holdout_predictions)
        )
        LOGGER.info("Hold-out metrics: %s", holdout_metrics)

        primary_metric = self.config.training.scoring_metric
        primary_value = holdout_metrics.get(primary_metric)

        metrics_payload = {
            "holdout_metrics": holdout_metrics,
            "cross_validation_metrics": cv_summary,
            "validation_summary": loaded.validation.summary(),
            "config": {
                "model": self.config.model.estimator_name,
                "environment": self.config.paths.environment_name,
                "primary_metric": primary_metric,
                "primary_metric_value": primary_value,
            },
        }
        metrics_path = self.config.paths.reports_dir / self.config.training.metrics_report_name
        save_json(metrics_payload, metrics_path)

        predictions_frame = pd.DataFrame(
            {
                self.config.data.id_column: X_holdout_df[self.config.data.id_column].reset_index(drop=True),
                "actual_price": y_holdout.reset_index(drop=True),
                "predicted_price": holdout_predictions,
            }
        )
        predictions_path = (
            self.config.paths.reports_dir / self.config.training.holdout_predictions_name
        )
        save_dataframe(predictions_frame, predictions_path)

        if oof_frame is not None:
            oof_path = (
                self.config.paths.reports_dir / self.config.training.oof_predictions_name
            )
            save_dataframe(oof_frame, oof_path)
        else:
            oof_path = ""

        model_path = self.config.paths.models_dir / self.config.model.artifact_name
        save_model(final_estimator, model_path)

        return TrainingArtifacts(
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            cv_metrics_path=str(self.config.paths.reports_dir / self.config.training.cv_report_name),
            holdout_predictions_path=str(predictions_path),
            oof_predictions_path=str(oof_path),
        )
