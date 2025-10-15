"""Data loading and splitting utilities bound to the central config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from number_pricing.config import CONFIG
from number_pricing.utils.logging_utils import get_logger
from number_pricing.utils.validation import ValidationResult, validate_and_clean_dataset

LOGGER = get_logger(__name__)


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    validation: ValidationResult


class DatasetLoader:
    """Load raw data, apply validation, and perform controlled splits."""

    def __init__(self) -> None:
        self.config = CONFIG
        self.id_column = self.config.data.id_column
        self.target_column = self.config.data.target_column
        backend = getattr(self.config.runtime, "pandas_backend", None)
        if backend:
            try:
                pd.options.mode.data_manager = backend  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - option may not exist in older pandas
                LOGGER.debug("Pandas backend '%s' not available in this environment.", backend)

    def load(self) -> LoadedDataset:
        path = self.config.paths.raw_dataset_path
        LOGGER.info("Loading dataset from %s", path)
        df = pd.read_csv(
            path,
            delimiter=self.config.data.delimiter,
            encoding=self.config.data.encoding,
            dtype=self.config.data.dtype_overrides,
        )
        validation = validate_and_clean_dataset(df)
        LOGGER.info("Dataset validation summary: %s", validation.summary())

        df[self.target_column] = pd.to_numeric(
            df[self.target_column], errors="coerce"
        ).astype(float)

        if self.config.data.clip_target:
            lower_q = self.config.data.clip_lower_quantile
            upper_q = self.config.data.clip_upper_quantile
            lower = df[self.target_column].quantile(lower_q)
            upper = df[self.target_column].quantile(upper_q)
            clip_mask = (df[self.target_column] < lower) | (df[self.target_column] > upper)
            clipped = int(clip_mask.sum())
            if clipped:
                LOGGER.info(
                    "Clipping %s target values outside quantiles %.3f - %.3f (%.2f - %.2f)",
                    clipped,
                    lower_q,
                    upper_q,
                    lower,
                    upper,
                )
                df[self.target_column] = df[self.target_column].clip(lower, upper)
            validation.clipped_targets = clipped

        return LoadedDataset(frame=df, validation=validation)

    @staticmethod
    def _create_stratification_series(y: pd.Series) -> Optional[pd.Series]:
        if y.nunique() < 2:
            return None

        bins = min(10, y.nunique())
        try:
            stratify = pd.qcut(y, q=bins, duplicates="drop")
        except ValueError:
            return None
        return stratify

    def split(
        self, frame: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        test_size = self.config.training.test_size
        if not 0 < test_size < 1:
            raise ValueError("CONFIG.training.test_size must be between 0 and 1.")

        stratify = None
        if "stratified" in self.config.training.validation_strategy:
            stratify_series = self._create_stratification_series(frame[self.target_column])
            if stratify_series is not None:
                stratify = stratify_series
                LOGGER.info("Using stratified hold-out split.")
            else:
                LOGGER.info("Unable to stratify hold-out split; proceeding without stratification.")

        X_train, X_test, y_train, y_test = train_test_split(
            frame[self.id_column],
            frame[self.target_column],
            test_size=test_size,
            random_state=self.config.training.random_seed,
            stratify=stratify,
            shuffle=self.config.training.validation_shuffle,
        )

        return (
            pd.DataFrame({self.id_column: X_train}),
            pd.DataFrame({self.id_column: X_test}),
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
        )
