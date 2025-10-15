"""Data validation helpers to keep preprocessing free from leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import pandas as pd

from number_pricing.config import CONFIG
from number_pricing.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ValidationResult:
    total_rows: int
    removed_invalid_length: int
    removed_missing_target: int
    dropped_duplicates: int
    clipped_targets: int = 0

    def summary(self) -> Dict[str, int]:
        return {
            "total_rows": self.total_rows,
            "removed_invalid_length": self.removed_invalid_length,
            "removed_missing_target": self.removed_missing_target,
            "dropped_duplicates": self.dropped_duplicates,
            "clipped_targets": self.clipped_targets,
        }


def _normalise_numbers(series: pd.Series) -> pd.Series:
    # Ensure consistent formatting before validation
    return (
        series.astype("string")
        .str.replace(r"\D", "", regex=True)
        .str.strip()
        .astype("string")
    )


def validate_and_clean_dataset(df: pd.DataFrame) -> ValidationResult:
    """
    Clean the frame in-place according to CONFIG.data rules and return stats.
    """
    id_col = CONFIG.data.id_column
    target_col = CONFIG.data.target_column
    valid_lengths: Sequence[int] = CONFIG.data.valid_number_lengths

    df[id_col] = _normalise_numbers(df[id_col])
    total_rows = len(df)

    invalid_length_mask = ~df[id_col].str.len().isin(valid_lengths)
    removed_invalid_length = int(invalid_length_mask.sum())
    if removed_invalid_length:
        LOGGER.info("Dropping %s rows with invalid phone length", removed_invalid_length)
        df.drop(index=df[invalid_length_mask].index, inplace=True)

    removed_missing_target = 0
    if CONFIG.data.drop_rows_with_missing_target:
        missing_target_mask = df[target_col].isna()
        removed_missing_target = int(missing_target_mask.sum())
        if removed_missing_target:
            LOGGER.info("Dropping %s rows with missing target", removed_missing_target)
            df.drop(index=df[missing_target_mask].index, inplace=True)

    dropped_duplicates = 0
    if CONFIG.data.enforce_unique_ids:
        duplicate_mask = df[id_col].duplicated(keep="first")
        dropped_duplicates = int(duplicate_mask.sum())
        if dropped_duplicates:
            LOGGER.info("Dropping %s duplicate phone numbers", dropped_duplicates)
            df.drop(index=df[duplicate_mask].index, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return ValidationResult(
        total_rows=total_rows,
        removed_invalid_length=removed_invalid_length,
        removed_missing_target=removed_missing_target,
        dropped_duplicates=dropped_duplicates,
    )
