"""I/O helpers that respect the central configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from number_pricing.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def save_json(data: Dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    LOGGER.info("Saved JSON payload to %s", destination)


def save_dataframe(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    LOGGER.info("Saved dataframe with %s rows to %s", len(df), destination)


def save_model(model: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    LOGGER.info("Persisted model to %s", destination)


def load_model(source: Path) -> Any:
    LOGGER.info("Loading model from %s", source)
    return joblib.load(source)

