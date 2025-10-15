"""Prediction pipeline to score new phone numbers with a trained model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from number_pricing.config import CONFIG
from number_pricing.utils.io import load_model
from number_pricing.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class PredictionPipeline:
    """Load a persisted estimator and produce price predictions."""

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        default_path = CONFIG.paths.models_dir / CONFIG.model.artifact_name
        self.model_path = Path(model_path) if model_path else default_path
        self.model = load_model(self.model_path)
        self.id_column = CONFIG.data.id_column

    def predict(self, numbers: Iterable[str]) -> pd.DataFrame:
        frame = pd.DataFrame({self.id_column: list(numbers)})
        predictions = self.model.predict(frame)
        LOGGER.info("Generated predictions for %s numbers", len(frame))
        return pd.DataFrame(
            {self.id_column: frame[self.id_column], "predicted_price": predictions}
        )

