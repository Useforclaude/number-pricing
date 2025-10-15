"""Generate predictions for new phone numbers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from number_pricing.config import CONFIG
from number_pricing.pipelines.prediction_pipeline import PredictionPipeline
from number_pricing.utils.io import save_dataframe
from number_pricing.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict phone number prices.")
    parser.add_argument(
        "--numbers",
        nargs="+",
        default=None,
        help="List of phone numbers to score.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=f"Path to a file containing a column '{CONFIG.data.id_column}'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIG.paths.reports_dir / CONFIG.runtime.default_prediction_output_name,
        help="Destination CSV for the predictions.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional override for the model artifact path.",
    )
    return parser.parse_args()


def _gather_numbers(args: argparse.Namespace) -> List[str]:
    numbers: List[str] = []
    if args.numbers:
        numbers.extend(args.numbers)
    if args.input_file:
        frame = pd.read_csv(args.input_file)
        if CONFIG.data.id_column not in frame.columns:
            raise ValueError(
                f"Input file must contain column '{CONFIG.data.id_column}'."
            )
        numbers.extend(frame[CONFIG.data.id_column].astype("string").tolist())
    if not numbers:
        raise ValueError("Provide at least one phone number via --numbers or --input-file.")
    return numbers


def main() -> None:
    args = _parse_args()
    numbers = _gather_numbers(args)
    pipeline = PredictionPipeline(model_path=args.model_path)
    predictions = pipeline.predict(numbers)

    save_dataframe(predictions, args.output)
    LOGGER.info("Predictions stored at %s", args.output)


if __name__ == "__main__":
    main()

