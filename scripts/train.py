"""Command-line entry point to launch the training pipeline."""

from __future__ import annotations

from number_pricing.pipelines.training_pipeline import TrainingPipeline
from number_pricing.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    pipeline = TrainingPipeline()
    artifacts = pipeline.run()
    LOGGER.info("Training artefacts saved to: %s", artifacts)


if __name__ == "__main__":
    main()

