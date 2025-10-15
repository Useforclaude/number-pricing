"""
Central configuration module for the phone-number pricing project.

All environment detection, path resolution, feature definitions, model
hyperparameters, and training controls live in this file so other modules
can remain free from hard-coded values. Update values here (or via the
documented environment variables) to tune behaviour across local machines,
Colab, Kaggle, or Paperspace without touching any other source file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


# --------------------------------------------------------------------------------------
# Environment & path resolution helpers
# --------------------------------------------------------------------------------------

_DEFAULT_ENVIRONMENT_HINT = os.getenv("NUMBER_PRICING_ENV", "auto").lower()

# These defaults can be overridden through environment variables:
# - NUMBER_PRICING_PROJECT_ROOT: Manually set the project base directory
# - NUMBER_PRICING_ARTIFACT_DIR: Override where derived assets should be written
# - NUMBER_PRICING_DATASET_FILENAME: Force a specific raw dataset filename

_DEFAULT_ENV_BASE_PATHS: Dict[str, Path] = {
    "local": Path(__file__).resolve().parent.parent,
    "colab": Path("/content/drive/MyDrive/number-ML"),
    "colab_ephemeral": Path("/content/number-ML"),
    "kaggle": Path("/kaggle/working"),
    "paperspace": Path("/notebooks/number-pricing"),
}


def _auto_detect_environment() -> Tuple[str, Path]:
    """Detect the runtime environment and return its name plus the base path."""
    explicit_root = os.getenv("NUMBER_PRICING_PROJECT_ROOT")
    if explicit_root:
        return "custom", Path(explicit_root).resolve()

    if _DEFAULT_ENVIRONMENT_HINT != "auto":
        env_name = _DEFAULT_ENVIRONMENT_HINT
    elif "PAPERSPACE_NOTEBOOK_REPO_ID" in os.environ or os.path.exists("/storage"):
        env_name = "paperspace"
    elif "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
        env_name = "kaggle"
    elif os.path.exists("/content"):
        # Distinguish between Drive-mounted and ephemeral Colab sessions
        drive_path = _DEFAULT_ENV_BASE_PATHS["colab"]
        env_name = "colab" if drive_path.exists() else "colab_ephemeral"
    else:
        env_name = "local"

    base_path = _DEFAULT_ENV_BASE_PATHS.get(
        env_name, Path(__file__).resolve().parent.parent
    )
    return env_name, base_path.resolve()


def _resolve_artifact_root(base_path: Path) -> Path:
    """Return the directory that will hold generated assets."""
    override = os.getenv("NUMBER_PRICING_ARTIFACT_DIR")
    if override:
        return Path(override).resolve()
    return (base_path / "number_pricing_artifacts").resolve()


def _resolve_dataset_filename(default_name: str) -> str:
    """Allow overriding the raw dataset filename via environment variable."""
    return os.getenv("NUMBER_PRICING_DATASET_FILENAME", default_name)


# --------------------------------------------------------------------------------------
# Dataclasses representing configuration sections
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PathSettings:
    environment_name: str
    project_root: Path
    artifact_root: Path
    raw_data_dir: Path
    raw_dataset_filename: str
    processed_data_dir: Path
    feature_store_dir: Path
    models_dir: Path
    reports_dir: Path
    cache_dir: Path
    logs_dir: Path

    @property
    def raw_dataset_path(self) -> Path:
        return self.raw_data_dir / self.raw_dataset_filename


@dataclass(frozen=True)
class DataSettings:
    raw_dataset_filename: str
    delimiter: str
    encoding: str
    id_column: str
    target_column: str
    enforce_unique_ids: bool
    valid_number_lengths: Tuple[int, ...]
    drop_rows_with_missing_target: bool
    dtype_overrides: Dict[str, str]
    cache_intermediate: bool

    @staticmethod
    def defaults() -> Dict[str, Any]:
        return {
            "raw_dataset_filename": _resolve_dataset_filename("numberdata.csv"),
            "delimiter": ",",
            "encoding": "utf-8",
            "id_column": "phone_num",
            "target_column": "price",
            "enforce_unique_ids": True,
            "valid_number_lengths": (10,),
            "drop_rows_with_missing_target": True,
            "dtype_overrides": {"phone_num": "string"},
            "cache_intermediate": True,
        }


@dataclass(frozen=True)
class FeatureSettings:
    digit_symbols: Tuple[str, ...]
    include_digit_counts: bool
    include_digit_shares: bool
    include_position_scores: bool
    include_pattern_flags: bool
    include_ngram_counts: bool
    ngram_sizes: Tuple[int, ...]
    premium_pairs: Tuple[str, ...]
    penalty_pairs: Tuple[str, ...]
    lucky_digits: Tuple[str, ...]
    unlucky_digits: Tuple[str, ...]
    rolling_window_sizes: Tuple[int, ...]
    aggregation_stats: Tuple[str, ...]
    positional_groups: Dict[str, Tuple[int, ...]]
    power_weights: Dict[str, float]
    special_pair_scores: Dict[str, float]
    premium_suffix_weights: Dict[str, float]
    premium_prefix_weights: Dict[str, float]
    ending_premium_scores: Dict[str, float]
    lucky_sequence_scores: Dict[str, float]
    double_scores: Dict[str, float]
    triple_scores: Dict[str, float]
    quad_scores: Dict[str, float]
    high_value_digits: Tuple[str, ...]
    rare_digits: Tuple[str, ...]
    negative_pairs: Tuple[str, ...]
    mystical_pairs: Dict[str, float]


@dataclass(frozen=True)
class ModelSettings:
    estimator_name: str
    hyperparameters: Dict[str, Any]
    calibration: bool
    calibration_cv: int
    feature_scaling: str
    artifact_name: str
    use_ensemble: bool
    ensemble_members: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class HyperparameterSearchSettings:
    enabled: bool
    strategy: str
    candidates: Tuple[Dict[str, Any], ...]
    result_report_name: str


@dataclass(frozen=True)
class TrainingSettings:
    validation_strategy: str
    validation_folds: int
    validation_shuffle: bool
    test_size: float
    random_seed: int
    scoring_metric: str
    minimize_metric: bool
    target_transform: str
    early_stopping_rounds: int
    store_train_predictions: bool
    metrics_report_name: str
    cv_report_name: str
    holdout_predictions_name: str
    oof_predictions_name: str
    hyperparameter_search: HyperparameterSearchSettings


@dataclass(frozen=True)
class EvaluationSettings:
    metrics: Tuple[str, ...]
    ranking_percentiles: Tuple[int, ...]
    calibration_bins: int
    generate_feature_importance: bool
    shap_sample_size: int


@dataclass(frozen=True)
class LoggingSettings:
    level: str
    format: str
    rotation: str
    max_bytes: int
    backup_count: int
    propagate: bool


@dataclass(frozen=True)
class RuntimeSettings:
    preferred_training_targets: Tuple[str, ...]
    fail_fast: bool
    num_workers: int
    pandas_backend: str
    progress_bar: bool
    default_prediction_output_name: str


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathSettings
    data: DataSettings
    features: FeatureSettings
    model: ModelSettings
    training: TrainingSettings
    evaluation: EvaluationSettings
    logging: LoggingSettings
    runtime: RuntimeSettings


# --------------------------------------------------------------------------------------
# Configuration builder
# --------------------------------------------------------------------------------------


def _build_path_settings(
    environment_name: str, project_root: Path, dataset_filename: str
) -> PathSettings:
    artifact_root = _resolve_artifact_root(project_root)
    return PathSettings(
        environment_name=environment_name,
        project_root=project_root,
        artifact_root=artifact_root,
        raw_data_dir=(project_root / "data" / "raw").resolve(),
        raw_dataset_filename=dataset_filename,
        processed_data_dir=(artifact_root / "processed").resolve(),
        feature_store_dir=(artifact_root / "features").resolve(),
        models_dir=(artifact_root / "models").resolve(),
        reports_dir=(artifact_root / "reports").resolve(),
        cache_dir=(artifact_root / "cache").resolve(),
        logs_dir=(artifact_root / "logs").resolve(),
    )


def _build_feature_settings() -> FeatureSettings:
    # หมายเหตุ: ถ้าไม่ต้องการใช้ฟีเจอร์จากความเชื่อ หรือคู่เลขต่าง ๆ
    # ให้ปรับ tuple ต่าง ๆ (เช่น premium_pairs, lucky_digits) เป็น tuple() ได้เลย
    # และตั้ง include_* ที่เกี่ยวข้องเป็น False เพื่อปิดการสร้างฟีเจอร์กลุ่มนั้น
    return FeatureSettings(
        digit_symbols=tuple(str(d) for d in range(10)),
        include_digit_counts=True,
        include_digit_shares=True,
        include_position_scores=True,
        include_pattern_flags=True,
        include_ngram_counts=True,
        ngram_sizes=(2, 3, 4),
        premium_pairs=(
            "639",
            "45",
            "54",
            "56",
            "65",
            "789",
            "519",
            "145",
            "915",
            "195",
            "415",
        ),
        penalty_pairs=("00", "13", "31", "47", "74", "17", "71", "25", "52", "77", "76", "67"),
        lucky_digits=("9", "6", "5"),
        unlucky_digits=("0", "7"),
        rolling_window_sizes=(2, 3, 5),
        aggregation_stats=("mean", "std", "min", "max"),
        positional_groups={
            "leading": (0, 1, 2),
            "middle": (3, 4, 5, 6),
            "trailing": (7, 8, 9),
        },
        power_weights={
            "5": 10,
            "9": 8,
            "8": 6,
            "6": 8,
            "3": 4,
            "4": 6,
            "1": 5,
            "7": -2,
            "0": -5,
            "2": -1,
        },
        special_pair_scores={
            "15": 10,
            "51": 10,
            "24": 8,
            "42": 8,
            "36": 9,
            "63": 9,
            "45": 12,
            "54": 12,
            "56": 15,
            "65": 15,
            "69": 10,
            "96": 10,
            "78": 12,
            "87": 12,
            "28": 8,
            "82": 8,
            "38": 7,
            "83": 7,
            "19": 9,
            "91": 9,
            "89": 11,
            "98": 11,
            "14": 10,
            "41": 10,
            "59": 13,
            "95": 13,
            "88": 14,
            "99": 16,
            "66": 12,
            "55": 14,
            "32": -5,
            "23": -5,
            "35": 9,
            "53": 9,
        },
        premium_suffix_weights={
            "9999": 200,
            "8888": 150,
            "6666": 120,
            "5555": 100,
            "999": 100,
            "888": 80,
            "666": 60,
            "555": 50,
            "99": 40,
            "88": 30,
            "66": 20,
            "55": 15,
            "89": 25,
            "98": 25,
            "56": 20,
            "65": 20,
            "789": 45,
            "456": 40,
            "123": 35,
            "678": 35,
            "5678": 60,
            "6789": 65,
            "4567": 55,
            "3456": 50,
            "1234": 45,
            "2345": 40,
            "639": 25,
            "6395": 35,
            "63915": 40,
            "6365": 35,
            "6595": 25,
            "6515": 25,
            "5195": 25,
            "5915": 25,
            "4159": 20,
            "4156": 25,
            "4165": 20,
            "2456": 30,
            "2465": 25,
            "4265": 25,
            "4256": 20,
            "1456": 30,
            "1465": 25,
            "1965": 22,
            "1956": 25,
            "3656": 30,
            "6356": 25,
            "3665": 20,
            "9156": 25,
            "63965": 40,
            "6465": 20,
            "465": 20,
            "956": 25,
            "965": 20,
            "5456": 30,
            "4556": 22,
            "5156": 25,
            "6456": 20,
            "9456": 25,
        },
        premium_prefix_weights={
            "089": 0.35,
            "088": 0.45,
            "086": 0.32,
            "080": 0.30,
            "081": 0.28,
            "082": 0.26,
            "092": 0.30,
            "091": 0.28,
            "090": 0.32,
        },
        ending_premium_scores={
            "9999": 200,
            "8888": 150,
            "6666": 120,
            "5555": 100,
            "999": 100,
            "888": 80,
            "666": 60,
            "555": 50,
            "99": 40,
            "88": 30,
            "66": 20,
            "55": 15,
            "789": 30,
            "6789": 65,
            "5678": 60,
            "4567": 55,
            "3456": 50,
            "1234": 45,
            "2345": 40,
            "456": 25,
            "639": 25,
            "6395": 35,
            "63915": 40,
            "3656": 30,
            "9156": 25,
        },
        lucky_sequence_scores={
            "123": 35,
            "234": 30,
            "345": 30,
            "456": 40,
            "567": 45,
            "678": 50,
            "789": 55,
            "890": 25,
            "987": 35,
            "876": 30,
            "765": 25,
            "111": 30,
            "222": -20,
            "333": 25,
            "444": 40,
            "555": 50,
            "666": 60,
            "777": -30,
            "888": 80,
            "999": 100,
            "000": -50,
        },
        double_scores={
            "00": 2,
            "11": 6,
            "22": 3,
            "33": 5,
            "44": 6,
            "55": 10,
            "66": 9,
            "77": 4,
            "88": 12,
            "99": 14,
        },
        triple_scores={
            "000": 3,
            "111": 8,
            "222": 4,
            "333": 7,
            "444": 8,
            "555": 15,
            "666": 12,
            "777": 6,
            "888": 20,
            "999": 20,
        },
        quad_scores={
            "0000": 5,
            "1111": 15,
            "2222": 6,
            "3333": 10,
            "4444": 12,
            "5555": 20,
            "6666": 18,
            "7777": 14,
            "8888": 30,
            "9999": 30,
        },
        high_value_digits=("7", "8", "9"),
        rare_digits=("0", "3", "4"),
        negative_pairs=(
            "10",
            "01",
            "12",
            "21",
            "13",
            "31",
            "17",
            "71",
            "18",
            "81",
            "20",
            "02",
            "27",
            "72",
            "30",
            "03",
            "34",
            "43",
            "37",
            "73",
            "38",
            "83",
            "48",
            "84",
            "67",
            "76",
            "68",
            "86",
            "70",
            "07",
            "80",
            "08",
            "32",
            "23",
        ),
        mystical_pairs={
            "13": -10,
            "31": -8,
            "17": -5,
            "71": -5,
            "04": -8,
            "40": -8,
            "20": -6,
            "02": -6,
            "27": -7,
            "72": -7,
            "70": -9,
            "07": -9,
            "08": -3,
            "80": -3,
            "48": -4,
            "84": -4,
            "21": -3,
            "12": -3,
            "37": -5,
            "73": -5,
            "29": -4,
            "92": -4,
            "18": 5,
            "81": 5,
            "16": 6,
            "61": 6,
            "22": -8,
            "77": -10,
            "00": -15,
            "44": 8,
            "33": -5,
            "11": 3,
        },
    )


def _build_model_settings() -> ModelSettings:
    # หมายเหตุ: เปลี่ยนชนิดโมเดลได้โดยปรับ estimator_name ให้ตรงกับ key ใน ESTIMATOR_REGISTRY
    # ปรับพารามิเตอร์อื่น ๆ ได้ใน hyperparameters โดยไม่ต้องแก้ไฟล์โค้ดส่วนอื่น
    return ModelSettings(
        estimator_name="hist_gradient_boosting",
        hyperparameters={
            "max_depth": None,
            "min_samples_leaf": 20,
            "max_leaf_nodes": 31,
            "learning_rate": 0.06,
            "l2_regularization": 0.0,
            "max_iter": 600,
            "validation_fraction": 0.1,
            "n_iter_no_change": 30,
            "tol": 1e-4,
        },
        calibration=False,
        calibration_cv=3,
        feature_scaling="none",
        artifact_name="hist_gradient_boosting_number_pricing.joblib",
        use_ensemble=True,
        ensemble_members=(
            {
                "name": "hist_gradient_boosting",
                "weight": 0.55,
                "hyperparameters": {
                    "max_depth": None,
                    "min_samples_leaf": 20,
                    "max_leaf_nodes": 55,
                    "learning_rate": 0.06,
                    "l2_regularization": 0.0,
                    "max_iter": 950,
                    "validation_fraction": 0.1,
                    "n_iter_no_change": 30,
                    "tol": 1e-4,
                },
            },
            {
                "name": "gradient_boosting",
                "weight": 0.25,
                "hyperparameters": {
                    "n_estimators": 800,
                    "learning_rate": 0.05,
                    "max_depth": 5,
                    "min_samples_split": 4,
                    "min_samples_leaf": 2,
                    "subsample": 0.9,
                    "max_features": "sqrt",
                },
            },
            {
                "name": "extra_trees",
                "weight": 0.20,
                "hyperparameters": {
                    "n_estimators": 600,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": 0.8,
                    "bootstrap": False,
                },
            },
        ),
    )


def _build_training_settings() -> TrainingSettings:
    # หมายเหตุ: ปรับแผนแบ่งข้อมูลและ metric ได้ที่นี่ เช่น validation_strategy, test_size, scoring_metric
    # หากไม่ต้องการบันทึก OOF ให้ตั้ง store_train_predictions เป็น False
    return TrainingSettings(
        validation_strategy="stratified_kfold",
        validation_folds=5,
        validation_shuffle=True,
        test_size=0.15,
        random_seed=42,
        scoring_metric="rmse",
        minimize_metric=True,
        target_transform="log1p",
        early_stopping_rounds=50,
        store_train_predictions=True,
        metrics_report_name="training_metrics.json",
        cv_report_name="cross_validation_metrics.json",
        holdout_predictions_name="holdout_predictions.csv",
        oof_predictions_name="oof_predictions.csv",
        hyperparameter_search=_build_hyperparameter_search_settings(),
    )


def _build_hyperparameter_search_settings() -> HyperparameterSearchSettings:
    return HyperparameterSearchSettings(
        enabled=True,
        strategy="grid",
        candidates=(
            {"learning_rate": 0.10, "max_leaf_nodes": 24, "min_samples_leaf": 18, "max_iter": 600},
            {"learning_rate": 0.08, "max_leaf_nodes": 36, "min_samples_leaf": 14, "max_iter": 850},
            {"learning_rate": 0.07, "max_leaf_nodes": 48, "min_samples_leaf": 12, "max_iter": 1000},
            {"learning_rate": 0.06, "max_leaf_nodes": 55, "min_samples_leaf": 18, "max_iter": 1100},
            {"learning_rate": 0.05, "max_leaf_nodes": 68, "min_samples_leaf": 10, "max_iter": 1300},
            {"learning_rate": 0.045, "max_leaf_nodes": 85, "min_samples_leaf": 8, "max_iter": 1500},
            {"learning_rate": 0.04, "max_leaf_nodes": 95, "min_samples_leaf": 6, "max_iter": 1700},
            {"learning_rate": 0.035, "max_leaf_nodes": 120, "min_samples_leaf": 5, "max_iter": 2000},
        ),
        result_report_name="hyperparameter_search.json",
    )


def _build_evaluation_settings() -> EvaluationSettings:
    return EvaluationSettings(
        metrics=("rmse", "mae", "mape", "r2"),
        ranking_percentiles=(5, 10, 25),
        calibration_bins=10,
        generate_feature_importance=True,
        shap_sample_size=500,
    )


def _build_logging_settings() -> LoggingSettings:
    return LoggingSettings(
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        rotation="day",
        max_bytes=10_000_000,
        backup_count=5,
        propagate=False,
    )


def _build_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        preferred_training_targets=("local", "paperspace", "colab", "kaggle"),
        fail_fast=False,
        num_workers=os.cpu_count() or 4,
        pandas_backend="pyarrow",
        progress_bar=True,
        default_prediction_output_name="inference_predictions.csv",
    )


def build_project_config() -> ProjectConfig:
    """Return the fully-populated project configuration."""
    environment_name, base_path = _auto_detect_environment()
    data_settings = DataSettings(**DataSettings.defaults())
    paths = _build_path_settings(
        environment_name, base_path, data_settings.raw_dataset_filename
    )

    return ProjectConfig(
        paths=paths,
        data=data_settings,
        features=_build_feature_settings(),
        model=_build_model_settings(),
        training=_build_training_settings(),
        evaluation=_build_evaluation_settings(),
        logging=_build_logging_settings(),
        runtime=_build_runtime_settings(),
    )


# Exposed singleton configuration used by the rest of the project.
CONFIG = build_project_config()


def ensure_directories(paths: Iterable[Path]) -> None:
    """
    Create any missing directories for the supplied iterable of path objects.

    This helper keeps directory creation logic in a single place to avoid
    unchecked side-effects inside other modules.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


# Make sure key directories exist when the module is imported.
ensure_directories(
    (
        CONFIG.paths.artifact_root,
        CONFIG.paths.processed_data_dir,
        CONFIG.paths.feature_store_dir,
        CONFIG.paths.models_dir,
        CONFIG.paths.reports_dir,
        CONFIG.paths.cache_dir,
        CONFIG.paths.logs_dir,
    )
)
