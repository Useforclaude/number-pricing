"""Evaluation utilities for regression performance analysis."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from number_pricing.config import CONFIG


def _mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute metrics defined in CONFIG.evaluation."""
    y_true_arr = y_true.to_numpy(dtype=float)
    y_pred_arr = y_pred.to_numpy(dtype=float)

    results: Dict[str, float] = {}
    for metric in CONFIG.evaluation.metrics:
        if metric == "rmse":
            results["rmse"] = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        elif metric == "mae":
            results["mae"] = float(mean_absolute_error(y_true_arr, y_pred_arr))
        elif metric == "mape":
            results["mape"] = _mean_absolute_percentage_error(y_true_arr, y_pred_arr)
        elif metric == "r2":
            results["r2"] = float(r2_score(y_true_arr, y_pred_arr))

    residuals = y_true_arr - y_pred_arr
    results["residual_mean"] = float(np.mean(residuals))
    results["residual_std"] = float(np.std(residuals, ddof=0))

    percentiles = CONFIG.evaluation.ranking_percentiles
    for percentile in percentiles:
        cutoff = np.percentile(y_pred_arr, 100 - percentile)
        mask = y_pred_arr >= cutoff
        if np.any(mask):
            results[f"top_{percentile}_pct_pred_mean"] = float(np.mean(y_pred_arr[mask]))
            results[f"top_{percentile}_pct_actual_mean"] = float(np.mean(y_true_arr[mask]))
        else:
            results[f"top_{percentile}_pct_pred_mean"] = 0.0
            results[f"top_{percentile}_pct_actual_mean"] = 0.0

    if CONFIG.evaluation.calibration_bins > 1:
        bins = CONFIG.evaluation.calibration_bins
        calibration_df = pd.DataFrame({"y_true": y_true_arr, "y_pred": y_pred_arr})
        calibration_df["bin"] = pd.qcut(
            calibration_df["y_pred"],
            q=min(bins, len(calibration_df.index)),
            duplicates="drop",
        )
        grouped = calibration_df.groupby("bin")
        calibration_error = (grouped["y_true"].mean() - grouped["y_pred"].mean()).abs().mean()
        results["calibration_mae"] = float(calibration_error)

    return results

