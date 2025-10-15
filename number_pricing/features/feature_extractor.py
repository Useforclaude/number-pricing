"""Feature extraction for phone number strings."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from number_pricing.config import CONFIG
from number_pricing.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _clean_number(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def _max_run_length(number: str) -> int:
    longest = 0
    current = 0
    previous = None
    for digit in number:
        if digit == previous:
            current += 1
        else:
            current = 1
            previous = digit
        longest = max(longest, current)
    return longest


def _count_patterns(number: str, patterns: Iterable[str]) -> int:
    total = 0
    for pattern in patterns:
        plen = len(pattern)
        if plen == 0:
            continue
        total += sum(
            1 for idx in range(len(number) - plen + 1) if number[idx : idx + plen] == pattern
        )
    return total


def _ngram_statistics(number: str, n: int) -> Dict[str, float]:
    if n <= 0 or len(number) < n:
        return {
            f"ngram_{n}_total": 0.0,
            f"ngram_{n}_unique": 0.0,
            f"ngram_{n}_max_frequency": 0.0,
            f"ngram_{n}_entropy": 0.0,
        }
    ngrams = [number[idx : idx + n] for idx in range(len(number) - n + 1)]
    counter = Counter(ngrams)
    total = float(len(ngrams))
    max_freq = float(max(counter.values()))
    entropy = 0.0
    for count in counter.values():
        prob = count / total
        entropy -= prob * math.log(prob + 1e-12, 2)
    return {
        f"ngram_{n}_total": total,
        f"ngram_{n}_unique": float(len(counter)),
        f"ngram_{n}_max_frequency": max_freq,
        f"ngram_{n}_entropy": entropy,
    }


class NumberFeatureTransformer(BaseEstimator, TransformerMixin):
    """Build model-ready numerical features from raw phone numbers."""

    def __init__(self) -> None:
        self.config = CONFIG
        self.feature_names_: List[str] = []
        self.id_column = self.config.data.id_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # type: ignore[override]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        if isinstance(X, pd.Series):
            numbers = X.astype("string")
        elif isinstance(X, pd.DataFrame):
            numbers = X[self.id_column].astype("string")
        else:
            numbers = pd.Series(X, dtype="string")

        cleaned = numbers.apply(_clean_number)
        feature_rows: List[Dict[str, float]] = [self._extract_features(num) for num in cleaned]
        frame = pd.DataFrame(feature_rows)
        if not self.feature_names_:
            self.feature_names_ = frame.columns.tolist()
        return frame[self.feature_names_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # type: ignore[override]
        if not self.feature_names_:
            raise RuntimeError("Transformer must be fitted before accessing feature names.")
        return np.array(self.feature_names_)

    def _extract_features(self, number: str) -> Dict[str, float]:
        digits = [int(ch) for ch in number] if number else []
        length = len(digits)
        features: Dict[str, float] = {
            "length": float(length),
            "max_run_length": float(_max_run_length(number)),
            "is_palindrome": float(number == number[::-1]),
        }

        if digits:
            digit_array = np.array(digits, dtype=float)
            for stat in self.config.features.aggregation_stats:
                if stat == "mean":
                    features["digit_mean"] = float(digit_array.mean())
                elif stat == "std":
                    features["digit_std"] = float(digit_array.std(ddof=0))
                elif stat == "min":
                    features["digit_min"] = float(digit_array.min())
                elif stat == "max":
                    features["digit_max"] = float(digit_array.max())
            features["digit_sum"] = float(digit_array.sum())
            features["unique_digit_count"] = float(len(set(digits)))
            features["unique_digit_ratio"] = (
                features["unique_digit_count"] / length if length else 0.0
            )
            if length > 1:
                diffs = np.diff(digit_array)
                abs_diffs = np.abs(diffs)
                features["mean_digit_diff"] = float(diffs.mean())
                features["std_digit_diff"] = float(diffs.std(ddof=0))
                features["mean_abs_digit_diff"] = float(abs_diffs.mean())
                features["positive_diff_ratio"] = float(np.mean(diffs > 0))
                features["zero_diff_ratio"] = float(np.mean(diffs == 0))
                features["negative_diff_ratio"] = float(np.mean(diffs < 0))
                features["sign_changes_count"] = float(
                    np.sum(np.sign(diffs[:-1]) * np.sign(diffs[1:]) < 0)
                )
            else:
                features["mean_digit_diff"] = 0.0
                features["std_digit_diff"] = 0.0
                features["mean_abs_digit_diff"] = 0.0
                features["positive_diff_ratio"] = 0.0
                features["zero_diff_ratio"] = 0.0
                features["negative_diff_ratio"] = 0.0
                features["sign_changes_count"] = 0.0
        else:
            features.update(
                {
                    "digit_mean": 0.0,
                    "digit_std": 0.0,
                    "digit_min": 0.0,
                    "digit_max": 0.0,
                    "digit_sum": 0.0,
                    "unique_digit_count": 0.0,
                    "unique_digit_ratio": 0.0,
                    "mean_digit_diff": 0.0,
                    "std_digit_diff": 0.0,
                    "mean_abs_digit_diff": 0.0,
                    "positive_diff_ratio": 0.0,
                    "zero_diff_ratio": 0.0,
                    "negative_diff_ratio": 0.0,
                    "sign_changes_count": 0.0,
                }
            )

        if self.config.features.include_digit_counts:
            digit_counter = Counter(number)
            for symbol in self.config.features.digit_symbols:
                count = float(digit_counter.get(symbol, 0))
                features[f"digit_count_{symbol}"] = count
                if self.config.features.include_digit_shares:
                    features[f"digit_share_{symbol}"] = count / length if length else 0.0

        if self.config.features.include_pattern_flags:
            is_increasing = number == "".join(sorted(number))
            is_decreasing = number == "".join(sorted(number, reverse=True))
            features.update(
                {
                    "is_monotonic_increasing": float(is_increasing),
                    "is_monotonic_decreasing": float(is_decreasing),
                    "has_triple_repeat": float(_max_run_length(number) >= 3),
                }
            )

        if self.config.features.include_ngram_counts:
            for n in self.config.features.ngram_sizes:
                features.update(_ngram_statistics(number, n))
            if length > 1:
                pairs = [number[idx : idx + 2] for idx in range(length - 1)]
                pair_counter = Counter(pairs)
                features["unique_pair_count"] = float(len(pair_counter))
                features["max_pair_frequency"] = float(max(pair_counter.values()))
                features["pair_entropy"] = float(
                    -sum((count / (length - 1)) * math.log(count / (length - 1) + 1e-12, 2)
                         for count in pair_counter.values())
                )
            else:
                features["unique_pair_count"] = 0.0
                features["max_pair_frequency"] = 0.0
                features["pair_entropy"] = 0.0

        if length:
            premium_count = float(
                _count_patterns(number, self.config.features.premium_pairs)
            )
            penalty_count = float(
                _count_patterns(number, self.config.features.penalty_pairs)
            )
            features["premium_pair_count"] = premium_count
            features["premium_pair_density"] = premium_count / (length - 1) if length > 1 else 0.0
            features["penalty_pair_count"] = penalty_count
            features["penalty_pair_density"] = penalty_count / (length - 1) if length > 1 else 0.0

            lucky_count = float(
                sum(number.count(symbol) for symbol in self.config.features.lucky_digits)
            )
            unlucky_count = float(
                sum(number.count(symbol) for symbol in self.config.features.unlucky_digits)
            )
            features["lucky_digit_count"] = lucky_count
            features["lucky_digit_ratio"] = lucky_count / length if length else 0.0
            features["unlucky_digit_count"] = unlucky_count
            features["unlucky_digit_ratio"] = unlucky_count / length if length else 0.0

        if self.config.features.include_position_scores and digits:
            for group_name, positions in self.config.features.positional_groups.items():
                group_values = [
                    digits[pos] for pos in positions if 0 <= pos < length
                ]
                if not group_values:
                    group_values = [0]
                group_array = np.array(group_values, dtype=float)
                features[f"{group_name}_sum"] = float(group_array.sum())
                for stat in self.config.features.aggregation_stats:
                    if stat == "mean":
                        features[f"{group_name}_mean"] = float(group_array.mean())
                    elif stat == "std":
                        features[f"{group_name}_std"] = float(group_array.std(ddof=0))
                    elif stat == "min":
                        features[f"{group_name}_min"] = float(group_array.min())
                    elif stat == "max":
                        features[f"{group_name}_max"] = float(group_array.max())

            midpoint = length // 2
            first_half = digits[:midpoint] if midpoint else digits
            second_half = digits[midpoint:] if midpoint else digits
            if first_half:
                features["first_half_sum"] = float(sum(first_half))
                features["first_half_mean"] = float(np.mean(first_half))
            else:
                features["first_half_sum"] = 0.0
                features["first_half_mean"] = 0.0
            if second_half:
                features["second_half_sum"] = float(sum(second_half))
                features["second_half_mean"] = float(np.mean(second_half))
            else:
                features["second_half_sum"] = 0.0
                features["second_half_mean"] = 0.0
            features["half_sum_difference"] = features["first_half_sum"] - features["second_half_sum"]

        if self.config.features.rolling_window_sizes and digits:
            digit_array = np.array(digits, dtype=float)
            for window in self.config.features.rolling_window_sizes:
                if window <= 0 or length < window:
                    features[f"rolling_sum_w{window}"] = 0.0
                    features[f"rolling_mean_w{window}"] = 0.0
                    features[f"rolling_std_w{window}"] = 0.0
                    continue
                rolling_sums = np.convolve(digit_array, np.ones(window), "valid")
                features[f"rolling_sum_w{window}"] = float(rolling_sums.max())
                features[f"rolling_mean_w{window}"] = float(rolling_sums.mean())
                features[f"rolling_std_w{window}"] = float(rolling_sums.std(ddof=0))

        if length >= 3:
            features["prefix3_value"] = float(int(number[:3]))
            features["suffix3_value"] = float(int(number[-3:]))
        else:
            features["prefix3_value"] = 0.0
            features["suffix3_value"] = 0.0

        if length >= 4:
            features["prefix4_value"] = float(int(number[:4]))
            features["suffix4_value"] = float(int(number[-4:]))
        else:
            features["prefix4_value"] = 0.0
            features["suffix4_value"] = 0.0

        if length:
            reversed_number = number[::-1]
            features["is_suffix_same_as_prefix"] = float(number[:3] == number[-3:])
            features["mirror_similarity"] = float(
                sum(1 for a, b in zip(number, reversed_number) if a == b) / length
            )

        return features
