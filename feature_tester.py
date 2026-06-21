from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators import compute_vpin_lsf, compute_vpin_spread
from backtester import (
    DEFAULT_INSTRUMENT,
    default_data_path,
    DEFAULT_RESAMPLE,
    default_output_path,
    filter_chart_data,
    load_data,
    open_chart,
    prepare_bars,
    resample_bars,
)


DEFAULT_OUTPUT_PATH = default_output_path(DEFAULT_INSTRUMENT, "feature_report")
DEFAULT_CSV_DIR = Path("output") / "features"
DEFAULT_DATE = "all"
DEFAULT_SESSION = "regular"
DEFAULT_HORIZONS = ("5m", "15m", "30m", "1h", "1d", "7d", "14d", "28d", "60d", "365d")
DEFAULT_VPIN_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
DEFAULT_FEATURE_WINDOWS = (5, 15, 30, 60, 390)
DEFAULT_FEATURE_SET = "vpin"
DEFAULT_QUANTILES = 5
DEFAULT_TOP_N = 20
DEFAULT_SAMPLE_ROWS = 0
EXPLORER_MAX_ROWS = 5_000
TABLE_TEXT_LEFT_COLUMNS = frozenset({"rank", "feature", "target", "target_family"})
REGULAR_SESSION_START = "09:30"
REGULAR_SESSION_END = "16:00"
MIN_OBSERVATIONS = 30
VPIN_SIGNAL_FEATURES = {
    "buyside_cdf",
    "bullish_div",
    "buy_pct",
    "long_signal",
    "vpin",
    "vpin_cdf",
    "vpin_signed",
    "vpin_spread",
    "vpin_spread_raw",
    "vpin_spread_signed",
    "vpin_spread_vpin",
    "spread_above_top",
    "spread_below_bottom",
    "spread_cross_top",
    "spread_cross_bottom",
}
VPIN_INTERNAL_FEATURES = {
    "div_vpin_y",
    "dp",
    "price_pivot",
    "p50",
    "p75",
    "p90",
    "sigma_dp",
    "sum_buy",
    "sum_sell",
    "sum_vol",
    "v_buy",
    "v_sell",
    "z_score",
}


@dataclass(frozen=True)
class HorizonSpec:
    minutes: int
    bars: int
    label: str


@dataclass(frozen=True)
class TimeframeSpec:
    label: str
    minutes: int


class ProgressReporter:
    def __init__(self, *, enabled: bool = True, width: int = 28) -> None:
        self.enabled = enabled
        self.width = width
        self._last_percent_by_label: dict[str, int] = {}

    def step(self, message: str) -> None:
        if self.enabled:
            print(message, flush=True)

    def update(self, label: str, current: int, total: int) -> None:
        if not self.enabled or total <= 0:
            return

        current = min(current, total)
        percent = int(current / total * 100)
        if self._last_percent_by_label.get(label) == percent and current != total:
            return

        self._last_percent_by_label[label] = percent
        filled = int(self.width * current / total)
        bar = "#" * filled + "-" * (self.width - filled)
        print(f"\r{label}: [{bar}] {percent:3d}% ({current:,}/{total:,})", end="", flush=True)
        if current == total:
            print(flush=True)


def filter_session(bars: pd.DataFrame, session: str) -> pd.DataFrame:
    normalized = session.lower()
    if normalized == "all":
        return bars
    if normalized != "regular":
        raise ValueError("session must be either 'regular' or 'all'.")

    return bars.between_time(REGULAR_SESSION_START, REGULAR_SESSION_END, inclusive="left")


def parse_horizons(value: str) -> list[HorizonSpec]:
    horizons: list[HorizonSpec] = []
    seen_labels: set[str] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        horizon = parse_horizon(part)
        if horizon.label in seen_labels:
            continue
        seen_labels.add(horizon.label)
        horizons.append(horizon)

    if not horizons:
        raise ValueError("at least one horizon is required.")
    return horizons


def parse_horizon(value: str) -> HorizonSpec:
    normalized = value.strip().lower().replace(" ", "")
    aliases = {
        "5": "5m",
        "15": "15m",
        "30": "30m",
        "60": "1h",
        "60m": "1h",
        "390": "1d",
        "390m": "1d",
        "1day": "1d",
        "7day": "7d",
        "14day": "14d",
        "28day": "28d",
        "60day": "60d",
        "365day": "365d",
    }
    label = aliases.get(normalized, normalized)
    if label.endswith("m"):
        minutes = int(label[:-1])
    elif label.endswith("h"):
        minutes = int(label[:-1]) * 60
    elif label.endswith("d"):
        minutes = int(label[:-1]) * 390
    else:
        raise ValueError(f"Unsupported horizon: {value}")
    if minutes < 1:
        raise ValueError("horizons must be positive.")
    return HorizonSpec(minutes=minutes, bars=minutes, label=label)


def parse_vpin_timeframes(value: str) -> list[TimeframeSpec]:
    timeframes: list[TimeframeSpec] = []
    seen_labels: set[str] = set()
    for raw_part in value.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue

        spec = parse_timeframe(part)
        if spec.label in seen_labels:
            continue

        seen_labels.add(spec.label)
        timeframes.append(spec)

    if not timeframes:
        raise ValueError("at least one VPIN timeframe is required.")
    return timeframes


def parse_timeframe(value: str) -> TimeframeSpec:
    normalized = value.strip().lower().replace(" ", "")
    aliases = {
        "1": "1m",
        "1min": "1m",
        "1minute": "1m",
        "5": "5m",
        "5min": "5m",
        "15": "15m",
        "15min": "15m",
        "30": "30m",
        "30min": "30m",
        "60": "1h",
        "60m": "1h",
        "60min": "1h",
        "1hour": "1h",
        "240": "4h",
        "240m": "4h",
        "240min": "4h",
        "4hour": "4h",
        "d": "1d",
        "day": "1d",
        "daily": "1d",
    }
    label = aliases.get(normalized, normalized)
    if label.endswith("m"):
        minutes = int(label[:-1])
    elif label.endswith("h"):
        minutes = int(label[:-1]) * 60
    elif label == "1d":
        minutes = 390
    else:
        raise ValueError(f"Unsupported VPIN timeframe: {value}")

    if minutes < 1:
        raise ValueError("VPIN timeframes must be positive.")
    return TimeframeSpec(label=label, minutes=minutes)


def build_horizon_specs(bars: pd.DataFrame, horizons: list[HorizonSpec | int]) -> list[HorizonSpec]:
    bar_minutes = infer_bar_minutes(bars)
    specs: list[HorizonSpec] = []
    for horizon in horizons:
        if isinstance(horizon, HorizonSpec):
            minutes = horizon.minutes
            label = horizon.label
        else:
            minutes = int(horizon)
            label = "1d" if minutes == 390 else f"{minutes}m"
        bars_for_horizon = max(1, int(math.ceil(minutes / bar_minutes)))
        specs.append(HorizonSpec(minutes=minutes, bars=bars_for_horizon, label=label))
    return specs


def infer_bar_minutes(bars: pd.DataFrame) -> float:
    if len(bars.index) < 2 or not isinstance(bars.index, pd.DatetimeIndex):
        return 1.0

    deltas = bars.index.to_series().diff().dropna()
    positive_minutes = deltas.dt.total_seconds().div(60.0)
    positive_minutes = positive_minutes[positive_minutes > 0]
    if positive_minutes.empty:
        return 1.0
    return float(max(1.0, positive_minutes.median()))


def infer_periods_per_year(bars: pd.DataFrame) -> float | None:
    if bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
        return None

    counts_by_day = bars.groupby(bars.index.date).size()
    if counts_by_day.empty:
        return None
    return float(counts_by_day.median() * 252)


def build_feature_frame(
    bars: pd.DataFrame,
    horizons: list[HorizonSpec | int],
    *,
    feature_windows: tuple[int, ...] = DEFAULT_FEATURE_WINDOWS,
    vpin_timeframes: list[TimeframeSpec] | None = None,
    progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    _validate_ohlcv_bars(bars)
    horizon_specs = build_horizon_specs(bars, horizons)
    vpin_timeframes = vpin_timeframes or parse_vpin_timeframes(",".join(DEFAULT_VPIN_TIMEFRAMES))

    if progress is not None:
        progress.step("Computing OHLCV technical features")
    feature_frame = bars.copy()
    technical = compute_technical_features(bars, feature_windows)

    if progress is not None:
        progress.step("Computing VPIN indicator features")
    indicators = compute_indicator_features(bars, vpin_timeframes=vpin_timeframes, progress=progress)

    if progress is not None:
        progress.step("Computing forward targets")
    targets = compute_forward_targets(bars, horizon_specs)

    return feature_frame.join([technical, indicators, targets])


def compute_technical_features(
    bars: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_FEATURE_WINDOWS,
) -> pd.DataFrame:
    _validate_ohlcv_bars(bars)

    open_ = bars["open"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)

    safe_close = close.where(close > 0)
    safe_open = open_.where(open_ > 0)
    safe_low = low.where(low > 0)

    log_close = np.log(safe_close)
    log_return = log_close.diff()
    simple_return = close.pct_change()
    high_low_range = (high / safe_low) - 1.0
    true_range_pct = pd.concat(
        [
            (high - low).abs() / safe_close,
            (high - close.shift(1)).abs() / safe_close,
            (low - close.shift(1)).abs() / safe_close,
        ],
        axis=1,
    ).max(axis=1)

    result = pd.DataFrame(index=bars.index)
    result["return_1b"] = simple_return
    result["log_return_1b"] = log_return
    result["intrabar_return"] = (close / safe_open) - 1.0
    result["high_low_range_pct"] = high_low_range
    result["true_range_pct"] = true_range_pct
    result["close_location"] = (close - low) / (high - low).replace(0.0, np.nan)
    result["dollar_volume"] = close * volume

    periods_per_year = infer_periods_per_year(bars)
    squared_log_return = log_return.pow(2)
    for window in windows:
        result[f"return_{window}b"] = close.pct_change(window)
        result[f"log_return_{window}b"] = log_close.diff(window)

        realized_variance = squared_log_return.rolling(window, min_periods=window).sum()
        result[f"realized_variance_{window}b"] = realized_variance
        result[f"realized_volatility_{window}b"] = np.sqrt(realized_variance)

        rolling_return_std = log_return.rolling(window, min_periods=window).std(ddof=0)
        if periods_per_year is not None:
            result[f"annualized_volatility_{window}b"] = rolling_return_std * math.sqrt(periods_per_year)
        else:
            result[f"annualized_volatility_{window}b"] = np.nan

        range_mean = high_low_range.rolling(window, min_periods=window).mean()
        result[f"range_mean_{window}b"] = range_mean
        result[f"true_range_mean_{window}b"] = true_range_pct.rolling(window, min_periods=window).mean()

        volume_mean = volume.rolling(window, min_periods=window).mean()
        volume_std = volume.rolling(window, min_periods=window).std(ddof=0)
        result[f"volume_zscore_{window}b"] = (volume - volume_mean) / volume_std.replace(0.0, np.nan)
        result[f"volume_change_{window}b"] = (volume / volume.shift(window).replace(0.0, np.nan)) - 1.0

        result[f"rolling_drawdown_{window}b"] = (close / close.rolling(window, min_periods=window).max()) - 1.0
        result[f"rolling_drawup_{window}b"] = (close / close.rolling(window, min_periods=window).min()) - 1.0

    return result.replace([np.inf, -np.inf], np.nan)


def compute_indicator_features(
    bars: pd.DataFrame,
    *,
    vpin_timeframes: list[TimeframeSpec] | None = None,
    progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    _validate_ohlcv_bars(bars)
    timeframes = vpin_timeframes or parse_vpin_timeframes(",".join(DEFAULT_VPIN_TIMEFRAMES))
    frames: list[pd.DataFrame] = []
    for position, timeframe in enumerate(timeframes, start=1):
        if progress is not None:
            progress.step(f"  VPIN timeframe {timeframe.label} ({position}/{len(timeframes)})")

        timeframe_bars = resample_timeframe_bars(bars, timeframe)
        if timeframe_bars.empty:
            continue

        indicators = compute_single_timeframe_vpin(timeframe_bars)
        indicators = suffix_timeframe_columns(indicators, timeframe.label)
        aligned = align_timeframe_features(indicators, bars.index)
        frames.append(aligned)

    if not frames:
        return pd.DataFrame(index=bars.index)
    return pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)


def compute_single_timeframe_vpin(bars: pd.DataFrame) -> pd.DataFrame:
    spread = compute_vpin_spread(bars)
    lsf = compute_vpin_lsf(bars)
    indicators = spread.join(lsf, rsuffix="_lsf")
    for column in indicators.select_dtypes(include=["bool"]).columns:
        indicators[column] = indicators[column].astype(int)
    return indicators.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)


def suffix_timeframe_columns(indicators: pd.DataFrame, label: str) -> pd.DataFrame:
    return indicators.rename(columns={column: f"{column}_{label}" for column in indicators.columns})


def align_timeframe_features(indicators: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    combined_index = target_index.union(indicators.index).sort_values()
    aligned = indicators.reindex(combined_index).ffill().reindex(target_index)
    aligned.index = target_index
    return aligned


def resample_timeframe_bars(bars: pd.DataFrame, timeframe: TimeframeSpec) -> pd.DataFrame:
    if timeframe.label == "1m" or timeframe.minutes <= infer_bar_minutes(bars):
        return bars
    if timeframe.label == "1d":
        return resample_daily_session_bars(bars)

    return resample_intraday_session_bars(bars, timeframe.minutes)


def resample_intraday_session_bars(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    grouped_frames = []
    rule = f"{minutes}min"
    session_hour, session_minute = REGULAR_SESSION_START.split(":")
    session_offset = f"{int(session_hour)}h{int(session_minute)}min"
    for _, day_bars in bars.groupby(bars.index.date, sort=True):
        resampled = day_bars.resample(
            rule,
            origin="start_day",
            offset=session_offset,
            label="right",
            closed="left",
        ).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        grouped_frames.append(resampled)

    if not grouped_frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=bars.index[:0])
    return pd.concat(grouped_frames).sort_index()


def resample_daily_session_bars(bars: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    index_values = []
    for _, day_bars in bars.groupby(bars.index.date, sort=True):
        if day_bars.empty:
            continue
        index_values.append(day_bars.index[-1])
        records.append(
            {
                "open": float(day_bars["open"].iloc[0]),
                "high": float(day_bars["high"].max()),
                "low": float(day_bars["low"].min()),
                "close": float(day_bars["close"].iloc[-1]),
                "volume": float(day_bars["volume"].sum()),
            }
        )

    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=bars.index[:0])
    return pd.DataFrame(records, index=pd.DatetimeIndex(index_values))


def compute_forward_targets(bars: pd.DataFrame, horizons: list[HorizonSpec]) -> pd.DataFrame:
    _validate_ohlcv_bars(bars)

    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    log_return = np.log(close.where(close > 0)).diff()
    squared_log_return = log_return.pow(2)

    targets = pd.DataFrame(index=bars.index)
    for horizon in horizons:
        h = horizon.bars
        label = horizon.label
        future_close = close.shift(-h)
        future_high = _future_rolling(high, h, "max")
        future_low = _future_rolling(low, h, "min")
        future_variance = _future_rolling(squared_log_return, h, "sum")

        targets[f"fwd_return_{label}"] = (future_close / close) - 1.0
        targets[f"fwd_realized_variance_{label}"] = future_variance
        targets[f"fwd_realized_volatility_{label}"] = np.sqrt(future_variance)
        targets[f"fwd_max_favorable_return_{label}"] = (future_high / close) - 1.0
        targets[f"fwd_max_adverse_return_{label}"] = (future_low / close) - 1.0

    return targets.replace([np.inf, -np.inf], np.nan)


def _future_rolling(series: pd.Series, horizon: int, aggregation: str) -> pd.Series:
    shifted = series.shift(-1)
    rolling = shifted.rolling(horizon, min_periods=horizon)
    if aggregation == "max":
        result = rolling.max()
    elif aggregation == "min":
        result = rolling.min()
    elif aggregation == "sum":
        result = rolling.sum()
    else:
        raise ValueError(f"Unsupported future aggregation: {aggregation}")
    return result.shift(-(horizon - 1))


def select_target_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("fwd_")]


def select_feature_columns(frame: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    target_columns = set(select_target_columns(frame))
    excluded = {"open", "high", "low", "close", "rtype", "publisher_id", "instrument_id"} | target_columns
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    base_columns = [column for column in numeric_columns if column not in excluded]

    normalized = feature_set.lower()
    if normalized == "all":
        return base_columns
    if normalized == "vpin":
        return [column for column in base_columns if is_vpin_feature(column)]
    if normalized == "ohlcv":
        return [column for column in base_columns if not is_vpin_related_feature(column)]
    raise ValueError("feature_set must be one of: vpin, all, ohlcv.")


def is_vpin_feature(column: str) -> bool:
    return base_vpin_feature_name(column) in VPIN_SIGNAL_FEATURES


def is_vpin_related_feature(column: str) -> bool:
    base_name = base_vpin_feature_name(column)
    return is_vpin_feature(column) or base_name in VPIN_INTERNAL_FEATURES or "vpin" in column


def base_vpin_feature_name(column: str) -> str:
    if "_" not in column:
        return column

    base_name, suffix = column.rsplit("_", 1)
    if is_timeframe_label(suffix):
        return base_name
    return column


def is_timeframe_label(value: str) -> bool:
    if len(value) < 2:
        return False
    unit = value[-1]
    amount = value[:-1]
    return unit in {"m", "h", "d"} and amount.isdigit() and int(amount) > 0


def finite_float_array(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric[~np.isfinite(numeric)] = np.nan
    return numeric


def rank_float_array(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    ranks[~np.isfinite(ranks)] = np.nan
    return ranks


def finite_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return 0.0
    std = float(finite.std(ddof=0))
    return 0.0 if pd.isna(std) else std


def count_finite_pairs(left: np.ndarray, right: np.ndarray) -> int:
    return int((np.isfinite(left) & np.isfinite(right)).sum())


def pearson_corr_arrays(left: np.ndarray, right: np.ndarray, min_observations: int) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    observations = int(mask.sum())
    if observations < min_observations:
        return np.nan

    left_values = left[mask]
    right_values = right[mask]
    left_std = float(left_values.std(ddof=0))
    right_std = float(right_values.std(ddof=0))
    if left_std == 0.0 or right_std == 0.0:
        return np.nan

    left_centered = left_values - float(left_values.mean())
    right_centered = right_values - float(right_values.mean())
    return float(np.dot(left_centered, right_centered) / (observations * left_std * right_std))


def compute_correlations(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    *,
    min_observations: int = MIN_OBSERVATIONS,
    progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    target_values = {target: finite_float_array(frame[target]) for target in target_columns}
    target_ranks = {target: rank_float_array(values) for target, values in target_values.items()}
    total_pairs = len(feature_columns) * len(target_columns)
    completed_pairs = 0

    for feature in feature_columns:
        feature_values = finite_float_array(frame[feature])
        feature_rank = rank_float_array(feature_values)
        if finite_std(feature_values) == 0.0:
            completed_pairs += len(target_columns)
            if progress is not None:
                progress.update("Correlations", completed_pairs, total_pairs)
            continue

        for target in target_columns:
            target_values_for_column = target_values[target]
            observations = count_finite_pairs(feature_values, target_values_for_column)
            if observations < min_observations:
                completed_pairs += 1
                if progress is not None:
                    progress.update("Correlations", completed_pairs, total_pairs)
                continue

            pearson = pearson_corr_arrays(feature_values, target_values_for_column, min_observations)
            spearman = pearson_corr_arrays(feature_rank, target_ranks[target], min_observations)
            if pd.isna(pearson) and pd.isna(spearman):
                completed_pairs += 1
                if progress is not None:
                    progress.update("Correlations", completed_pairs, total_pairs)
                continue

            records.append(
                {
                    "feature": feature,
                    "target": target,
                    "target_family": classify_target(target),
                    "observations": observations,
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_pearson": abs(pearson) if pd.notna(pearson) else np.nan,
                    "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
                }
            )
            completed_pairs += 1
            if progress is not None:
                progress.update("Correlations", completed_pairs, total_pairs)

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    return result.sort_values(["abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True)


def compute_quantile_buckets(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    *,
    quantiles: int = DEFAULT_QUANTILES,
    min_observations: int = MIN_OBSERVATIONS,
    progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2.")

    target_values = {target: finite_float_array(frame[target]) for target in target_columns}
    total_pairs = len(feature_columns) * len(target_columns)
    completed_pairs = 0

    for feature in feature_columns:
        feature_values = finite_float_array(frame[feature])
        if finite_std(feature_values) == 0.0:
            completed_pairs += len(target_columns)
            if progress is not None:
                progress.update("Quantile buckets", completed_pairs, total_pairs)
            continue

        buckets = make_quantile_buckets(pd.Series(feature_values, index=frame.index), quantiles).to_numpy(dtype=float)
        finite_buckets = buckets[np.isfinite(buckets)]
        if len(np.unique(finite_buckets)) < 2:
            completed_pairs += len(target_columns)
            if progress is not None:
                progress.update("Quantile buckets", completed_pairs, total_pairs)
            continue

        for target in target_columns:
            target_array = target_values[target]
            mask = np.isfinite(buckets) & np.isfinite(target_array)
            if int(mask.sum()) < min_observations:
                completed_pairs += 1
                if progress is not None:
                    progress.update("Quantile buckets", completed_pairs, total_pairs)
                continue

            bucket_values = buckets[mask].astype(int)
            outcome_values = target_array[mask]
            bucket_numbers: list[int] = []
            bucket_means: list[float] = []
            bucket_rows: list[dict[str, object]] = []
            for bucket in sorted(np.unique(bucket_values)):
                values_for_bucket = outcome_values[bucket_values == bucket]
                if len(values_for_bucket) == 0:
                    continue

                bucket_numbers.append(int(bucket))
                bucket_mean = float(values_for_bucket.mean())
                bucket_means.append(bucket_mean)
                bucket_rows.append(
                    {
                        "bucket": int(bucket),
                        "bucket_count": int(len(values_for_bucket)),
                        "target_mean": bucket_mean,
                        "target_median": float(np.median(values_for_bucket)),
                        "target_std": float(values_for_bucket.std(ddof=1)) if len(values_for_bucket) > 1 else np.nan,
                        "hit_rate": float((values_for_bucket > 0.0).mean()),
                    }
                )

            monotonicity = spearman_for_small_vectors(bucket_numbers, bucket_means)
            for row in bucket_rows:
                records.append(
                    {
                        "feature": feature,
                        "target": target,
                        "target_family": classify_target(target),
                        "bucket": row["bucket"],
                        "bucket_count": row["bucket_count"],
                        "target_mean": row["target_mean"],
                        "target_median": row["target_median"],
                        "target_std": row["target_std"],
                        "hit_rate": row["hit_rate"],
                        "bucket_target_mean_spearman": float(monotonicity) if pd.notna(monotonicity) else np.nan,
                    }
                )
            completed_pairs += 1
            if progress is not None:
                progress.update("Quantile buckets", completed_pairs, total_pairs)

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    return result.sort_values(["feature", "target", "bucket"]).reset_index(drop=True)


def make_quantile_buckets(values: pd.Series, quantiles: int) -> pd.Series:
    numeric = finite_float_array(values)
    result = np.full(len(numeric), np.nan)
    valid_positions = np.flatnonzero(np.isfinite(numeric))
    if len(valid_positions) < quantiles:
        return pd.Series(result, index=values.index)

    valid_values = numeric[valid_positions]
    if len(np.unique(valid_values)) < 2:
        return pd.Series(result, index=values.index)

    bucket_count = min(quantiles, len(valid_positions))
    order = np.argsort(valid_values, kind="mergesort")
    ordered_positions = valid_positions[order]
    bucket_numbers = np.floor(np.arange(len(ordered_positions)) * bucket_count / len(ordered_positions)) + 1.0
    result[ordered_positions] = bucket_numbers
    return pd.Series(result, index=values.index)


def spearman_for_small_vectors(left: list[int], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return np.nan

    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 2:
        return np.nan
    return float(frame["left"].corr(frame["right"], method="spearman"))


def build_top_features(correlations: pd.DataFrame, quantile_buckets: pd.DataFrame) -> pd.DataFrame:
    if correlations.empty:
        return correlations.copy()

    top = correlations.copy()
    if not quantile_buckets.empty:
        sorted_buckets = quantile_buckets.sort_values(["feature", "target", "bucket"])
        bucket_summary = (
            sorted_buckets.groupby(["feature", "target"], as_index=False)
            .agg(
                bucket_count=("bucket", "count"),
                low_bucket_mean=("target_mean", "first"),
                high_bucket_mean=("target_mean", "last"),
                bucket_target_mean_spearman=("bucket_target_mean_spearman", "first"),
            )
        )
        bucket_summary["high_minus_low_bucket_mean"] = (
            bucket_summary["high_bucket_mean"] - bucket_summary["low_bucket_mean"]
        )
        top = top.merge(bucket_summary, on=["feature", "target"], how="left")

    top = top.sort_values(["abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    return top


def classify_target(target: str) -> str:
    if "realized_variance" in target or "realized_volatility" in target:
        return "Variance/Volatility"
    if "max_favorable" in target or "max_adverse" in target:
        return "Path Return"
    if "return" in target:
        return "Forward Return"
    return "Other"


def parse_feature_timeframe_label(feature: str) -> str | None:
    if "_" not in feature:
        return None

    suffix = feature.rsplit("_", 1)[-1]
    if is_timeframe_label(suffix):
        return suffix
    return None


def native_chart_timeframe(bar_minutes: float) -> str:
    minutes = max(1, int(round(bar_minutes)))
    if minutes <= 1:
        return "1m"
    if minutes % 60 == 0:
        hours = minutes // 60
        return "1h" if hours == 1 else f"{hours}h"
    if minutes == 390:
        return "1d"
    return f"{minutes}m"


def resolve_chart_timeframe(feature: str, bar_minutes: float) -> str:
    feature_timeframe = parse_feature_timeframe_label(feature)
    if feature_timeframe is not None:
        return feature_timeframe
    return native_chart_timeframe(bar_minutes)


def resample_explorer_frame(frame: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
    bar_minutes = infer_bar_minutes(frame)
    native_timeframe = native_chart_timeframe(bar_minutes)
    if timeframe_label in {native_timeframe, "1m"} and timeframe_label == native_timeframe:
        return frame

    try:
        timeframe = parse_timeframe(timeframe_label)
    except ValueError:
        return frame

    if timeframe.label == native_timeframe or timeframe.minutes <= bar_minutes:
        return frame

    ohlcv_columns = [column for column in ("open", "high", "low", "close", "volume") if column in frame.columns]
    if len(ohlcv_columns) < 4 or "close" not in ohlcv_columns:
        return frame

    resampled_ohlcv = resample_timeframe_bars(frame[ohlcv_columns], timeframe)
    if resampled_ohlcv.empty:
        return frame

    other_columns = [column for column in frame.columns if column not in ohlcv_columns]
    if not other_columns:
        return resampled_ohlcv

    resampled_index = resampled_ohlcv.index
    aligned = (
        frame[other_columns]
        .reindex(frame.index.union(resampled_index))
        .sort_index()
        .ffill()
        .loc[resampled_index]
    )
    return resampled_ohlcv.join(aligned)


def parse_target_horizon_label(target: str) -> str | None:
    if not target.startswith("fwd_"):
        return None

    suffix = target.rsplit("_", 1)[-1]
    if is_timeframe_label(suffix):
        return suffix
    return None


def horizon_label_to_bars(horizon_label: str, bar_minutes: float) -> int:
    horizon = parse_horizon(horizon_label)
    return max(1, int(math.ceil(horizon.minutes / max(bar_minutes, 1.0))))


def resolve_log_return_series(
    frame: pd.DataFrame,
    horizon_label: str,
    *,
    bar_minutes: float | None = None,
) -> tuple[str, pd.Series]:
    bar_minutes = bar_minutes if bar_minutes is not None else infer_bar_minutes(frame)
    bars = horizon_label_to_bars(horizon_label, bar_minutes)
    series_key = f"log_return_{horizon_label}"
    existing_column = f"log_return_{bars}b"
    if existing_column in frame.columns:
        return series_key, frame[existing_column]

    close = frame["close"].astype(float)
    log_close = np.log(close.where(close > 0))
    return series_key, log_close.diff(bars)


def build_horizon_log_return_columns(
    frame: pd.DataFrame,
    targets: pd.Series | list[str],
) -> tuple[dict[str, str], pd.DataFrame]:
    bar_minutes = infer_bar_minutes(frame)
    extensions = pd.DataFrame(index=frame.index)
    mapping: dict[str, str] = {}

    for target in targets:
        horizon_label = parse_target_horizon_label(str(target))
        if horizon_label is None or horizon_label in mapping:
            continue

        series_key, series = resolve_log_return_series(
            frame,
            horizon_label,
            bar_minutes=bar_minutes,
        )
        extensions[series_key] = series
        mapping[horizon_label] = series_key

    if extensions.empty:
        return mapping, frame
    return mapping, frame.join(extensions)


def collect_top_pairs(top_features: pd.DataFrame, top_n: int | None = None) -> list[dict[str, object]]:
    if top_features.empty:
        return []

    table = top_features if top_n is None or top_n <= 0 else top_features.head(top_n)
    pairs: list[dict[str, object]] = []
    for offset, record in enumerate(table.to_dict(orient="records"), start=1):
        rank = record.get("rank", offset)
        feature = record.get("feature")
        target = record.get("target")
        if feature is None or target is None:
            continue
        pairs.append(
            {
                "rank": int(rank),
                "feature": str(feature),
                "target": str(target),
                "horizon": parse_target_horizon_label(str(target)),
                "target_family": str(record.get("target_family", "")),
                "spearman": _json_safe_value(record.get("spearman")),
                "pearson": _json_safe_value(record.get("pearson")),
            }
        )
    return pairs


def _json_safe_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return None if not np.isfinite(numeric) else numeric
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _series_to_json_list(series: pd.Series) -> list[object]:
    numeric = pd.to_numeric(series, errors="coerce")
    return [_json_safe_value(value) for value in numeric.to_numpy()]


def _explorer_columns(
    working_frame: pd.DataFrame,
    *,
    log_returns_by_horizon: dict[str, str],
    top_features: pd.DataFrame | None,
    pairs: list[dict[str, object]],
) -> set[str]:
    columns: set[str] = set()
    if "close" in working_frame.columns:
        columns.add("close")
    columns.update(log_returns_by_horizon.values())

    if top_features is not None and not top_features.empty:
        for feature_name in top_features["feature"].dropna().unique():
            if feature_name in working_frame.columns:
                columns.add(str(feature_name))
        for target_name in top_features["target"].dropna().unique():
            if target_name in working_frame.columns:
                columns.add(str(target_name))
    else:
        for pair in pairs:
            feature = str(pair["feature"])
            target = str(pair["target"])
            if feature in working_frame.columns:
                columns.add(feature)
            if target in working_frame.columns:
                columns.add(target)
    return columns


def _serialize_explorer_view(
    working_frame: pd.DataFrame,
    columns: set[str],
    *,
    max_rows: int,
    log_returns_by_horizon: dict[str, str],
) -> dict[str, object]:
    if not columns:
        return {"timestamps": [], "series": {}, "log_returns_by_horizon": log_returns_by_horizon}

    available_columns = [column for column in sorted(columns) if column in working_frame.columns]
    sample = downsample_frame(working_frame.loc[:, available_columns], max_rows)
    return {
        "timestamps": [timestamp.isoformat() for timestamp in sample.index],
        "series": {column: _series_to_json_list(sample[column]) for column in available_columns},
        "log_returns_by_horizon": log_returns_by_horizon,
    }


def build_explorer_series_bundle(
    frame: pd.DataFrame,
    pairs: list[dict[str, object]],
    *,
    max_rows: int = EXPLORER_MAX_ROWS,
    top_features: pd.DataFrame | None = None,
) -> dict[str, object]:
    bar_minutes = infer_bar_minutes(frame)
    default_timeframe = native_chart_timeframe(bar_minutes)
    targets = (
        top_features["target"].dropna().unique()
        if top_features is not None and not top_features.empty
        else [pair["target"] for pair in pairs]
    )

    for pair in pairs:
        pair["feature_timeframe"] = resolve_chart_timeframe(str(pair["feature"]), bar_minutes)

    chart_timeframes = {str(pair["feature_timeframe"]) for pair in pairs}
    if top_features is not None and not top_features.empty:
        for feature_name in top_features["feature"].dropna().unique():
            chart_timeframes.add(resolve_chart_timeframe(str(feature_name), bar_minutes))
    if not chart_timeframes:
        chart_timeframes = {default_timeframe}

    views_by_timeframe: dict[str, dict[str, object]] = {}
    for timeframe_label in sorted(chart_timeframes):
        resampled_frame = resample_explorer_frame(frame, timeframe_label)
        log_returns_by_horizon, working_frame = build_horizon_log_return_columns(resampled_frame, targets)
        columns = _explorer_columns(
            working_frame,
            log_returns_by_horizon=log_returns_by_horizon,
            top_features=top_features,
            pairs=pairs,
        )
        views_by_timeframe[timeframe_label] = _serialize_explorer_view(
            working_frame,
            columns,
            max_rows=max_rows,
            log_returns_by_horizon=log_returns_by_horizon,
        )

    return {
        "views_by_timeframe": views_by_timeframe,
        "pairs": pairs,
        "default_timeframe": default_timeframe,
    }


def _format_table_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return html.escape(str(value))


def build_clickable_top_table_html(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body_rows: list[str] = []
    for offset, record in enumerate(table.to_dict(orient="records"), start=1):
        rank = int(record["rank"]) if "rank" in record and pd.notna(record["rank"]) else offset
        cells = []
        for column in columns:
            alignment = "left" if column in TABLE_TEXT_LEFT_COLUMNS else "right"
            cells.append(f'<td style="text-align:{alignment}">{_format_table_cell(record[column])}</td>')
        body_rows.append(f'<tr class="pair-row" data-rank="{rank}">{"".join(cells)}</tr>')

    return (
        "<p class='meta'>Click a row to explore price, horizon-matched log return, signal, and target on the feature's timeframe.</p>"
        "<div id='top-pairs-table-wrapper'>"
        "<table id='top-pairs-table'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def plotly_cdn_script() -> str:
    fragment = go.Figure().to_html(full_html=False, include_plotlyjs="cdn")
    graph_div_start = fragment.find('<div id="')
    if graph_div_start < 0:
        return ""

    header = fragment[:graph_div_start].strip()
    if header.startswith("<div>"):
        header = header[5:].strip()
    return header


def build_pair_explorer_script(instrument: str) -> str:
    instrument_js = json.dumps(instrument)
    return f"""
<script>
(function() {{
  const instrument = {instrument_js};
  const dataElement = document.getElementById("pair-explorer-data");
  if (!dataElement) {{
    return;
  }}

  const bundle = JSON.parse(dataElement.textContent);
  const chartElement = document.getElementById("pair-explorer-chart");

  function whenPlotlyReady(callback) {{
    if (window.Plotly) {{
      callback();
      return;
    }}

    const startedAt = Date.now();
    const timer = window.setInterval(() => {{
      if (window.Plotly) {{
        window.clearInterval(timer);
        callback();
        return;
      }}
      if (Date.now() - startedAt > 10_000) {{
        window.clearInterval(timer);
        console.error("Plotly failed to load for pair explorer.");
      }}
    }}, 50);
  }}

  function pairForRank(rank) {{
    return bundle.pairs.find((pair) => pair.rank === rank);
  }}

  function viewForPair(pair) {{
    const timeframe = pair.feature_timeframe || bundle.default_timeframe || "1m";
    const views = bundle.views_by_timeframe || {{}};
    return views[timeframe] || views[bundle.default_timeframe] || Object.values(views)[0];
  }}

  function renderPair(rank) {{
    const pair = pairForRank(rank);
    const view = pair ? viewForPair(pair) : null;
    if (!pair || !view || !chartElement) {{
      return;
    }}

    const timeframe = pair.feature_timeframe || bundle.default_timeframe || "1m";
    const timestamps = view.timestamps;
    const close = view.series.close || [];
    const horizon = pair.horizon;
    const logReturnKey = horizon ? view.log_returns_by_horizon[horizon] : null;
    const logReturn = logReturnKey ? (view.series[logReturnKey] || []) : [];
    const logReturnLabel = horizon ? `log return (${{horizon}})` : null;
    const feature = view.series[pair.feature] || [];
    const target = view.series[pair.target] || [];
    const spearman = pair.spearman == null ? "" : ` (Spearman ${{Number(pair.spearman).toFixed(4)}})`;
    const rowCount = logReturnKey ? 4 : 3;

    const traces = [
      {{
        x: timestamps,
        y: close,
        type: "scatter",
        mode: "lines",
        name: `${{instrument}} close`,
        line: {{color: "#111827"}},
        xaxis: "x",
        yaxis: "y",
      }},
    ];

    if (logReturnKey) {{
      traces.push({{
        x: timestamps,
        y: logReturn,
        type: "scatter",
        mode: "lines",
        name: logReturnLabel,
        line: {{color: "#7c3aed"}},
        xaxis: "x2",
        yaxis: "y2",
      }});
    }}

    traces.push(
      {{
        x: timestamps,
        y: feature,
        type: "scatter",
        mode: "lines",
        name: pair.feature,
        line: {{color: "#2563eb"}},
        xaxis: logReturnKey ? "x3" : "x2",
        yaxis: logReturnKey ? "y3" : "y2",
      }},
      {{
        x: timestamps,
        y: target,
        type: "scatter",
        mode: "lines",
        name: pair.target,
        line: {{color: "#dc2626"}},
        xaxis: logReturnKey ? "x4" : "x3",
        yaxis: logReturnKey ? "y4" : "y3",
      }}
    );

    const layout = {{
      title: `${{instrument}} · ${{timeframe}} · ${{pair.feature}} → ${{pair.target}}${{spearman}}`,
      template: "plotly_white",
      height: logReturnKey ? 960 : 720,
      hovermode: "x unified",
      showlegend: true,
      grid: {{rows: rowCount, columns: 1, pattern: "independent", roworder: "top to bottom"}},
      yaxis: {{title: "Price"}},
      xaxis: {{title: logReturnKey ? "" : "Time"}},
    }};

    if (logReturnKey) {{
      layout.yaxis2 = {{title: horizon ? `Log Return (${{horizon}})` : "Log Return"}};
      layout.yaxis3 = {{title: "Signal"}};
      layout.yaxis4 = {{title: "Target"}};
      layout.xaxis4 = {{title: "Time"}};
    }} else {{
      layout.yaxis2 = {{title: "Signal"}};
      layout.yaxis3 = {{title: "Target"}};
      layout.xaxis3 = {{title: "Time"}};
    }}

    Plotly.newPlot(chartElement, traces, layout, {{responsive: true}});
  }}

  function bindPairExplorer() {{
    document.querySelectorAll(".pair-row").forEach((row) => {{
      row.addEventListener("click", () => {{
        const rank = Number(row.dataset.rank);
        document.querySelectorAll(".pair-row").forEach((entry) => entry.classList.remove("selected"));
        row.classList.add("selected");
        renderPair(rank);
      }});
    }});

    if (bundle.pairs.length > 0) {{
      const firstRank = bundle.pairs[0].rank;
      const firstRow = document.querySelector(`.pair-row[data-rank="${{firstRank}}"]`);
      if (firstRow) {{
        firstRow.click();
      }}
    }}
  }}

  whenPlotlyReady(bindPairExplorer);
}})();
</script>
"""


def write_csv_outputs(
    *,
    csv_dir: Path,
    correlations: pd.DataFrame,
    quantile_buckets: pd.DataFrame,
    top_features: pd.DataFrame,
    feature_frame: pd.DataFrame,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(csv_dir / "correlations.csv", index=False)
    quantile_buckets.to_csv(csv_dir / "quantile_buckets.csv", index=False)
    top_features.to_csv(csv_dir / "top_features.csv", index=False)

    if sample_rows > 0:
        sample = downsample_frame(feature_frame, sample_rows)
        sample.to_csv(csv_dir / "feature_frame_sample.csv")


def write_html_report(
    *,
    output_path: Path,
    frame: pd.DataFrame,
    correlations: pd.DataFrame,
    quantile_buckets: pd.DataFrame,
    top_features: pd.DataFrame,
    instrument: str = DEFAULT_INSTRUMENT,
    top_n: int = DEFAULT_TOP_N,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figures = [
        build_correlation_heatmap(correlations, top_n),
        build_quantile_chart(quantile_buckets, top_features),
        build_timeseries_chart(frame, top_features, instrument=instrument),
    ]
    table_columns = [
        column
        for column in [
            "rank",
            "feature",
            "target",
            "target_family",
            "observations",
            "spearman",
            "pearson",
            "bucket_target_mean_spearman",
            "high_minus_low_bucket_mean",
        ]
        if column in top_features.columns
    ]
    top_table = top_features.loc[:, table_columns]
    all_pairs = collect_top_pairs(top_features)
    explorer_bundle = build_explorer_series_bundle(frame, all_pairs, top_features=top_features)
    explorer_bundle["instrument"] = instrument

    html_parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{instrument} Indicator Feature Report</title>",
        plotly_cdn_script(),
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;color:#111827;background:#f8fafc;}",
        "h1,h2{margin:0 0 12px 0;} section{margin:0 0 28px 0;}",
        "table{border-collapse:collapse;width:100%;background:white;font-size:13px;}",
        "th,td{border:1px solid #d1d5db;padding:6px 8px;text-align:right;}",
        "th{text-align:left;background:#e5e7eb;}",
        ".meta{color:#4b5563;margin-bottom:20px;}",
        ".pair-row{cursor:pointer;}",
        ".pair-row:hover{background:#f1f5f9;}",
        ".pair-row.selected{background:#dbeafe;}",
        "#top-pairs-table-wrapper{max-height:520px;overflow:auto;border:1px solid #d1d5db;background:white;}",
        "#top-pairs-table thead th{position:sticky;top:0;z-index:1;}",
        "#pair-explorer-chart{min-height:720px;background:white;border:1px solid #d1d5db;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{instrument} Indicator Feature Report</h1>",
        (
            "<p class='meta'>"
            f"Rows: {len(frame):,} | "
            f"Range: {frame.index.min()} to {frame.index.max()} | "
            f"Ranked pairs: {len(top_features):,}"
            "</p>"
        ),
        f"<section><h2>All Feature Relationships ({len(top_table):,})</h2>",
        build_clickable_top_table_html(top_table),
        "</section>",
        "<section id='pair-explorer'>",
        "<h2>Pair Explorer</h2>",
        "<div id='pair-explorer-chart'></div>",
        "</section>",
        (
            "<script id='pair-explorer-data' type='application/json'>"
            f"{json.dumps(explorer_bundle)}"
            "</script>"
        ),
        build_pair_explorer_script(instrument),
    ]

    for figure in figures:
        html_parts.append("<section>")
        html_parts.append(figure.to_html(full_html=False, include_plotlyjs=False))
        html_parts.append("</section>")

    html_parts.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(html_parts), encoding="utf-8")


def build_correlation_heatmap(correlations: pd.DataFrame, top_n: int) -> go.Figure:
    if correlations.empty:
        figure = go.Figure()
        figure.add_annotation(text="No correlations available", showarrow=False)
        return figure

    max_features = max(5, top_n)
    max_targets = max(5, min(top_n, 25))
    feature_order = (
        correlations.groupby("feature")["abs_spearman"].max().sort_values(ascending=False).head(max_features).index
    )
    target_order = (
        correlations.groupby("target")["abs_spearman"].max().sort_values(ascending=False).head(max_targets).index
    )
    selected = correlations[correlations["feature"].isin(feature_order) & correlations["target"].isin(target_order)]
    pivot = selected.pivot_table(index="feature", columns="target", values="spearman")
    pivot = pivot.reindex(index=feature_order, columns=target_order)

    figure = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Spearman"},
        )
    )
    figure.update_layout(
        title="Top Spearman Correlations",
        template="plotly_white",
        height=max(500, 28 * len(pivot.index)),
        margin={"l": 220, "r": 50, "t": 70, "b": 140},
    )
    return figure


def build_quantile_chart(quantile_buckets: pd.DataFrame, top_features: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if quantile_buckets.empty or top_features.empty:
        figure.add_annotation(text="No quantile buckets available", showarrow=False)
        return figure

    selected_pairs = select_quantile_pairs(top_features)
    for feature, target in selected_pairs:
        buckets = quantile_buckets[
            (quantile_buckets["feature"] == feature) & (quantile_buckets["target"] == target)
        ].sort_values("bucket")
        if buckets.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=buckets["bucket"],
                y=buckets["target_mean"],
                mode="lines+markers",
                name=f"{feature} -> {target}",
            )
        )

    if not figure.data:
        figure.add_annotation(text="No selected quantile buckets available", showarrow=False)
    figure.update_layout(
        title="Forward Outcome by Feature Quantile",
        template="plotly_white",
        xaxis_title="Feature quantile bucket",
        yaxis_title="Average forward target",
        hovermode="x unified",
        height=520,
    )
    return figure


def select_quantile_pairs(top_features: pd.DataFrame) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for family in ["Forward Return", "Variance/Volatility", "Path Return"]:
        family_rows = top_features[top_features["target_family"] == family]
        if not family_rows.empty:
            row = family_rows.iloc[0]
            pair = (str(row["feature"]), str(row["target"]))
            if pair not in pairs:
                pairs.append(pair)
    if not pairs and not top_features.empty:
        row = top_features.iloc[0]
        pairs.append((str(row["feature"]), str(row["target"])))
    return pairs


def build_timeseries_chart(
    frame: pd.DataFrame,
    top_features: pd.DataFrame,
    *,
    instrument: str = DEFAULT_INSTRUMENT,
) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    sample = downsample_frame(frame.dropna(subset=["close"]), 5_000)
    if sample.empty:
        figure.add_annotation(text="No price data available", showarrow=False)
        return figure

    figure.add_trace(
        go.Scatter(
            x=sample.index,
            y=sample["close"],
            mode="lines",
            name=f"{instrument} close",
            line={"color": "#111827"},
        ),
        secondary_y=False,
    )

    selected_features = []
    if not top_features.empty and "feature" in top_features.columns:
        for feature in top_features["feature"].drop_duplicates().head(3):
            if feature in sample.columns:
                selected_features.append(feature)

    for feature in selected_features:
        normalized = zscore(sample[feature])
        figure.add_trace(
            go.Scatter(x=sample.index, y=normalized, mode="lines", name=f"{feature} z-score"),
            secondary_y=True,
        )

    figure.update_layout(
        title="Price and Top Feature Z-Scores",
        template="plotly_white",
        hovermode="x unified",
        height=520,
    )
    figure.update_yaxes(title_text="Close", secondary_y=False)
    figure.update_yaxes(title_text="Feature z-score", secondary_y=True)
    return figure


def downsample_frame(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    step = int(math.ceil(len(frame) / max_rows))
    return frame.iloc[::step]


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0.0:
        return numeric * np.nan
    return (numeric - numeric.mean()) / std


def run_feature_analysis(
    bars: pd.DataFrame,
    *,
    horizon_minutes: list[HorizonSpec | int],
    date: str = DEFAULT_DATE,
    start: str | None = None,
    end: str | None = None,
    quantiles: int = DEFAULT_QUANTILES,
    feature_set: str = DEFAULT_FEATURE_SET,
    vpin_timeframes: list[TimeframeSpec] | None = None,
    progress: ProgressReporter | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_frame = build_feature_frame(bars, horizon_minutes, vpin_timeframes=vpin_timeframes, progress=progress)
    if progress is not None:
        progress.step("Filtering analysis window")
    analysis_frame = filter_chart_data(full_frame, date=date, start=start, end=end)
    feature_columns = select_feature_columns(analysis_frame, feature_set=feature_set)
    target_columns = select_target_columns(analysis_frame)
    if not feature_columns:
        raise ValueError(f"No features selected for feature_set={feature_set!r}.")
    correlations = compute_correlations(analysis_frame, feature_columns, target_columns, progress=progress)
    quantile_buckets = compute_quantile_buckets(
        analysis_frame,
        feature_columns,
        target_columns,
        quantiles=quantiles,
        progress=progress,
    )
    if progress is not None:
        progress.step("Ranking feature relationships")
    top_features = build_top_features(correlations, quantile_buckets)
    return analysis_frame, correlations, quantile_buckets, top_features


def _validate_ohlcv_bars(bars: pd.DataFrame) -> None:
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(bars.columns)
    if missing_columns:
        raise ValueError(f"Expected OHLCV columns. Missing: {sorted(missing_columns)}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("Feature tester requires bars indexed by a pandas DatetimeIndex.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test indicator and OHLCV features against forward return and volatility targets.",
    )
    parser.add_argument(
        "--instrument",
        default=DEFAULT_INSTRUMENT,
        help=f"Instrument symbol to load and analyze. Default: {DEFAULT_INSTRUMENT}.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the Databento DBN file. Default: data/<instrument>/ohlcv-1m.dbn.zst",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Path for the output HTML report. Default: output/<instrument>_feature_report.html ({DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help=f"Directory for CSV outputs. Default: {DEFAULT_CSV_DIR}",
    )
    parser.add_argument("--start", help="Optional start date/time filter. Overrides --date when used.")
    parser.add_argument("--end", help="Optional end date/time filter. Overrides --date when used.")
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="Date to analyze: 'latest', 'all', or YYYY-MM-DD. Default: all.",
    )
    parser.add_argument(
        "--resample",
        default=DEFAULT_RESAMPLE,
        help="Pandas resample interval, or 'none' for raw 1-minute bars. Default: none.",
    )
    parser.add_argument(
        "--session",
        choices=["regular", "all"],
        default=DEFAULT_SESSION,
        help="Trading session to analyze. Default: regular.",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(value) for value in DEFAULT_HORIZONS),
        help="Comma-separated forward target horizons. Supports m/h/d labels. Default: 5m,15m,30m,1h,1d,7d,14d,28d,60d,365d.",
    )
    parser.add_argument(
        "--vpin-timeframes",
        default=",".join(DEFAULT_VPIN_TIMEFRAMES),
        help="Comma-separated VPIN indicator timeframes. Default: 1m,5m,15m,30m,1h,4h,1d.",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=DEFAULT_QUANTILES,
        help=f"Quantile bucket count. Default: {DEFAULT_QUANTILES}.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["vpin", "all", "ohlcv"],
        default=DEFAULT_FEATURE_SET,
        help=(
            "Features to test: vpin for VPIN indicator/signal columns, "
            "all for every feature including VPIN internals, ohlcv for non-VPIN features. Default: vpin."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=(
            "Number of top relationships highlighted in summary charts. "
            f"The clickable results table always includes every ranked pair. Default: {DEFAULT_TOP_N}."
        ),
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="Write a downsampled feature_frame_sample.csv with at most this many rows. Default: disabled.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Write outputs without opening the report in the default browser.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable console progress output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data or default_data_path(args.instrument)
    output_path = args.output or default_output_path(args.instrument, "feature_report")
    progress = ProgressReporter(enabled=not args.no_progress)
    try:
        horizon_minutes = parse_horizons(args.horizons)
        vpin_timeframes = parse_vpin_timeframes(args.vpin_timeframes)
        progress.step(f"Loading {args.instrument} data from {data_path}")
        df = load_data(data_path, instrument=args.instrument)
        progress.step("Preparing bars")
        bars = resample_bars(prepare_bars(df), args.resample)
        bars = filter_session(bars, args.session)
        analysis_frame, correlations, quantile_buckets, top_features = run_feature_analysis(
            bars,
            horizon_minutes=horizon_minutes,
            date=args.date,
            start=args.start,
            end=args.end,
            quantiles=args.quantiles,
            feature_set=args.feature_set,
            vpin_timeframes=vpin_timeframes,
            progress=progress,
        )
        progress.step(f"Writing CSVs to {args.csv_dir}")
        write_csv_outputs(
            csv_dir=args.csv_dir,
            correlations=correlations,
            quantile_buckets=quantile_buckets,
            top_features=top_features,
            feature_frame=analysis_frame,
            sample_rows=args.sample_rows,
        )
        progress.step(f"Writing HTML report to {output_path}")
        write_html_report(
            output_path=output_path,
            frame=analysis_frame,
            correlations=correlations,
            quantile_buckets=quantile_buckets,
            top_features=top_features,
            instrument=args.instrument,
            top_n=args.top_n,
        )
        progress.step("Done")
        opened = False if args.no_open else open_chart(output_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df):,} {args.instrument} rows from {data_path}")
    print(
        "Analyzed "
        f"{len(analysis_frame):,} {args.session} rows from {analysis_frame.index.min()} "
        f"to {analysis_frame.index.max()}"
    )
    print(f"Feature set: {args.feature_set}")
    print(f"VPIN timeframes: {', '.join(timeframe.label for timeframe in vpin_timeframes)}")
    print(f"Features tested: {len(select_feature_columns(analysis_frame, feature_set=args.feature_set)):,}")
    print(f"Targets tested: {len(select_target_columns(analysis_frame)):,}")
    print(f"Ranked feature-target pairs: {len(top_features):,}")
    print(f"Wrote report: {output_path}")
    print(f"Wrote CSVs: {args.csv_dir}")
    if not top_features.empty:
        best = top_features.iloc[0]
        print(
            "Top relationship: "
            f"{best['feature']} -> {best['target']} "
            f"Spearman={best['spearman']:.4f} n={int(best['observations']):,}"
        )
    if not args.no_open:
        print("Opened report in default browser" if opened else "Report written, but browser did not open")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
