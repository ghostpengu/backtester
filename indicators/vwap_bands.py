from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HLC3 = "hlc3"
CLOSE = "close"
HL2 = "hl2"
VWAP_LOOKBACK_NONE = "none"
VWAP_LOOKBACK_1W = "1w"
VWAP_LOOKBACK_1Y = "1y"
REGULAR_SESSION_BARS = 390
TRADING_DAYS_PER_WEEK = 5
TRADING_DAYS_PER_YEAR = 252
VWAP_LOOKBACK_WINDOWS = {
    VWAP_LOOKBACK_1W: TRADING_DAYS_PER_WEEK * REGULAR_SESSION_BARS,
    VWAP_LOOKBACK_1Y: TRADING_DAYS_PER_YEAR * REGULAR_SESSION_BARS,
}


@dataclass(frozen=True)
class VWAPBandsConfig:
    sigma_levels: tuple[float, ...] = (2.0, 3.0)
    typical_price: str = HLC3
    lookback: str = VWAP_LOOKBACK_NONE
    signal_sigma: float = 2.0

    @property
    def warmup_bars(self) -> int:
        if self.lookback.lower() != VWAP_LOOKBACK_NONE:
            return _lookback_window_bars(self.lookback)
        return 1


def compute_vwap_bands(
    bars: pd.DataFrame,
    config: VWAPBandsConfig | None = None,
) -> pd.DataFrame:
    config = config or VWAPBandsConfig()
    _validate_bars(bars)
    _validate_config(config)

    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    typical = _typical_price(high, low, close, config.typical_price)

    volume_sum, pv_sum, vp2_sum = _weighted_sums(typical, volume, bars.index, config.lookback)
    vwap = (pv_sum / volume_sum.replace(0.0, np.nan)).where(volume_sum > 0)

    variance = ((vp2_sum / volume_sum.replace(0.0, np.nan)) - vwap.pow(2)).clip(lower=0.0)
    vwap_std = np.sqrt(variance).where(volume_sum > 0)

    safe_std = vwap_std.where(vwap_std > 1e-10)
    safe_vwap = vwap.where(vwap > 0)

    result = pd.DataFrame(index=bars.index)
    result["vwap"] = vwap
    result["vwap_std"] = vwap_std
    result["vwap_zscore"] = ((close - vwap) / safe_std).where(safe_std.notna())
    result["vwap_distance_pct"] = ((close - vwap) / safe_vwap).where(safe_vwap.notna())
    _add_signal_columns(result, config.signal_sigma)

    for level in config.sigma_levels:
        label = _sigma_label(level)
        upper = vwap + level * vwap_std
        lower = vwap - level * vwap_std
        above_upper = close >= upper
        below_lower = close <= lower
        cross_upper = above_upper & (close.shift(1) < upper.shift(1))
        cross_lower = below_lower & (close.shift(1) > lower.shift(1))

        result[f"vwap_upper_{label}"] = upper
        result[f"vwap_lower_{label}"] = lower
        result[f"vwap_above_upper_{label}"] = above_upper.fillna(False)
        result[f"vwap_below_lower_{label}"] = below_lower.fillna(False)
        result[f"vwap_cross_upper_{label}"] = cross_upper.fillna(False)
        result[f"vwap_cross_lower_{label}"] = cross_lower.fillna(False)

    return result


def _weighted_sums(
    typical: pd.Series,
    volume: pd.Series,
    index: pd.DatetimeIndex,
    lookback: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    pv = typical * volume
    vp2 = typical.pow(2) * volume

    if lookback.lower() == VWAP_LOOKBACK_NONE:
        session_key = pd.Series(index.date, index=index)
        return (
            volume.groupby(session_key).cumsum(),
            pv.groupby(session_key).cumsum(),
            vp2.groupby(session_key).cumsum(),
        )

    window = _lookback_window_bars(lookback)
    return (
        volume.rolling(window=window, min_periods=1).sum(),
        pv.rolling(window=window, min_periods=1).sum(),
        vp2.rolling(window=window, min_periods=1).sum(),
    )


def _add_signal_columns(result: pd.DataFrame, signal_sigma: float) -> None:
    zscore = result["vwap_zscore"]
    abs_zscore = zscore.abs()
    signal_strength = (abs_zscore / signal_sigma).where(abs_zscore >= signal_sigma, 0.0).fillna(0.0)
    signal_direction = pd.Series(0.0, index=result.index)
    signal_direction = signal_direction.mask(zscore >= signal_sigma, -1.0)
    signal_direction = signal_direction.mask(zscore <= -signal_sigma, 1.0)

    result["vwap_signal_strength"] = signal_strength
    result["vwap_signal"] = signal_direction * signal_strength


def _typical_price(high: pd.Series, low: pd.Series, close: pd.Series, mode: str) -> pd.Series:
    if mode == HLC3:
        return (high + low + close) / 3.0
    if mode == CLOSE:
        return close
    if mode == HL2:
        return (high + low) / 2.0
    raise ValueError(f"Unsupported typical_price mode: {mode}")


def _sigma_label(level: float) -> str:
    if float(level).is_integer():
        return f"{int(level)}sigma"
    normalized = str(level).replace(".", "p")
    return f"{normalized}sigma"


def _validate_bars(bars: pd.DataFrame) -> None:
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(bars.columns)
    if missing_columns:
        available_columns = ", ".join(map(str, bars.columns))
        raise ValueError(
            f"VWAP bands require columns {sorted(required_columns)}. "
            f"Missing: {sorted(missing_columns)}. Available columns: {available_columns}"
        )

    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("VWAP bands require bars indexed by a pandas DatetimeIndex.")


def _validate_config(config: VWAPBandsConfig) -> None:
    if config.typical_price not in {HLC3, CLOSE, HL2}:
        raise ValueError(f"Unsupported typical_price mode: {config.typical_price}")
    if config.lookback.lower() not in {VWAP_LOOKBACK_NONE, *VWAP_LOOKBACK_WINDOWS.keys()}:
        raise ValueError("lookback must be one of: none, 1w, 1y.")
    if not config.sigma_levels:
        raise ValueError("sigma_levels must contain at least one level.")
    if any(level <= 0 for level in config.sigma_levels):
        raise ValueError("sigma_levels must be positive.")
    if config.signal_sigma <= 0:
        raise ValueError("signal_sigma must be positive.")


def _lookback_window_bars(lookback: str) -> int:
    try:
        return VWAP_LOOKBACK_WINDOWS[lookback.lower()]
    except KeyError as exc:
        raise ValueError("lookback must be one of: none, 1w, 1y.") from exc
