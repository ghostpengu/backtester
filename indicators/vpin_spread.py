from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SIGNED_MINUS_VPIN = "Signed - VPIN"
VPIN_MINUS_SIGNED = "VPIN - Signed"
ABS_DISTANCE = "Abs Distance"


@dataclass(frozen=True)
class VPINSpreadConfig:
    n_bars: int = 50
    sigma_len: int = 30
    spread_mode: str = SIGNED_MINUS_VPIN
    smooth_len: int = 1
    top_limit: float = -0.20
    bottom_limit: float = -0.70
    max_bars_back: int = 1000

    @property
    def warmup_bars(self) -> int:
        return max(self.max_bars_back, self.n_bars, self.sigma_len, self.smooth_len)


def compute_vpin_spread(bars: pd.DataFrame, config: VPINSpreadConfig | None = None) -> pd.DataFrame:
    config = config or VPINSpreadConfig()
    _validate_bars(bars)
    _validate_config(config)

    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)

    dp = close.diff()
    sigma_dp = dp.rolling(config.sigma_len, min_periods=config.sigma_len).std(ddof=0)
    z_score = (dp / sigma_dp).where(sigma_dp > 1e-10, 0.0).fillna(0.0)

    v_buy = volume * _normal_cdf(z_score)
    v_sell = volume - v_buy
    imbalance = (v_buy - v_sell).abs()

    sum_buy = v_buy.rolling(config.n_bars, min_periods=config.n_bars).sum()
    sum_sell = v_sell.rolling(config.n_bars, min_periods=config.n_bars).sum()
    sum_vol = volume.rolling(config.n_bars, min_periods=config.n_bars).sum()
    sum_imb = imbalance.rolling(config.n_bars, min_periods=config.n_bars).sum()

    vpin = (sum_imb / sum_vol).where(sum_vol > 0)
    vpin_signed = ((sum_buy - sum_sell) / sum_vol).where(sum_vol > 0)

    if config.spread_mode == SIGNED_MINUS_VPIN:
        spread_raw = vpin_signed - vpin
    elif config.spread_mode == VPIN_MINUS_SIGNED:
        spread_raw = vpin - vpin_signed
    else:
        spread_raw = (vpin_signed - vpin).abs()

    spread = spread_raw.ewm(span=config.smooth_len, adjust=False, min_periods=1).mean()
    if config.smooth_len == 1:
        spread = spread_raw

    above_top = spread >= config.top_limit
    below_bottom = spread <= config.bottom_limit
    cross_top = above_top & (spread.shift(1) < config.top_limit)
    cross_bottom = below_bottom & (spread.shift(1) > config.bottom_limit)

    result = pd.DataFrame(index=bars.index)
    result["vpin_spread_raw"] = spread_raw
    result["vpin_spread"] = spread
    result["vpin_spread_vpin"] = vpin
    result["vpin_spread_signed"] = vpin_signed
    result["spread_above_top"] = above_top.fillna(False)
    result["spread_below_bottom"] = below_bottom.fillna(False)
    result["spread_cross_top"] = cross_top.fillna(False)
    result["spread_cross_bottom"] = cross_bottom.fillna(False)
    return result


def _normal_cdf(z_score: pd.Series) -> pd.Series:
    z = z_score.fillna(0.0).to_numpy(dtype=float)
    sign = np.where(z >= 0.0, 1.0, -1.0)
    x = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * x)
    p = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    values = 0.5 * (1.0 + sign * (1.0 - p * np.exp(-x * x)))
    return pd.Series(values, index=z_score.index)


def _validate_bars(bars: pd.DataFrame) -> None:
    required_columns = {"close", "volume"}
    missing_columns = required_columns.difference(bars.columns)
    if missing_columns:
        available_columns = ", ".join(map(str, bars.columns))
        raise ValueError(
            f"VPIN spread requires columns {sorted(required_columns)}. "
            f"Missing: {sorted(missing_columns)}. Available columns: {available_columns}"
        )

    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("VPIN spread requires bars indexed by a pandas DatetimeIndex.")


def _validate_config(config: VPINSpreadConfig) -> None:
    if config.spread_mode not in {SIGNED_MINUS_VPIN, VPIN_MINUS_SIGNED, ABS_DISTANCE}:
        raise ValueError(f"Unsupported spread_mode: {config.spread_mode}")
    if config.smooth_len < 1:
        raise ValueError("smooth_len must be at least 1.")
