from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VPINLSFConfig:
    n_bars: int = 50
    sigma_len: int = 30
    cdf_len: int = 250
    pivot_lookback: int = 10
    min_price_drop_pct: float = 0.10
    elevated_threshold: float = 0.75
    confirm_threshold: float = 0.65
    sign_trigger: float = -0.2
    max_bars_back: int = 1000

    @property
    def warmup_bars(self) -> int:
        return max(
            self.max_bars_back,
            self.n_bars,
            self.sigma_len,
            self.cdf_len,
            self.pivot_lookback * 2,
        )


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


def _rolling_percent_rank(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).rank(method="max", pct=True)


def _validate_bars(bars: pd.DataFrame) -> None:
    required_columns = {"low", "close", "volume"}
    missing_columns = required_columns.difference(bars.columns)
    if missing_columns:
        available_columns = ", ".join(map(str, bars.columns))
        raise ValueError(
            f"VPIN-LSF requires columns {sorted(required_columns)}. "
            f"Missing: {sorted(missing_columns)}. Available columns: {available_columns}"
        )

    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("VPIN-LSF requires bars indexed by a pandas DatetimeIndex.")


def _detect_confirmed_pivot_lows(low: pd.Series, pivot_lookback: int) -> pd.Series:
    price_pivot = pd.Series(np.nan, index=low.index)
    lows = low.to_numpy()

    for center in range(pivot_lookback, len(lows) - pivot_lookback):
        window = lows[center - pivot_lookback : center + pivot_lookback + 1]
        if np.isfinite(lows[center]) and lows[center] == np.nanmin(window):
            price_pivot.iloc[center + pivot_lookback] = lows[center]

    return price_pivot


def _detect_bullish_divergence(
    price_pivot: pd.Series,
    vpin_signed: pd.Series,
    config: VPINLSFConfig,
) -> pd.Series:
    bullish_div = pd.Series(False, index=price_pivot.index)
    previous_price = np.nan
    last_price = np.nan
    previous_signed = np.nan
    last_signed = np.nan
    signed_values = vpin_signed.to_numpy()

    for position, pivot_value in enumerate(price_pivot.to_numpy()):
        if np.isnan(pivot_value):
            continue

        previous_price = last_price
        previous_signed = last_signed
        last_price = pivot_value
        pivot_signed = signed_values[position - config.pivot_lookback] if position >= config.pivot_lookback else np.nan
        last_signed = pivot_signed

        if (
            np.isfinite(previous_price)
            and np.isfinite(previous_signed)
            and pivot_value < previous_price * (1.0 - config.min_price_drop_pct / 100.0)
            and np.isfinite(pivot_signed)
            and pivot_signed > previous_signed
        ):
            bullish_div.iloc[position] = True

    return bullish_div


def compute_vpin_lsf(bars: pd.DataFrame, config: VPINLSFConfig | None = None) -> pd.DataFrame:
    config = config or VPINLSFConfig()
    _validate_bars(bars)

    close = bars["close"].astype(float)
    low = bars["low"].astype(float)
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
    vpin_cdf = _rolling_percent_rank(vpin, config.cdf_len)
    buyside_cdf = _rolling_percent_rank(vpin_signed, config.cdf_len)

    price_pivot = _detect_confirmed_pivot_lows(low, config.pivot_lookback)
    bullish_div = _detect_bullish_divergence(price_pivot, vpin_signed, config)

    crossunder = (vpin_signed <= config.sign_trigger) & (vpin_signed.shift(1) > config.sign_trigger)
    long_signal = crossunder.shift(1, fill_value=False).astype(bool)

    result = pd.DataFrame(index=bars.index)
    result["dp"] = dp
    result["sigma_dp"] = sigma_dp
    result["z_score"] = z_score
    result["v_buy"] = v_buy
    result["v_sell"] = v_sell
    result["sum_buy"] = sum_buy
    result["sum_sell"] = sum_sell
    result["sum_vol"] = sum_vol
    result["vpin"] = vpin
    result["vpin_signed"] = vpin_signed
    result["vpin_cdf"] = vpin_cdf
    result["buyside_cdf"] = buyside_cdf
    result["p50"] = vpin.rolling(config.cdf_len, min_periods=config.cdf_len).quantile(0.50, interpolation="nearest")
    result["p75"] = vpin.rolling(config.cdf_len, min_periods=config.cdf_len).quantile(0.75, interpolation="nearest")
    result["p90"] = vpin.rolling(config.cdf_len, min_periods=config.cdf_len).quantile(0.90, interpolation="nearest")
    result["price_pivot"] = price_pivot
    result["bullish_div"] = bullish_div
    result["long_signal"] = long_signal
    result["div_vpin_y"] = vpin.shift(config.pivot_lookback).where(bullish_div)
    result["buy_pct"] = (sum_buy / sum_vol * 100.0).where(sum_vol > 0)
    result["signal_text"] = np.where(long_signal, "LONG", np.where(bullish_div, "DIV", "-"))
    return result
