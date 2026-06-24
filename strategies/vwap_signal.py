from __future__ import annotations

import math

import pandas as pd

from portfolio import FLATTEN, TARGET, Order, Portfolio
from .base import BaseStrategy


VWAP_SIGNAL_MODE_ANALOG = "analog"
VWAP_SIGNAL_MODE_BINARY = "binary"


class VWAPSignalStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str = "SPY",
        signal_mode: str = VWAP_SIGNAL_MODE_ANALOG,
        base_exposure: float = 0.5,
        max_exposure: float = 3.0,
        close_after_bars: int | None = None,
    ) -> None:
        if signal_mode not in {VWAP_SIGNAL_MODE_ANALOG, VWAP_SIGNAL_MODE_BINARY}:
            raise ValueError("signal_mode must be one of: analog, binary.")
        if base_exposure <= 0:
            raise ValueError("base_exposure must be positive.")
        if max_exposure <= 0:
            raise ValueError("max_exposure must be positive.")
        if close_after_bars is not None and close_after_bars <= 0:
            raise ValueError("close_after_bars must be a positive integer when set.")

        self.symbol = symbol
        self.signal_mode = signal_mode
        self.base_exposure = float(base_exposure)
        self.max_exposure = float(max_exposure)
        self.close_after_bars = close_after_bars
        self.bars_in_position = 0

    def generate_orders(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        portfolio: Portfolio,
    ) -> list[Order]:
        position = portfolio.get_position(self.symbol)
        price = _finite_float(row.get("close", 0.0), default=0.0)
        vwap = _finite_float(row.get("vwap", math.nan), default=math.nan)

        if price <= 0:
            return []

        if position.quantity != 0:
            self.bars_in_position += 1

        if position.quantity != 0 and self.close_after_bars is not None and self.bars_in_position >= self.close_after_bars:
            self._reset_position_state()
            return [
                Order(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    action=FLATTEN,
                    target_exposure=0.0,
                    reason=f"Timed exit after {self.close_after_bars} bars",
                )
            ]

        if position.quantity != 0 and math.isfinite(vwap) and _touches_vwap(position.quantity, price, vwap):
            self._reset_position_state()
            return [
                Order(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    action=FLATTEN,
                    target_exposure=0.0,
                    reason="VWAP mean reversion exit",
                )
            ]

        signal = _finite_float(row.get("vwap_signal", 0.0), default=0.0)
        if signal == 0.0:
            return []

        target_exposure = self._target_exposure(signal)
        current_exposure = portfolio.snapshot(timestamp, {self.symbol: price}).net_exposure_pct
        if abs(target_exposure - current_exposure) <= 1e-9:
            return []

        if position.quantity == 0:
            self.bars_in_position = 0

        return [
            Order(
                timestamp=timestamp,
                symbol=self.symbol,
                action=TARGET,
                target_exposure=target_exposure,
                reason=f"VWAP {self.signal_mode} signal",
            )
        ]

    def _target_exposure(self, signal: float) -> float:
        direction = 1.0 if signal > 0 else -1.0
        if self.signal_mode == VWAP_SIGNAL_MODE_BINARY:
            exposure = self.base_exposure
        else:
            exposure = self.base_exposure * abs(signal)
        return direction * min(self.max_exposure, exposure)

    def _reset_position_state(self) -> None:
        self.bars_in_position = 0


def _touches_vwap(quantity: float, price: float, vwap: float) -> bool:
    if quantity > 0:
        return price >= vwap
    return price <= vwap


def _finite_float(value: object, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result
