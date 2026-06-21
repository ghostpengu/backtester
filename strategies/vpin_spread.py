from __future__ import annotations

import pandas as pd

from portfolio import FLATTEN, TARGET, Order, Portfolio
from .base import BaseStrategy


class VPINSpreadStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str = "SPY",
        entry_step_exposure: float = 0.5,
        max_exposure: float = 3.0,
        close_after_bars: int | None = None,
    ) -> None:
        if close_after_bars is not None and close_after_bars <= 0:
            raise ValueError("close_after_bars must be a positive integer when set.")

        self.symbol = symbol
        self.entry_step_exposure = float(entry_step_exposure)
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
        price = float(row.get("close", 0.0))

        if position.quantity != 0:
            self.bars_in_position += 1

        if position.quantity != 0 and self.close_after_bars is not None and self.bars_in_position >= self.close_after_bars:
            self.bars_in_position = 0
            return [
                Order(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    action=FLATTEN,
                    target_exposure=0.0,
                    reason=f"Timed exit after {self.close_after_bars} bars",
                )
            ]

        if not bool(row.get("spread_cross_top", False)):
            return []

        current_exposure = portfolio.snapshot(timestamp, {self.symbol: price}).net_exposure_pct
        target_exposure = min(self.max_exposure, current_exposure + self.entry_step_exposure)
        if target_exposure <= current_exposure + 1e-9:
            return []

        if position.quantity == 0:
            self.bars_in_position = 0

        return [
            Order(
                timestamp=timestamp,
                symbol=self.symbol,
                action=TARGET,
                target_exposure=target_exposure,
                reason="VPIN spread crossed top limit",
            )
        ]
