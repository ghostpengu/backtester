from __future__ import annotations

import pandas as pd

from portfolio import FLATTEN, TARGET, Order, Portfolio
from .base import BaseStrategy


class VPINDemoStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str = "SPY",
        entry_step_exposure: float = 1,
        max_exposure: float = 5.0,
        trigger_level: float = -0.279,
        exit_signed_above: float = 1,
        close_after_bars: int | None = None,
    ) -> None:
        if close_after_bars is not None and close_after_bars <= 0:
            raise ValueError("close_after_bars must be a positive integer when set.")

        self.symbol = symbol
        self.entry_step_exposure = float(entry_step_exposure)
        self.max_exposure = float(max_exposure)
        self.trigger_level = float(trigger_level)
        self.exit_signed_above = float(exit_signed_above)
        self.close_after_bars = close_after_bars
        self.is_armed = False
        self.last_add_date: object | None = None
        self.bars_in_position = 0

    def generate_orders(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        portfolio: Portfolio,
    ) -> list[Order]:
        position = portfolio.get_position(self.symbol)
        orders: list[Order] = []
        signed_vpin = float(row.get("vpin_signed", 0.0))
        price = float(row.get("close", 0.0))

        if position.quantity != 0:
            self.bars_in_position += 1

        if position.quantity != 0 and self.close_after_bars is not None and self.bars_in_position >= self.close_after_bars:
            orders.append(
                Order(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    action=FLATTEN,
                    target_exposure=0.0,
                    reason=f"Timed exit after {self.close_after_bars} bars",
                )
            )
            self._reset_position_state()
            return orders

        if position.quantity != 0 and signed_vpin >= self.exit_signed_above:
            orders.append(
                Order(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    action=FLATTEN,
                    target_exposure=0.0,
                    reason="VPIN signed recovered",
                )
            )
            self._reset_position_state()
            return orders

        if bool(row.get("long_signal", False)):
            self.is_armed = True

        if not self.is_armed or signed_vpin > self.trigger_level:
            return orders

        current_date = timestamp.date()
        if self.last_add_date == current_date:
            return orders

        current_exposure = portfolio.snapshot(timestamp, {self.symbol: price}).net_exposure_pct
        target_exposure = min(self.max_exposure, current_exposure + self.entry_step_exposure)
        if target_exposure <= current_exposure + 1e-9:
            return orders

        self.last_add_date = current_date
        if position.quantity == 0:
            self.bars_in_position = 0
        orders.append(
            Order(
                timestamp=timestamp,
                symbol=self.symbol,
                action=TARGET,
                target_exposure=target_exposure,
                reason="VPIN below trigger scale-in",
            )
        )

        return orders

    def _reset_position_state(self) -> None:
        self.is_armed = False
        self.last_add_date = None
        self.bars_in_position = 0
