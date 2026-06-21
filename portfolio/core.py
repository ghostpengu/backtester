from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import pandas as pd


BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"
FLATTEN = "FLATTEN"
TARGET = "TARGET"


@dataclass(frozen=True)
class Order:
    timestamp: pd.Timestamp
    symbol: str
    action: str
    target_exposure: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class Fill:
    timestamp: pd.Timestamp
    symbol: str
    action: str
    quantity: float
    price: float
    notional: float
    commission: float = 0.0
    slippage: float = 0.0
    realized_pnl: float = 0.0
    reason: str = ""


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0
    multiplier: float = 1.0
    delta_per_unit: float = 1.0

    def apply_fill(self, quantity: float, price: float) -> float:
        if quantity == 0:
            return 0.0

        previous_quantity = self.quantity
        previous_average = self.average_price
        realized = 0.0

        if previous_quantity == 0 or _same_direction(previous_quantity, quantity):
            new_quantity = previous_quantity + quantity
            self.average_price = (
                (abs(previous_quantity) * previous_average + abs(quantity) * price) / abs(new_quantity)
            )
            self.quantity = new_quantity
            return 0.0

        closing_quantity = min(abs(previous_quantity), abs(quantity))
        previous_direction = 1.0 if previous_quantity > 0 else -1.0
        realized = closing_quantity * (price - previous_average) * previous_direction * self.multiplier

        new_quantity = previous_quantity + quantity
        if new_quantity == 0:
            self.quantity = 0.0
            self.average_price = 0.0
        elif _same_direction(previous_quantity, new_quantity):
            self.quantity = new_quantity
        else:
            self.quantity = new_quantity
            self.average_price = price

        self.realized_pnl += realized
        return realized

    def market_value(self, price: float) -> float:
        return self.quantity * price * self.multiplier

    def unrealized_pnl(self, price: float) -> float:
        if self.quantity == 0:
            return 0.0
        return (price - self.average_price) * self.quantity * self.multiplier

    def delta_notional(self, price: float) -> float:
        return self.market_value(price) * self.delta_per_unit


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: pd.Timestamp
    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    gross_exposure: float
    net_exposure: float
    delta_exposure: float
    gross_exposure_pct: float
    net_exposure_pct: float
    delta_exposure_pct: float
    position_quantity: float
    position_average_price: float
    position_market_value: float
    position_unrealized_pnl: float


class Portfolio:
    def __init__(
        self,
        initial_cash: float = 100_000.0,
        default_symbol: str = "SPY",
        multiplier: float = 1.0,
        delta_per_unit: float = 1.0,
        commission_per_order: float = 0.0,
        commission_per_unit: float = 0.0,
        slippage_per_unit: float = 0.0,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.default_symbol = default_symbol
        self.multiplier = float(multiplier)
        self.delta_per_unit = float(delta_per_unit)
        self.commission_per_order = float(commission_per_order)
        self.commission_per_unit = float(commission_per_unit)
        self.slippage_per_unit = float(slippage_per_unit)
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.snapshots: list[PortfolioSnapshot] = []
        self.last_prices: dict[str, float] = {}

    def get_position(self, symbol: str | None = None) -> Position:
        symbol = symbol or self.default_symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                multiplier=self.multiplier,
                delta_per_unit=self.delta_per_unit,
            )
        return self.positions[symbol]

    def order_target_exposure(
        self,
        timestamp: pd.Timestamp,
        price: float,
        target_exposure: float,
        symbol: str | None = None,
        reason: str = "",
    ) -> Fill | None:
        order = Order(
            timestamp=timestamp,
            symbol=symbol or self.default_symbol,
            action=TARGET,
            target_exposure=target_exposure,
            reason=reason,
        )
        return self.execute_order(order, price)

    def buy(
        self,
        timestamp: pd.Timestamp,
        price: float,
        target_exposure: float = 1.0,
        symbol: str | None = None,
        reason: str = "",
    ) -> Fill | None:
        return self.execute_order(
            Order(timestamp, symbol or self.default_symbol, BUY, abs(target_exposure), reason),
            price,
        )

    def sell(
        self,
        timestamp: pd.Timestamp,
        price: float,
        target_exposure: float = -1.0,
        symbol: str | None = None,
        reason: str = "",
    ) -> Fill | None:
        return self.execute_order(
            Order(timestamp, symbol or self.default_symbol, SELL, -abs(target_exposure), reason),
            price,
        )

    def flatten(
        self,
        timestamp: pd.Timestamp,
        price: float,
        symbol: str | None = None,
        reason: str = "",
    ) -> Fill | None:
        return self.execute_order(
            Order(timestamp, symbol or self.default_symbol, FLATTEN, 0.0, reason),
            price,
        )

    def hold(self, timestamp: pd.Timestamp, symbol: str | None = None, reason: str = "") -> Order:
        return Order(timestamp, symbol or self.default_symbol, HOLD, None, reason)

    def execute_order(self, order: Order, price: float) -> Fill | None:
        if order.action == HOLD:
            return None

        target_exposure = self._resolve_target_exposure(order)
        position = self.get_position(order.symbol)
        marked_equity = self.equity({order.symbol: price})
        target_notional = marked_equity * target_exposure
        target_quantity = target_notional / (price * position.multiplier)
        quantity = target_quantity - position.quantity

        if abs(quantity) < 1e-9:
            return None

        fill_direction = 1.0 if quantity > 0 else -1.0
        fill_price = price + fill_direction * self.slippage_per_unit
        commission = self.commission_per_order + abs(quantity) * self.commission_per_unit
        notional = quantity * fill_price * position.multiplier

        realized = position.apply_fill(quantity, fill_price)
        self.cash -= notional + commission

        fill = Fill(
            timestamp=order.timestamp,
            symbol=order.symbol,
            action=order.action,
            quantity=quantity,
            price=fill_price,
            notional=notional,
            commission=commission,
            slippage=self.slippage_per_unit,
            realized_pnl=realized - commission,
            reason=order.reason,
        )
        self.fills.append(fill)
        self.last_prices[order.symbol] = fill_price
        return fill

    def mark(self, timestamp: pd.Timestamp, prices: Mapping[str, float]) -> PortfolioSnapshot:
        self.last_prices.update({symbol: float(price) for symbol, price in prices.items()})
        snapshot = self.snapshot(timestamp, self.last_prices)
        self.snapshots.append(snapshot)
        return snapshot

    def snapshot(self, timestamp: pd.Timestamp, prices: Mapping[str, float]) -> PortfolioSnapshot:
        equity = self.equity(prices)
        realized = sum(position.realized_pnl for position in self.positions.values())
        unrealized = sum(
            position.unrealized_pnl(float(prices.get(symbol, self.last_prices.get(symbol, 0.0))))
            for symbol, position in self.positions.items()
        )
        market_values = {
            symbol: position.market_value(float(prices.get(symbol, self.last_prices.get(symbol, 0.0))))
            for symbol, position in self.positions.items()
        }
        gross_exposure = sum(abs(value) for value in market_values.values())
        net_exposure = sum(market_values.values())
        delta_exposure = sum(
            position.delta_notional(float(prices.get(symbol, self.last_prices.get(symbol, 0.0))))
            for symbol, position in self.positions.items()
        )

        primary = self.get_position(self.default_symbol)
        primary_price = float(prices.get(self.default_symbol, self.last_prices.get(self.default_symbol, 0.0)))
        position_market_value = primary.market_value(primary_price)
        position_unrealized = primary.unrealized_pnl(primary_price)

        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            equity=equity,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            total_pnl=equity - self.initial_cash,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            delta_exposure=delta_exposure,
            gross_exposure_pct=_safe_divide(gross_exposure, equity),
            net_exposure_pct=_safe_divide(net_exposure, equity),
            delta_exposure_pct=_safe_divide(delta_exposure, equity),
            position_quantity=primary.quantity,
            position_average_price=primary.average_price,
            position_market_value=position_market_value,
            position_unrealized_pnl=position_unrealized,
        )

    def equity(self, prices: Mapping[str, float]) -> float:
        market_value = 0.0
        for symbol, position in self.positions.items():
            price = float(prices.get(symbol, self.last_prices.get(symbol, 0.0)))
            market_value += position.market_value(price)
        return self.cash + market_value

    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame()
        return pd.DataFrame(asdict(fill) for fill in self.fills).set_index("timestamp")

    def snapshots_frame(self) -> pd.DataFrame:
        if not self.snapshots:
            return pd.DataFrame()
        return pd.DataFrame(asdict(snapshot) for snapshot in self.snapshots).set_index("timestamp")

    def _resolve_target_exposure(self, order: Order) -> float:
        if order.action == FLATTEN:
            return 0.0
        if order.action in {BUY, SELL, TARGET}:
            if order.target_exposure is None:
                raise ValueError(f"{order.action} orders require target_exposure.")
            return float(order.target_exposure)
        raise ValueError(f"Unsupported order action: {order.action}")


def _same_direction(a: float, b: float) -> bool:
    return (a > 0 and b > 0) or (a < 0 and b < 0)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
