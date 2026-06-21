from __future__ import annotations

import unittest

import pandas as pd

from portfolio import Portfolio
from backtester import add_buy_and_hold_benchmark


class PortfolioAccountingTests(unittest.TestCase):
    def test_buy_to_target_exposure_and_mark_unrealized_pnl(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        ts = pd.Timestamp("2026-01-01 09:30")

        fill = portfolio.buy(ts, price=100.0, target_exposure=1.0)
        snapshot = portfolio.mark(ts, {"SPY": 110.0})

        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.quantity, 1_000.0)
        self.assertAlmostEqual(portfolio.cash, 0.0)
        self.assertAlmostEqual(snapshot.equity, 110_000.0)
        self.assertAlmostEqual(snapshot.unrealized_pnl, 10_000.0)
        self.assertAlmostEqual(snapshot.net_exposure_pct, 1.0)
        self.assertAlmostEqual(snapshot.delta_exposure_pct, 1.0)

    def test_sell_can_cross_from_long_to_short(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        ts = pd.Timestamp("2026-01-01 09:30")

        portfolio.buy(ts, price=100.0, target_exposure=1.0)
        fill = portfolio.sell(ts, price=110.0, target_exposure=-0.5)
        snapshot = portfolio.mark(ts, {"SPY": 110.0})
        position = portfolio.get_position("SPY")

        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.quantity, -1_500.0)
        self.assertAlmostEqual(position.quantity, -500.0)
        self.assertAlmostEqual(position.average_price, 110.0)
        self.assertAlmostEqual(position.realized_pnl, 10_000.0)
        self.assertAlmostEqual(snapshot.equity, 110_000.0)
        self.assertAlmostEqual(snapshot.net_exposure_pct, -0.5)

    def test_flatten_closes_short_and_updates_realized_pnl(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        ts = pd.Timestamp("2026-01-01 09:30")

        portfolio.buy(ts, price=100.0, target_exposure=1.0)
        portfolio.sell(ts, price=110.0, target_exposure=-0.5)
        flatten = portfolio.flatten(ts, price=100.0)
        snapshot = portfolio.mark(ts, {"SPY": 100.0})
        position = portfolio.get_position("SPY")

        self.assertIsNotNone(flatten)
        self.assertAlmostEqual(flatten.quantity, 500.0)
        self.assertAlmostEqual(position.quantity, 0.0)
        self.assertAlmostEqual(position.average_price, 0.0)
        self.assertAlmostEqual(position.realized_pnl, 15_000.0)
        self.assertAlmostEqual(snapshot.equity, 115_000.0)
        self.assertAlmostEqual(snapshot.total_pnl, 15_000.0)
        self.assertAlmostEqual(snapshot.gross_exposure_pct, 0.0)

    def test_delta_exposure_uses_position_delta_per_unit(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY", delta_per_unit=0.5)
        ts = pd.Timestamp("2026-01-01 09:30")

        portfolio.buy(ts, price=100.0, target_exposure=1.0)
        snapshot = portfolio.mark(ts, {"SPY": 100.0})

        self.assertAlmostEqual(snapshot.net_exposure_pct, 1.0)
        self.assertAlmostEqual(snapshot.delta_exposure_pct, 0.5)

    def test_three_times_target_exposure_creates_300_delta(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        ts = pd.Timestamp("2026-01-01 09:30")

        fill = portfolio.buy(ts, price=100.0, target_exposure=3.0)
        snapshot = portfolio.mark(ts, {"SPY": 100.0})

        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.quantity, 3_000.0)
        self.assertAlmostEqual(portfolio.cash, -200_000.0)
        self.assertAlmostEqual(snapshot.net_exposure_pct, 3.0)
        self.assertAlmostEqual(snapshot.delta_exposure_pct, 3.0)

    def test_buy_and_hold_benchmark_uses_first_close(self) -> None:
        bars = pd.DataFrame(
            {"close": [100.0, 105.0, 95.0]},
            index=pd.date_range("2026-01-01 09:30", periods=3, freq="min"),
        )

        benchmark = add_buy_and_hold_benchmark(bars, initial_cash=100_000)

        self.assertAlmostEqual(benchmark["buy_hold_quantity"].iloc[0], 1_000.0)
        self.assertAlmostEqual(benchmark["buy_hold_equity"].iloc[1], 105_000.0)
        self.assertAlmostEqual(benchmark["buy_hold_pnl"].iloc[2], -5_000.0)
        self.assertAlmostEqual(benchmark["buy_hold_return_pct"].iloc[1], 0.05)


if __name__ == "__main__":
    unittest.main()
