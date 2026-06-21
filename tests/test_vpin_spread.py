from __future__ import annotations

import unittest

import pandas as pd

from indicators import VPINSpreadConfig, compute_vpin_spread
from portfolio import FLATTEN, TARGET, Portfolio
from strategies import VPINSpreadStrategy


class VPINSpreadIndicatorTests(unittest.TestCase):
    def test_compute_vpin_spread_outputs_signal_columns(self) -> None:
        bars = pd.DataFrame(
            {
                "close": [100 + i * 0.1 for i in range(80)],
                "volume": [1_000 + i for i in range(80)],
            },
            index=pd.date_range("2026-01-01 09:30", periods=80, freq="min"),
        )

        result = compute_vpin_spread(bars, VPINSpreadConfig(n_bars=10, sigma_len=5))

        self.assertIn("vpin_spread", result.columns)
        self.assertIn("spread_cross_top", result.columns)
        self.assertIn("spread_cross_bottom", result.columns)
        self.assertEqual(len(result), len(bars))


class VPINSpreadStrategyTests(unittest.TestCase):
    def test_spread_cross_top_buys_target_exposure_step(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VPINSpreadStrategy(entry_step_exposure=0.5, max_exposure=3.0)
        timestamp = pd.Timestamp("2026-01-01 10:00")
        row = pd.Series({"close": 100.0, "spread_cross_top": True})

        orders = strategy.generate_orders(timestamp, row, portfolio)

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].action, TARGET)
        self.assertAlmostEqual(orders[0].target_exposure, 0.5)

    def test_no_cross_does_not_buy(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VPINSpreadStrategy(entry_step_exposure=0.5, max_exposure=3.0)
        row = pd.Series({"close": 100.0, "spread_cross_top": False})

        orders = strategy.generate_orders(pd.Timestamp("2026-01-01 10:00"), row, portfolio)

        self.assertEqual(orders, [])

    def test_close_after_bars_flattens_spread_position(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VPINSpreadStrategy(entry_step_exposure=0.5, max_exposure=3.0, close_after_bars=2)
        entry_timestamp = pd.Timestamp("2026-01-01 10:00")
        entry_order = strategy.generate_orders(
            entry_timestamp,
            pd.Series({"close": 100.0, "spread_cross_top": True}),
            portfolio,
        )[0]
        portfolio.execute_order(entry_order, 100.0)
        portfolio.mark(entry_timestamp, {"SPY": 100.0})

        first_hold = strategy.generate_orders(
            pd.Timestamp("2026-01-01 10:01"),
            pd.Series({"close": 100.0, "spread_cross_top": False}),
            portfolio,
        )
        second_hold = strategy.generate_orders(
            pd.Timestamp("2026-01-01 10:02"),
            pd.Series({"close": 100.0, "spread_cross_top": False}),
            portfolio,
        )

        self.assertEqual(first_hold, [])
        self.assertEqual(len(second_hold), 1)
        self.assertEqual(second_hold[0].action, FLATTEN)
        self.assertEqual(second_hold[0].reason, "Timed exit after 2 bars")


if __name__ == "__main__":
    unittest.main()
