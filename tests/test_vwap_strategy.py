from __future__ import annotations

import unittest

import pandas as pd

from portfolio import FLATTEN, TARGET, Portfolio
from strategies import VWAPSignalStrategy


class VWAPSignalStrategyTests(unittest.TestCase):
    def test_analog_signal_targets_signed_mean_reversion_exposure(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VWAPSignalStrategy(base_exposure=0.5, max_exposure=3.0)
        timestamp = pd.Timestamp("2026-01-01 10:00")

        upper_orders = strategy.generate_orders(
            timestamp,
            pd.Series({"close": 110.0, "vwap": 100.0, "vwap_signal": -1.5}),
            portfolio,
        )
        lower_orders = strategy.generate_orders(
            timestamp,
            pd.Series({"close": 90.0, "vwap": 100.0, "vwap_signal": 1.5}),
            portfolio,
        )

        self.assertEqual(len(upper_orders), 1)
        self.assertEqual(upper_orders[0].action, TARGET)
        self.assertAlmostEqual(upper_orders[0].target_exposure, -0.75)
        self.assertEqual(len(lower_orders), 1)
        self.assertAlmostEqual(lower_orders[0].target_exposure, 0.75)

    def test_analog_signal_caps_absolute_exposure(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VWAPSignalStrategy(base_exposure=0.5, max_exposure=3.0)

        orders = strategy.generate_orders(
            pd.Timestamp("2026-01-01 10:00"),
            pd.Series({"close": 110.0, "vwap": 100.0, "vwap_signal": -10.0}),
            portfolio,
        )

        self.assertEqual(len(orders), 1)
        self.assertAlmostEqual(orders[0].target_exposure, -3.0)

    def test_binary_signal_uses_fixed_exposure(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VWAPSignalStrategy(signal_mode="binary", base_exposure=0.5, max_exposure=3.0)

        orders = strategy.generate_orders(
            pd.Timestamp("2026-01-01 10:00"),
            pd.Series({"close": 110.0, "vwap": 100.0, "vwap_signal": -2.5}),
            portfolio,
        )

        self.assertEqual(len(orders), 1)
        self.assertAlmostEqual(orders[0].target_exposure, -0.5)

    def test_vwap_touch_flattens_open_position(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VWAPSignalStrategy(base_exposure=0.5, max_exposure=3.0)
        entry_timestamp = pd.Timestamp("2026-01-01 10:00")
        entry_order = strategy.generate_orders(
            entry_timestamp,
            pd.Series({"close": 90.0, "vwap": 100.0, "vwap_signal": 1.0}),
            portfolio,
        )[0]
        portfolio.execute_order(entry_order, 90.0)
        portfolio.mark(entry_timestamp, {"SPY": 90.0})

        exit_orders = strategy.generate_orders(
            pd.Timestamp("2026-01-01 10:01"),
            pd.Series({"close": 100.0, "vwap": 100.0, "vwap_signal": 0.0}),
            portfolio,
        )

        self.assertEqual(len(exit_orders), 1)
        self.assertEqual(exit_orders[0].action, FLATTEN)
        self.assertEqual(exit_orders[0].target_exposure, 0.0)
        self.assertEqual(exit_orders[0].reason, "VWAP mean reversion exit")

    def test_close_after_bars_flattens_open_position(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VWAPSignalStrategy(base_exposure=0.5, max_exposure=3.0, close_after_bars=2)
        entry_timestamp = pd.Timestamp("2026-01-01 10:00")
        entry_order = strategy.generate_orders(
            entry_timestamp,
            pd.Series({"close": 90.0, "vwap": 100.0, "vwap_signal": 1.0}),
            portfolio,
        )[0]
        portfolio.execute_order(entry_order, 90.0)
        portfolio.mark(entry_timestamp, {"SPY": 90.0})

        first_hold = strategy.generate_orders(
            pd.Timestamp("2026-01-01 10:01"),
            pd.Series({"close": 90.0, "vwap": 100.0, "vwap_signal": 1.0}),
            portfolio,
        )
        second_hold = strategy.generate_orders(
            pd.Timestamp("2026-01-01 10:02"),
            pd.Series({"close": 90.0, "vwap": 100.0, "vwap_signal": 1.0}),
            portfolio,
        )

        self.assertEqual(first_hold, [])
        self.assertEqual(len(second_hold), 1)
        self.assertEqual(second_hold[0].action, FLATTEN)
        self.assertEqual(second_hold[0].reason, "Timed exit after 2 bars")


if __name__ == "__main__":
    unittest.main()
