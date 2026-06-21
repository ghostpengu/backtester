from __future__ import annotations

import unittest

import pandas as pd

from portfolio import FLATTEN, TARGET, Portfolio
from strategies import VPINDemoStrategy


class VPINDemoStrategyTests(unittest.TestCase):
    def test_scales_in_once_per_date_after_signal_while_below_trigger(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VPINDemoStrategy(
            entry_step_exposure=0.5,
            max_exposure=3.0,
            trigger_level=-0.279,
            exit_signed_above=0.15,
        )

        first_timestamp = pd.Timestamp("2026-01-01 10:00")
        first_row = pd.Series({"close": 100.0, "long_signal": True, "vpin_signed": -0.40})
        first_orders = strategy.generate_orders(first_timestamp, first_row, portfolio)

        self.assertEqual(len(first_orders), 1)
        self.assertEqual(first_orders[0].action, TARGET)
        self.assertAlmostEqual(first_orders[0].target_exposure, 0.5)
        portfolio.execute_order(first_orders[0], price=100.0)
        portfolio.mark(first_timestamp, {"SPY": 100.0})

        same_day_row = pd.Series({"close": 100.0, "long_signal": False, "vpin_signed": -0.45})
        same_day_orders = strategy.generate_orders(pd.Timestamp("2026-01-01 12:00"), same_day_row, portfolio)
        self.assertEqual(same_day_orders, [])

        next_day_row = pd.Series({"close": 100.0, "long_signal": False, "vpin_signed": -0.45})
        next_day_orders = strategy.generate_orders(pd.Timestamp("2026-01-02 10:00"), next_day_row, portfolio)
        self.assertEqual(len(next_day_orders), 1)
        self.assertAlmostEqual(next_day_orders[0].target_exposure, 1.0)

    def test_recovery_flattens_and_resets_scale_in_state(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VPINDemoStrategy(
            entry_step_exposure=0.5,
            max_exposure=3.0,
            trigger_level=-0.279,
            exit_signed_above=0.15,
        )

        first_timestamp = pd.Timestamp("2026-01-01 10:00")
        first_row = pd.Series({"close": 100.0, "long_signal": True, "vpin_signed": -0.40})
        first_order = strategy.generate_orders(first_timestamp, first_row, portfolio)[0]
        portfolio.execute_order(first_order, price=100.0)
        portfolio.mark(first_timestamp, {"SPY": 100.0})

        recovery_row = pd.Series({"close": 101.0, "long_signal": False, "vpin_signed": 0.16})
        recovery_orders = strategy.generate_orders(pd.Timestamp("2026-01-01 13:00"), recovery_row, portfolio)

        self.assertEqual(len(recovery_orders), 1)
        self.assertEqual(recovery_orders[0].action, FLATTEN)
        self.assertEqual(recovery_orders[0].target_exposure, 0.0)

    def test_close_after_bars_flattens_open_position(self) -> None:
        portfolio = Portfolio(initial_cash=100_000, default_symbol="SPY")
        strategy = VPINDemoStrategy(
            entry_step_exposure=0.5,
            max_exposure=3.0,
            trigger_level=-0.279,
            close_after_bars=2,
        )

        entry_timestamp = pd.Timestamp("2026-01-01 10:00")
        entry_row = pd.Series({"close": 100.0, "long_signal": True, "vpin_signed": -0.40})
        entry_order = strategy.generate_orders(entry_timestamp, entry_row, portfolio)[0]
        portfolio.execute_order(entry_order, price=100.0)
        portfolio.mark(entry_timestamp, {"SPY": 100.0})

        first_hold_row = pd.Series({"close": 100.0, "long_signal": False, "vpin_signed": -0.40})
        first_hold_orders = strategy.generate_orders(pd.Timestamp("2026-01-01 10:01"), first_hold_row, portfolio)
        self.assertEqual(first_hold_orders, [])

        second_hold_row = pd.Series({"close": 100.0, "long_signal": False, "vpin_signed": -0.40})
        timed_exit_orders = strategy.generate_orders(pd.Timestamp("2026-01-01 10:02"), second_hold_row, portfolio)

        self.assertEqual(len(timed_exit_orders), 1)
        self.assertEqual(timed_exit_orders[0].action, FLATTEN)
        self.assertEqual(timed_exit_orders[0].target_exposure, 0.0)
        self.assertEqual(timed_exit_orders[0].reason, "Timed exit after 2 bars")

    def test_close_after_bars_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            VPINDemoStrategy(close_after_bars=0)


if __name__ == "__main__":
    unittest.main()
