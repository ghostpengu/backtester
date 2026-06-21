from __future__ import annotations

import unittest

import pandas as pd

from portfolio.performance import compute_performance_metrics, format_performance_debug


class PerformanceMetricsTests(unittest.TestCase):
    def test_computes_return_drawdown_and_cagr_for_multi_day_curve(self) -> None:
        equity = pd.Series(
            [100_000.0, 110_000.0, 105_000.0, 121_000.0],
            index=pd.to_datetime(
                [
                    "2026-01-01 10:00",
                    "2026-01-02 10:00",
                    "2026-01-03 10:00",
                    "2026-01-04 10:00",
                ]
            ),
        )

        metrics = compute_performance_metrics(equity, "Strategy")

        self.assertAlmostEqual(metrics.total_pnl, 21_000.0)
        self.assertAlmostEqual(metrics.total_return, 0.21)
        self.assertAlmostEqual(metrics.max_drawdown, -5_000.0)
        self.assertAlmostEqual(metrics.max_drawdown_pct, -5_000.0 / 110_000.0)
        self.assertIsNotNone(metrics.cagr)
        self.assertIsNotNone(metrics.sharpe)

    def test_cagr_is_none_for_single_intraday_session(self) -> None:
        equity = pd.Series(
            [100_000.0, 100_100.0, 100_200.0],
            index=pd.to_datetime(
                [
                    "2026-01-01 10:00",
                    "2026-01-01 11:00",
                    "2026-01-01 12:00",
                ]
            ),
        )

        metrics = compute_performance_metrics(equity, "Strategy")

        self.assertIsNone(metrics.cagr)
        self.assertEqual(metrics.trading_days, 1)

    def test_formats_strategy_and_benchmark_debug(self) -> None:
        index = pd.to_datetime(["2026-01-01 10:00", "2026-01-01 11:00"])
        chart_data = pd.DataFrame(
            {
                "equity": [100_000.0, 100_250.0],
                "buy_hold_equity": [100_000.0, 100_100.0],
                "delta_exposure_pct": [0.0, 0.5],
                "net_exposure_pct": [0.0, 0.5],
                "realized_pnl": [0.0, 100.0],
                "unrealized_pnl": [0.0, 150.0],
            },
            index=index,
        )
        fills = pd.DataFrame(
            {
                "quantity": [50.0],
                "notional": [50_000.0],
            },
            index=[index[1]],
        )

        output = format_performance_debug(chart_data, fills)

        self.assertIn("Performance Debug", output)
        self.assertIn("Strategy:", output)
        self.assertIn("Buy & Hold:", output)
        self.assertIn("Max delta exposure", output)
        self.assertIn("Fills: 1", output)


if __name__ == "__main__":
    unittest.main()
