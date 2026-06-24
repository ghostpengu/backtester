from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from feature_tester import compute_indicator_features, parse_timeframes, select_feature_columns
from indicators import VWAPBandsConfig, compute_vwap_bands


class VWAPBandsIndicatorTests(unittest.TestCase):
    def test_compute_vwap_bands_outputs_expected_columns(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0 + i for i in range(40)],
                "high": [101.0 + i for i in range(40)],
                "low": [99.0 + i for i in range(40)],
                "close": [100.5 + i for i in range(40)],
                "volume": [1_000.0 + i * 10 for i in range(40)],
            },
            index=pd.date_range("2026-01-01 09:30", periods=40, freq="min"),
        )

        result = compute_vwap_bands(bars, VWAPBandsConfig())

        self.assertIn("vwap", result.columns)
        self.assertIn("vwap_std", result.columns)
        self.assertIn("vwap_zscore", result.columns)
        self.assertIn("vwap_signal", result.columns)
        self.assertIn("vwap_signal_strength", result.columns)
        self.assertIn("vwap_upper_2sigma", result.columns)
        self.assertIn("vwap_lower_3sigma", result.columns)
        self.assertIn("vwap_cross_upper_2sigma", result.columns)
        self.assertEqual(len(result), len(bars))

    def test_vwap_matches_volume_weighted_average_for_session(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0, 110.0],
                "high": [100.0, 110.0],
                "low": [100.0, 110.0],
                "close": [100.0, 110.0],
                "volume": [100.0, 100.0],
            },
            index=pd.date_range("2026-01-01 09:30", periods=2, freq="min"),
        )

        result = compute_vwap_bands(bars, VWAPBandsConfig(typical_price="close"))

        self.assertAlmostEqual(result["vwap"].iloc[0], 100.0)
        self.assertAlmostEqual(result["vwap"].iloc[1], 105.0)
        self.assertAlmostEqual(result["vwap_std"].iloc[1], 5.0)
        self.assertAlmostEqual(result["vwap_upper_2sigma"].iloc[1], 115.0)
        self.assertAlmostEqual(result["vwap_lower_2sigma"].iloc[1], 95.0)
        self.assertAlmostEqual(result["vwap_upper_3sigma"].iloc[1], 120.0)
        self.assertAlmostEqual(result["vwap_lower_3sigma"].iloc[1], 90.0)

    def test_vwap_resets_each_session(self) -> None:
        day_one = pd.date_range("2026-01-01 09:30", periods=2, freq="min")
        day_two = pd.date_range("2026-01-02 09:30", periods=2, freq="min")
        index = day_one.union(day_two)
        bars = pd.DataFrame(
            {
                "open": [100.0, 110.0, 200.0, 220.0],
                "high": [100.0, 110.0, 200.0, 220.0],
                "low": [100.0, 110.0, 200.0, 220.0],
                "close": [100.0, 110.0, 200.0, 220.0],
                "volume": [100.0, 100.0, 100.0, 100.0],
            },
            index=index,
        )

        result = compute_vwap_bands(bars, VWAPBandsConfig(typical_price="close"))

        self.assertAlmostEqual(result["vwap"].iloc[2], 200.0)
        self.assertAlmostEqual(result["vwap"].iloc[3], 210.0)
        self.assertAlmostEqual(result["vwap_std"].iloc[2], 0.0)
        self.assertAlmostEqual(result["vwap_std"].iloc[3], 10.0)

    def test_rolling_vwap_lookbacks_do_not_reset_each_session(self) -> None:
        day_one = pd.date_range("2026-01-01 09:30", periods=2, freq="min")
        day_two = pd.date_range("2026-01-02 09:30", periods=2, freq="min")
        index = day_one.union(day_two)
        bars = pd.DataFrame(
            {
                "open": [100.0, 110.0, 200.0, 220.0],
                "high": [100.0, 110.0, 200.0, 220.0],
                "low": [100.0, 110.0, 200.0, 220.0],
                "close": [100.0, 110.0, 200.0, 220.0],
                "volume": [100.0, 100.0, 100.0, 100.0],
            },
            index=index,
        )

        for lookback in ("1w", "1y"):
            with self.subTest(lookback=lookback):
                result = compute_vwap_bands(
                    bars,
                    VWAPBandsConfig(lookback=lookback, typical_price="close"),
                )

                self.assertAlmostEqual(result["vwap"].iloc[2], (100.0 + 110.0 + 200.0) / 3.0)
                self.assertNotAlmostEqual(result["vwap"].iloc[2], 200.0)

    def test_vwap_signal_is_signed_and_strength_increases_past_threshold(self) -> None:
        index = pd.date_range("2026-01-01 09:30", periods=20, freq="min")
        upper_close = [100.0] * 19 + [110.0]
        lower_close = [100.0] * 19 + [90.0]

        upper_bars = pd.DataFrame(
            {
                "open": upper_close,
                "high": upper_close,
                "low": upper_close,
                "close": upper_close,
                "volume": [100.0] * 20,
            },
            index=index,
        )
        lower_bars = pd.DataFrame(
            {
                "open": lower_close,
                "high": lower_close,
                "low": lower_close,
                "close": lower_close,
                "volume": [100.0] * 20,
            },
            index=index,
        )

        upper_result = compute_vwap_bands(upper_bars, VWAPBandsConfig(typical_price="close"))
        lower_result = compute_vwap_bands(lower_bars, VWAPBandsConfig(typical_price="close"))

        self.assertEqual(upper_result["vwap_signal"].iloc[1], 0.0)
        self.assertLess(upper_result["vwap_signal"].iloc[-1], 0.0)
        self.assertGreater(lower_result["vwap_signal"].iloc[-1], 0.0)
        self.assertGreater(upper_result["vwap_signal_strength"].iloc[-1], 1.0)

    def test_cross_signals_trigger_on_band_penetration(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0, 99.0, 105.0],
                "high": [100.0, 99.0, 105.0],
                "low": [100.0, 99.0, 105.0],
                "close": [100.0, 99.0, 105.0],
                "volume": [100.0, 100.0, 100.0],
            },
            index=pd.date_range("2026-01-01 09:30", periods=3, freq="min"),
        )

        result = compute_vwap_bands(
            bars,
            VWAPBandsConfig(sigma_levels=(1.0,), typical_price="close"),
        )

        self.assertFalse(result["vwap_above_upper_1sigma"].iloc[1])
        self.assertFalse(result["vwap_cross_upper_1sigma"].iloc[0])
        self.assertFalse(result["vwap_cross_upper_1sigma"].iloc[1])
        self.assertTrue(result["vwap_cross_upper_1sigma"].iloc[2])


class VWAPFeatureTesterTests(unittest.TestCase):
    def test_indicator_features_include_vwap_columns(self) -> None:
        periods = 360
        index = pd.date_range("2026-01-01 09:30", periods=periods, freq="min")
        close = 100.0 + np.sin(np.arange(periods) / 8.0).cumsum() * 0.05
        bars = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": 1_000.0 + np.arange(periods),
            },
            index=index,
        )

        indicators = compute_indicator_features(
            bars,
            timeframes=parse_timeframes("1m,5m"),
        )

        self.assertIn("vwap_1m", indicators.columns)
        self.assertIn("vwap_zscore_1m", indicators.columns)
        self.assertIn("vwap_signal_1m", indicators.columns)
        self.assertIn("vwap_upper_2sigma_5m", indicators.columns)
        self.assertIn("vwap_cross_lower_3sigma_5m", indicators.columns)

    def test_vwap_feature_selection_excludes_vpin_and_ohlcv(self) -> None:
        frame = pd.DataFrame(
            {
                "range_mean_15b": [0.1, 0.2],
                "vpin_1m": [0.3, 0.4],
                "vwap_zscore_15m": [1.2, -0.5],
                "vwap_signal_15m": [0.0, -1.2],
                "vwap_cross_upper_2sigma_1h": [0, 1],
                "fwd_return_5m": [0.01, -0.01],
            }
        )

        vwap_features = select_feature_columns(frame, feature_set="vwap")
        all_features = select_feature_columns(frame, feature_set="all")
        ohlcv_features = select_feature_columns(frame, feature_set="ohlcv")

        self.assertIn("vwap_zscore_15m", vwap_features)
        self.assertIn("vwap_signal_15m", vwap_features)
        self.assertIn("vwap_cross_upper_2sigma_1h", vwap_features)
        self.assertNotIn("vpin_1m", vwap_features)
        self.assertNotIn("range_mean_15b", vwap_features)
        self.assertIn("vwap_zscore_15m", all_features)
        self.assertNotIn("vwap_zscore_15m", ohlcv_features)


if __name__ == "__main__":
    unittest.main()
