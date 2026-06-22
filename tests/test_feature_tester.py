from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from feature_tester import (
    HorizonSpec,
    build_explorer_series_bundle,
    build_top_features,
    collect_top_pairs,
    parse_feature_timeframe_label,
    parse_target_horizon_label,
    plotly_cdn_script,
    resample_explorer_frame,
    resolve_chart_timeframe,
    resolve_log_return_series,
    compute_indicator_features,
    compute_correlations,
    compute_forward_targets,
    compute_quantile_buckets,
    compute_technical_features,
    make_quantile_buckets,
    parse_horizons,
    parse_timeframes,
    run_feature_analysis,
    select_feature_columns,
    write_csv_outputs,
    write_html_report,
)


class FeatureTesterTests(unittest.TestCase):
    def test_forward_targets_use_future_rows_only(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0, 105.0, 110.0, 121.0],
                "high": [101.0, 107.0, 112.0, 125.0],
                "low": [99.0, 104.0, 108.0, 120.0],
                "close": [100.0, 105.0, 110.0, 121.0],
                "volume": [1_000.0, 1_100.0, 1_200.0, 1_300.0],
            },
            index=pd.date_range("2026-01-01 09:30", periods=4, freq="min"),
        )

        targets = compute_forward_targets(bars, [HorizonSpec(minutes=2, bars=2, label="2m")])

        self.assertAlmostEqual(targets["fwd_return_2m"].iloc[0], 0.10)
        self.assertAlmostEqual(targets["fwd_max_favorable_return_2m"].iloc[0], 0.12)
        self.assertAlmostEqual(targets["fwd_max_adverse_return_2m"].iloc[0], 0.04)
        expected_variance = np.log(105.0 / 100.0) ** 2 + np.log(110.0 / 105.0) ** 2
        self.assertAlmostEqual(targets["fwd_realized_variance_2m"].iloc[0], expected_variance)
        self.assertTrue(pd.isna(targets["fwd_return_2m"].iloc[-2]))

    def test_parse_horizons_supports_daily_return_horizons(self) -> None:
        horizons = parse_horizons("1d,7d,14d,28d,60d,365d,390")

        labels = [horizon.label for horizon in horizons]
        minutes = [horizon.minutes for horizon in horizons]

        self.assertEqual(labels, ["1d", "7d", "14d", "28d", "60d", "365d"])
        self.assertEqual(minutes[0], 390)
        self.assertEqual(minutes[1], 7 * 390)
        self.assertEqual(minutes[-1], 365 * 390)

    def test_realized_variance_features_match_known_returns(self) -> None:
        closes = [100.0, 101.0, 103.0, 102.0]
        bars = pd.DataFrame(
            {
                "open": closes,
                "high": [value + 1.0 for value in closes],
                "low": [value - 1.0 for value in closes],
                "close": closes,
                "volume": [1_000.0, 1_100.0, 1_200.0, 1_300.0],
            },
            index=pd.date_range("2026-01-01 09:30", periods=4, freq="min"),
        )

        features = compute_technical_features(bars, windows=(2,))

        expected = np.log(103.0 / 101.0) ** 2 + np.log(101.0 / 100.0) ** 2
        self.assertAlmostEqual(features["realized_variance_2b"].iloc[2], expected)
        self.assertAlmostEqual(features["realized_volatility_2b"].iloc[2], np.sqrt(expected))

    def test_quantile_buckets_include_counts_means_and_hit_rates(self) -> None:
        frame = pd.DataFrame(
            {
                "feature": [1.0, 2.0, 3.0, 4.0],
                "fwd_return_1m": [-0.02, -0.01, 0.01, 0.02],
            }
        )

        buckets = compute_quantile_buckets(
            frame,
            ["feature"],
            ["fwd_return_1m"],
            quantiles=2,
            min_observations=1,
        )

        self.assertEqual(list(buckets["bucket"]), [1, 2])
        self.assertEqual(list(buckets["bucket_count"]), [2, 2])
        self.assertAlmostEqual(buckets["target_mean"].iloc[0], -0.015)
        self.assertAlmostEqual(buckets["target_mean"].iloc[1], 0.015)
        self.assertAlmostEqual(buckets["hit_rate"].iloc[0], 0.0)
        self.assertAlmostEqual(buckets["hit_rate"].iloc[1], 1.0)
        self.assertAlmostEqual(buckets["bucket_target_mean_spearman"].iloc[0], 1.0)

    def test_correlations_skip_constant_features(self) -> None:
        frame = pd.DataFrame(
            {
                "constant": [1.0, 1.0, 1.0, 1.0],
                "changing": [1.0, 2.0, 3.0, 4.0],
                "fwd_return_1m": [0.01, 0.02, 0.03, 0.04],
            }
        )

        correlations = compute_correlations(
            frame,
            ["constant", "changing"],
            ["fwd_return_1m"],
            min_observations=2,
        )

        self.assertEqual(list(correlations["feature"]), ["changing"])
        self.assertAlmostEqual(correlations["spearman"].iloc[0], 1.0)

    def test_make_quantile_buckets_returns_nan_for_constant_values(self) -> None:
        buckets = make_quantile_buckets(pd.Series([1.0, 1.0, 1.0]), 2)

        self.assertTrue(buckets.isna().all())

    def test_default_feature_selection_is_vpin_focused(self) -> None:
        frame = pd.DataFrame(
            {
                "range_mean_15b": [0.1, 0.2],
                "sigma_dp_15m": [0.1, 0.2],
                "sum_buy_4h": [100.0, 200.0],
                "vpin_1m": [0.3, 0.4],
                "vpin_spread_15m": [0.01, 0.02],
                "spread_cross_top_1h": [0, 1],
                "fwd_return_5m": [0.01, -0.01],
            }
        )

        default_features = select_feature_columns(frame)
        all_features = select_feature_columns(frame, feature_set="all")
        ohlcv_features = select_feature_columns(frame, feature_set="ohlcv")

        self.assertIn("vpin_1m", default_features)
        self.assertIn("vpin_spread_15m", default_features)
        self.assertIn("spread_cross_top_1h", default_features)
        self.assertNotIn("range_mean_15b", default_features)
        self.assertNotIn("sigma_dp_15m", default_features)
        self.assertNotIn("sum_buy_4h", default_features)
        self.assertIn("range_mean_15b", all_features)
        self.assertIn("sigma_dp_15m", all_features)
        self.assertNotIn("sigma_dp_15m", ohlcv_features)

    def test_indicator_features_include_requested_timeframes(self) -> None:
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
            timeframes=parse_timeframes("1m,5m,15m"),
        )

        self.assertEqual(list(indicators.index), list(bars.index))
        self.assertIn("vpin_1m", indicators.columns)
        self.assertIn("vpin_5m", indicators.columns)
        self.assertIn("vpin_spread_raw_15m", indicators.columns)
        self.assertIn("spread_cross_top_5m", indicators.columns)

    def test_parse_feature_timeframe_label_extracts_indicator_windows(self) -> None:
        self.assertEqual(parse_feature_timeframe_label("spread_below_bottom_4h"), "4h")
        self.assertEqual(parse_feature_timeframe_label("vpin_spread_15m"), "15m")
        self.assertIsNone(parse_feature_timeframe_label("range_mean_15b"))

    def test_resample_explorer_frame_downsamples_to_feature_timeframe(self) -> None:
        periods = 480
        index = pd.date_range("2026-01-01 09:30", periods=periods, freq="min")
        close = 100.0 + np.arange(periods) * 0.01
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": 1_000.0 + np.arange(periods),
                "spread_below_bottom_4h": np.arange(periods) % 2,
            },
            index=index,
        )

        resampled = resample_explorer_frame(frame, "4h")

        self.assertLess(len(resampled), len(frame))
        self.assertIn("spread_below_bottom_4h", resampled.columns)
        self.assertEqual(resolve_chart_timeframe("spread_below_bottom_4h", 1.0), "4h")

    def test_parse_target_horizon_label_extracts_daily_horizons(self) -> None:
        self.assertEqual(parse_target_horizon_label("fwd_realized_volatility_7d"), "7d")
        self.assertEqual(parse_target_horizon_label("fwd_return_5m"), "5m")
        self.assertIsNone(parse_target_horizon_label("vpin_spread_1m"))

    def test_resolve_log_return_series_uses_horizon_labels(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "volume": [1_000.0, 1_100.0, 1_200.0, 1_300.0, 1_400.0, 1_500.0],
                "log_return_5b": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
            },
            index=pd.date_range("2026-01-01 09:30", periods=6, freq="min"),
        )

        series_key, series = resolve_log_return_series(bars, "5m", bar_minutes=1.0)
        self.assertEqual(series_key, "log_return_5m")
        self.assertEqual(list(series), list(bars["log_return_5b"]))

    def test_plotly_cdn_script_includes_plotly_library(self) -> None:
        script_tags = plotly_cdn_script()
        self.assertIn("cdn.plot.ly", script_tags)
        self.assertIn("PlotlyConfig", script_tags)

    def test_build_explorer_series_bundle_includes_pair_columns(self) -> None:
        periods = 240
        index = pd.date_range("2026-01-01 09:30", periods=periods, freq="min")
        close = 100.0 + np.sin(np.arange(periods) / 10.0)
        bars = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1_000.0 + np.arange(periods),
            },
            index=index,
        )

        analysis_frame, _, _, top_features = run_feature_analysis(
            bars,
            horizon_minutes=[5],
            date="all",
            quantiles=3,
            timeframes=parse_timeframes("1m"),
        )
        pairs = collect_top_pairs(top_features)
        bundle = build_explorer_series_bundle(
            analysis_frame,
            pairs,
            max_rows=100,
            top_features=top_features,
        )

        self.assertGreater(len(pairs), 0)
        view = bundle["views_by_timeframe"]["1m"]
        self.assertIn("close", view["series"])
        self.assertIn("5m", view["log_returns_by_horizon"])
        self.assertIn("log_return_5m", view["series"])
        self.assertIn(pairs[0]["feature"], view["series"])
        self.assertIn(pairs[0]["target"], view["series"])
        self.assertEqual(len(view["timestamps"]), len(view["series"]["close"]))
        self.assertIn("feature_timeframe", pairs[0])

    def test_synthetic_analysis_writes_report_and_csvs(self) -> None:
        periods = 1_120
        index = pd.date_range("2026-01-01 09:30", periods=periods, freq="min")
        trend = np.linspace(100.0, 112.0, periods)
        wave = np.sin(np.arange(periods) / 12.0)
        close = trend + wave
        bars = pd.DataFrame(
            {
                "open": close - 0.05,
                "high": close + 0.25,
                "low": close - 0.25,
                "close": close,
                "volume": 1_000.0 + np.arange(periods) % 50,
            },
            index=index,
        )

        analysis_frame, correlations, quantile_buckets, top_features = run_feature_analysis(
            bars,
            horizon_minutes=[5],
            date="all",
            quantiles=3,
            timeframes=parse_timeframes("1m,5m"),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_dir = root / "csv"
            output_path = root / "report.html"
            write_csv_outputs(
                csv_dir=csv_dir,
                correlations=correlations,
                quantile_buckets=quantile_buckets,
                top_features=top_features,
                feature_frame=analysis_frame,
                sample_rows=25,
            )
            write_html_report(
                output_path=output_path,
                frame=analysis_frame,
                correlations=correlations,
                quantile_buckets=quantile_buckets,
                top_features=top_features,
                top_n=5,
            )

            self.assertTrue((csv_dir / "correlations.csv").exists())
            self.assertTrue((csv_dir / "quantile_buckets.csv").exists())
            self.assertTrue((csv_dir / "top_features.csv").exists())
            self.assertTrue((csv_dir / "feature_frame_sample.csv").exists())
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertFalse(build_top_features(correlations, quantile_buckets).empty)

            report_html = output_path.read_text(encoding="utf-8")
            self.assertIn("pair-explorer-chart", report_html)
            self.assertIn("pair-row", report_html)
            self.assertIn("pair-explorer-data", report_html)
            self.assertIn("All Feature Relationships", report_html)
            self.assertIn("views_by_timeframe", report_html)
            self.assertIn("feature_timeframe", report_html)
            if not top_features.empty:
                self.assertIn(str(top_features.iloc[0]["feature"]), report_html)
                self.assertEqual(report_html.count('class="pair-row"'), len(top_features))


if __name__ == "__main__":
    unittest.main()
