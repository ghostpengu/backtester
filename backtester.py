from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import databento as db
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators import VPINSpreadConfig, VWAPBandsConfig, compute_vpin_spread, compute_vwap_bands
from portfolio import Portfolio, format_performance_debug
from strategies import VPINSpreadStrategy, VWAPSignalStrategy


DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_DATE = "latest"
DEFAULT_RESAMPLE = "none"
DEFAULT_SPREAD_CONFIG = VPINSpreadConfig()
DEFAULT_VWAP_CONFIG = VWAPBandsConfig()
DEFAULT_INSTRUMENT = "SPY"
STRATEGY_VPIN_SPREAD = "vpin-spread"
STRATEGY_VWAP = "vwap"
DEFAULT_STRATEGY = STRATEGY_VPIN_SPREAD


def default_data_path(instrument: str = DEFAULT_INSTRUMENT) -> Path:
    instrument_dir = Path("data") / instrument.upper()
    canonical = instrument_dir / "ohlcv-1m.dbn.zst"
    if canonical.exists():
        return canonical

    matches = sorted(instrument_dir.glob("*.dbn.zst"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        return matches[-1]

    return canonical


DEFAULT_DATA_PATH = default_data_path(DEFAULT_INSTRUMENT)
DEFAULT_OUTPUT_PATH = Path("output") / f"{DEFAULT_INSTRUMENT.lower()}_price.html"
DEFAULT_INITIAL_CASH = 100_000.0
DEFAULT_ENTRY_STEP_EXPOSURE = 0.5
DEFAULT_MAX_ENTRY_EXPOSURE = 3.0
DEFAULT_VWAP_SIGNAL_MODE = "analog"
DEFAULT_VWAP_BASE_EXPOSURE = 0.5
DEFAULT_VWAP_MAX_EXPOSURE = 3.0
DEFAULT_VWAP_ENTRY_SIGMA = 2.0
DEFAULT_VWAP_LOOKBACK = "none"
DEFAULT_CLOSE_AFTER_BARS: int | None = None


def default_output_path(instrument: str, stem: str) -> Path:
    return Path("output") / f"{instrument.lower()}_{stem}.html"


def load_data(path: Path, *, instrument: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    store = db.DBNStore.from_file(path)
    df = store.to_df(
        price_type="float",
        pretty_ts=True,
        map_symbols=False,
        schema="ohlcv-1m",
        tz=DEFAULT_TIMEZONE,
    )
    if instrument is not None:
        df = filter_instrument(df, instrument)
    return df


def filter_instrument(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    symbol_column = next(
        (column for column in ("symbol", "raw_symbol", "instrument") if column in df.columns),
        None,
    )
    if symbol_column is None:
        return df

    filtered = df[df[symbol_column].astype(str).str.upper() == instrument.upper()]
    if filtered.empty:
        raise ValueError(f"No data found for instrument {instrument!r}.")
    return filtered


def get_timestamp_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index

    if "ts_event" in df.columns:
        return pd.DatetimeIndex(pd.to_datetime(df["ts_event"]))

    available_columns = ", ".join(map(str, df.columns))
    raise ValueError(
        "Expected a DatetimeIndex or a 'ts_event' timestamp column. "
        f"Available columns: {available_columns}"
    )


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        available_columns = ", ".join(map(str, df.columns))
        raise ValueError(
            f"Expected columns {sorted(required_columns)}. "
            f"Missing: {sorted(missing_columns)}. Available columns: {available_columns}"
        )

    bars = df.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    bars.index = get_timestamp_index(df)
    return bars.dropna(subset=["close"]).sort_index()


def resample_bars(bars: pd.DataFrame, interval: str) -> pd.DataFrame:
    if interval.lower() == "none":
        return bars

    resampled = bars.resample(interval).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["open", "high", "low", "close"])


def filter_chart_data(
    data: pd.DataFrame,
    *,
    date: str = DEFAULT_DATE,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if start is not None or end is not None:
        filtered = data.loc[start:end]
    elif date.lower() == "all":
        filtered = data
    else:
        target_date = data.index.max().date() if date.lower() == "latest" else pd.to_datetime(date).date()
        filtered = data[data.index.date == target_date]
        if filtered.empty:
            raise ValueError(f"No price data found for {target_date.isoformat()}.")

    if filtered.empty:
        raise ValueError("No price data remains after applying filters.")

    return filtered


def add_history_context(
    bars: pd.DataFrame,
    chart_bars: pd.DataFrame,
    context_bars: int,
) -> pd.DataFrame:
    if chart_bars.index[0] == bars.index[0]:
        return chart_bars

    first_position = int(bars.index.searchsorted(chart_bars.index[0], side="left"))
    last_position = int(bars.index.searchsorted(chart_bars.index[-1], side="right"))
    context_start = max(0, first_position - context_bars)
    return bars.iloc[context_start:last_position]


def prepare_chart_data(
    df: pd.DataFrame,
    *,
    strategy: str = DEFAULT_STRATEGY,
    date: str = DEFAULT_DATE,
    start: str | None = None,
    end: str | None = None,
    resample: str = DEFAULT_RESAMPLE,
    spread_config: VPINSpreadConfig = DEFAULT_SPREAD_CONFIG,
    vwap_config: VWAPBandsConfig = DEFAULT_VWAP_CONFIG,
) -> pd.DataFrame:
    bars = resample_bars(prepare_bars(df), resample)
    chart_bars = filter_chart_data(bars, date=date, start=start, end=end)
    if strategy == STRATEGY_VPIN_SPREAD:
        analysis_bars = add_history_context(bars, chart_bars, spread_config.warmup_bars)
        indicators = compute_vpin_spread(analysis_bars, spread_config)
    elif strategy == STRATEGY_VWAP:
        analysis_bars = add_history_context(bars, chart_bars, vwap_config.warmup_bars)
        indicators = compute_vwap_bands(analysis_bars, vwap_config)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    chart_data = analysis_bars.join(indicators)
    return chart_data.loc[chart_bars.index]


def run_portfolio_backtest(
    chart_data: pd.DataFrame,
    *,
    strategy: str = DEFAULT_STRATEGY,
    instrument: str = DEFAULT_INSTRUMENT,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    entry_step_exposure: float = DEFAULT_ENTRY_STEP_EXPOSURE,
    max_entry_exposure: float = DEFAULT_MAX_ENTRY_EXPOSURE,
    vwap_signal_mode: str = DEFAULT_VWAP_SIGNAL_MODE,
    vwap_base_exposure: float = DEFAULT_VWAP_BASE_EXPOSURE,
    vwap_max_exposure: float = DEFAULT_VWAP_MAX_EXPOSURE,
    close_after_bars: int | None = DEFAULT_CLOSE_AFTER_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio = Portfolio(initial_cash=initial_cash, default_symbol=instrument)
    if strategy == STRATEGY_VPIN_SPREAD:
        strategy_impl = VPINSpreadStrategy(
            symbol=instrument,
            entry_step_exposure=entry_step_exposure,
            max_exposure=max_entry_exposure,
            close_after_bars=close_after_bars,
        )
    elif strategy == STRATEGY_VWAP:
        strategy_impl = VWAPSignalStrategy(
            symbol=instrument,
            signal_mode=vwap_signal_mode,
            base_exposure=vwap_base_exposure,
            max_exposure=vwap_max_exposure,
            close_after_bars=close_after_bars,
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    for timestamp, row in chart_data.iterrows():
        price = float(row["close"])
        orders = strategy_impl.generate_orders(timestamp, row, portfolio)
        for order in orders:
            portfolio.execute_order(order, price)
        portfolio.mark(timestamp, {instrument: price})

    snapshots = portfolio.snapshots_frame()
    fills = portfolio.fills_frame()
    enriched = chart_data.join(snapshots, how="left")
    enriched = add_buy_and_hold_benchmark(enriched, initial_cash=initial_cash)
    return enriched, fills


def add_buy_and_hold_benchmark(chart_data: pd.DataFrame, *, initial_cash: float) -> pd.DataFrame:
    if chart_data.empty:
        return chart_data

    benchmark = chart_data.copy()
    first_close = float(benchmark["close"].iloc[0])
    if first_close <= 0:
        raise ValueError("Cannot compute buy-and-hold benchmark with a non-positive first close.")

    benchmark_quantity = initial_cash / first_close
    benchmark["buy_hold_quantity"] = benchmark_quantity
    benchmark["buy_hold_equity"] = benchmark_quantity * benchmark["close"]
    benchmark["buy_hold_pnl"] = benchmark["buy_hold_equity"] - initial_cash
    benchmark["buy_hold_return_pct"] = benchmark["buy_hold_pnl"] / initial_cash
    return benchmark


def spread_color_masks(chart_data: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    return [
        ("VPIN Spread", "#22d3ee", ~(chart_data["spread_above_top"] | chart_data["spread_below_bottom"])),
        ("Spread Above Top", "#22c55e", chart_data["spread_above_top"]),
        ("Spread Below Bottom", "#ef4444", chart_data["spread_below_bottom"]),
    ]


def add_last_value_table(
    figure: go.Figure,
    chart_data: pd.DataFrame,
    spread_config: VPINSpreadConfig,
) -> None:
    valid = chart_data.dropna(subset=["vpin_spread"])
    if valid.empty:
        return

    last = valid.iloc[-1]
    state = "ABOVE TOP" if bool(last["spread_above_top"]) else "BELOW BOTTOM" if bool(last["spread_below_bottom"]) else "-"
    signal = "CROSS TOP" if bool(last["spread_cross_top"]) else "CROSS BOTTOM" if bool(last["spread_cross_bottom"]) else "-"
    text = (
        f"Spread {last['vpin_spread']:.4f}<br>"
        f"Top {spread_config.top_limit:.2f}<br>"
        f"Bottom {spread_config.bottom_limit:.2f}<br>"
        f"State {state}<br>"
        f"Signal {signal}"
    )
    figure.add_annotation(
        x=0.995,
        y=0.98,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        align="left",
        bordercolor="rgba(120,120,120,0.45)",
        borderwidth=1,
        bgcolor="rgba(0,0,0,0.72)",
        font={"color": "white", "size": 11},
    )


def add_vwap_last_value_table(
    figure: go.Figure,
    chart_data: pd.DataFrame,
    vwap_config: VWAPBandsConfig,
) -> None:
    valid = chart_data.dropna(subset=["vwap", "vwap_zscore"])
    if valid.empty:
        return

    last = valid.iloc[-1]
    text = (
        f"VWAP {last['vwap']:.2f}<br>"
        f"Z {last['vwap_zscore']:.2f}<br>"
        f"Signal {last['vwap_signal']:.2f}<br>"
        f"Strength {last['vwap_signal_strength']:.2f}<br>"
        f"Sigma {vwap_config.signal_sigma:.2f}<br>"
        f"Lookback {vwap_config.lookback}"
    )
    figure.add_annotation(
        x=0.995,
        y=0.98,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        align="left",
        bordercolor="rgba(120,120,120,0.45)",
        borderwidth=1,
        bgcolor="rgba(0,0,0,0.72)",
        font={"color": "white", "size": 11},
    )


def add_trade_markers(figure: go.Figure, fills: pd.DataFrame) -> None:
    if fills.empty:
        return

    buy_fills = fills[fills["quantity"] > 0]
    sell_fills = fills[fills["quantity"] < 0]

    if not buy_fills.empty:
        figure.add_trace(
            go.Scatter(
                x=buy_fills.index,
                y=buy_fills["price"],
                mode="markers",
                name="Buy / Increase",
                marker={"color": "#16a34a", "size": 11, "symbol": "triangle-up"},
            ),
            row=1,
            col=1,
        )

    if not sell_fills.empty:
        figure.add_trace(
            go.Scatter(
                x=sell_fills.index,
                y=sell_fills["price"],
                mode="markers",
                name="Sell / Reduce",
                marker={"color": "#dc2626", "size": 11, "symbol": "triangle-down"},
            ),
            row=1,
            col=1,
        )


def add_portfolio_table(figure: go.Figure, chart_data: pd.DataFrame) -> None:
    valid = chart_data.dropna(subset=["equity"])
    if valid.empty:
        return

    last = valid.iloc[-1]
    text = (
        f"Equity ${last['equity']:,.2f}<br>"
        f"Total PnL ${last['total_pnl']:,.2f}<br>"
        f"Buy/Hold ${last.get('buy_hold_pnl', float('nan')):,.2f}<br>"
        f"Realized ${last['realized_pnl']:,.2f}<br>"
        f"Unrealized ${last['unrealized_pnl']:,.2f}<br>"
        f"Net Exp {last['net_exposure_pct'] * 100.0:.1f}%<br>"
        f"Delta Exp {last['delta_exposure_pct'] * 100.0:.1f}%<br>"
        f"Qty {last['position_quantity']:.2f}"
    )
    figure.add_annotation(
        x=0.995,
        y=0.27,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        align="left",
        bordercolor="rgba(120,120,120,0.45)",
        borderwidth=1,
        bgcolor="rgba(0,0,0,0.72)",
        font={"color": "white", "size": 11},
    )


def padded_range(values: pd.Series, *, include_zero: bool = True) -> list[float] | None:
    clean = values.dropna()
    if clean.empty:
        return None

    low = float(clean.min())
    high = float(clean.max())
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)

    if low == high:
        padding = max(abs(low) * 0.05, 1.0)
    else:
        padding = (high - low) * 0.12

    return [low - padding, high + padding]


def sigma_column_label(level: float) -> str:
    if float(level).is_integer():
        return f"{int(level)}sigma"
    return f"{str(level).replace('.', 'p')}sigma"


def plot_backtest(
    chart_data: pd.DataFrame,
    fills: pd.DataFrame,
    output_path: Path,
    *,
    strategy: str = DEFAULT_STRATEGY,
    instrument: str = DEFAULT_INSTRUMENT,
    spread_config: VPINSpreadConfig = DEFAULT_SPREAD_CONFIG,
    vwap_config: VWAPBandsConfig = DEFAULT_VWAP_CONFIG,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_dates = pd.Index(chart_data.index.date).unique()
    indicator_title = "VWAP Signal" if strategy == STRATEGY_VWAP else "VPIN Spread"
    title = f"{instrument} Backtest with {indicator_title} and Portfolio"
    if len(chart_dates) == 1:
        title = f"{title} - {chart_dates[0].isoformat()}"

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.50, 0.28, 0.22],
        subplot_titles=(f"{instrument} Price", indicator_title, "Portfolio"),
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["close"],
            mode="lines",
            name=f"{instrument} close",
            line={"width": 1.4, "color": "#1f77b4"},
        ),
        row=1,
        col=1,
    )
    add_trade_markers(figure, fills)

    if strategy == STRATEGY_VWAP:
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["vwap"],
                mode="lines",
                name="VWAP",
                line={"width": 1.4, "color": "#0f766e"},
            ),
            row=1,
            col=1,
        )
        sigma_label = sigma_column_label(vwap_config.signal_sigma)
        upper_column = f"vwap_upper_{sigma_label}"
        lower_column = f"vwap_lower_{sigma_label}"
        if upper_column in chart_data.columns and lower_column in chart_data.columns:
            figure.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data[upper_column],
                    mode="lines",
                    name=f"VWAP +{vwap_config.signal_sigma:g} sigma",
                    line={"width": 1.0, "color": "#dc2626", "dash": "dash"},
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data[lower_column],
                    mode="lines",
                    name=f"VWAP -{vwap_config.signal_sigma:g} sigma",
                    line={"width": 1.0, "color": "#16a34a", "dash": "dash"},
                ),
                row=1,
                col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["vwap_signal"],
                mode="lines",
                name="VWAP Signal",
                line={"width": 1.8, "color": "#2563eb"},
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["vwap_signal_strength"],
                mode="lines",
                name="Signal Strength",
                line={"width": 1.2, "color": "#f97316", "dash": "dot"},
            ),
            row=2,
            col=1,
        )
        figure.add_hline(y=0, line_dash="dash", line_color="rgba(80,80,80,0.45)", row=2, col=1)
        add_vwap_last_value_table(figure, chart_data, vwap_config)
    else:
        for trace_name, color, mask in spread_color_masks(chart_data):
            figure.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data["vpin_spread"].where(mask),
                    mode="lines",
                    name=trace_name,
                    line={"width": 2.4, "color": color},
                    connectgaps=False,
                    showlegend=trace_name == "VPIN Spread",
                ),
                row=2,
                col=1,
            )

        figure.add_hline(y=0, line_dash="dash", line_color="rgba(80,80,80,0.45)", row=2, col=1)
        figure.add_hline(
            y=spread_config.top_limit,
            line_dash="dash",
            line_color="rgba(100,116,139,0.75)",
            annotation_text="Spread Top",
            annotation_position="top right",
            row=2,
            col=1,
        )
        figure.add_hline(
            y=spread_config.bottom_limit,
            line_dash="dash",
            line_color="rgba(100,116,139,0.75)",
            annotation_text="Spread Bottom",
            annotation_position="bottom right",
            row=2,
            col=1,
        )
        add_last_value_table(figure, chart_data, spread_config)
    add_portfolio_table(figure, chart_data)

    if "equity" in chart_data.columns:
        pnl_columns = chart_data[["total_pnl", "realized_pnl", "unrealized_pnl", "buy_hold_pnl"]]
        pnl_range = padded_range(pnl_columns.stack())
        exposure_range = padded_range(
            pd.concat(
                [
                    chart_data["net_exposure_pct"] * 100.0,
                    chart_data["delta_exposure_pct"] * 100.0,
                ]
            )
        )
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["total_pnl"],
                mode="lines",
                name="Equity Change",
                line={"width": 1.8, "color": "#2563eb"},
                customdata=chart_data["equity"],
                hovertemplate="Equity Change: $%{y:,.2f}<br>Equity: $%{customdata:,.2f}<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["realized_pnl"],
                mode="lines",
                name="Realized PnL",
                line={"width": 1.2, "color": "#059669", "dash": "dot"},
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["unrealized_pnl"],
                mode="lines",
                name="Unrealized PnL",
                line={"width": 1.2, "color": "#f59e0b", "dash": "dash"},
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["buy_hold_pnl"],
                mode="lines",
                name="Buy & Hold PnL",
                line={"width": 1.6, "color": "#64748b"},
                customdata=chart_data["buy_hold_equity"],
                hovertemplate="Buy & Hold PnL: $%{y:,.2f}<br>Buy & Hold Equity: $%{customdata:,.2f}<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["net_exposure_pct"] * 100.0,
                mode="lines",
                name="Net Exposure %",
                line={"width": 1.2, "color": "#9333ea", "dash": "dot"},
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        figure.update_yaxes(range=pnl_range, row=3, col=1, secondary_y=False)
        figure.update_yaxes(range=exposure_range, row=3, col=1, secondary_y=True)
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["delta_exposure_pct"] * 100.0,
                mode="lines",
                name="Delta Exposure %",
                line={"width": 1.2, "color": "#ea580c", "dash": "dash"},
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

    figure.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 60, "r": 35, "t": 95, "b": 50},
    )
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="VWAP Signal" if strategy == STRATEGY_VWAP else "Spread", row=2, col=1)
    figure.update_yaxes(title_text="Equity Change / PnL ($)", row=3, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Exposure %", row=3, col=1, secondary_y=True)
    figure.update_xaxes(title_text="Time", row=3, col=1)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def open_chart(output_path: Path) -> bool:
    return webbrowser.open(output_path.resolve().as_uri())


def build_vwap_config(args: argparse.Namespace) -> VWAPBandsConfig:
    return VWAPBandsConfig(
        sigma_levels=unique_sigma_levels(2.0, 3.0, args.vwap_entry_sigma),
        lookback=args.vwap_lookback,
        signal_sigma=args.vwap_entry_sigma,
    )


def unique_sigma_levels(*levels: float) -> tuple[float, ...]:
    unique: list[float] = []
    for level in levels:
        normalized = float(level)
        if all(abs(normalized - existing) > 1e-12 for existing in unique):
            unique.append(normalized)
    return tuple(unique)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load local Databento OHLCV data and plot a strategy backtest.",
    )
    parser.add_argument(
        "--instrument",
        default=DEFAULT_INSTRUMENT,
        help=f"Instrument symbol to load and backtest. Default: {DEFAULT_INSTRUMENT}.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=f"Path to the Databento DBN file. Default: data/<instrument>/ohlcv-1m.dbn.zst",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Path for the output HTML chart. Default: output/<instrument>_price.html ({DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--start",
        help="Optional start date/time filter. Overrides --date when used.",
    )
    parser.add_argument(
        "--end",
        help="Optional end date/time filter. Overrides --date when used.",
    )
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="Date to plot: 'latest', 'all', or YYYY-MM-DD. Default: latest.",
    )
    parser.add_argument(
        "--resample",
        default=DEFAULT_RESAMPLE,
        help="Pandas resample interval, or 'none' for raw 1-minute bars. Default: none.",
    )
    parser.add_argument(
        "--strategy",
        choices=[STRATEGY_VPIN_SPREAD, STRATEGY_VWAP],
        default=DEFAULT_STRATEGY,
        help=f"Strategy to run. Default: {DEFAULT_STRATEGY}.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Write the chart without opening it in the default browser.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=DEFAULT_INITIAL_CASH,
        help=f"Starting cash for the portfolio tracker. Default: {DEFAULT_INITIAL_CASH:,.0f}.",
    )
    parser.add_argument(
        "--entry-step-exposure",
        "--entry-exposure",
        dest="entry_step_exposure",
        type=float,
        default=DEFAULT_ENTRY_STEP_EXPOSURE,
        help=(
            "Exposure added on each VPIN spread top-limit crossover. Used by --strategy vpin-spread. "
            f"Default: {DEFAULT_ENTRY_STEP_EXPOSURE:.2f}."
        ),
    )
    parser.add_argument(
        "--max-entry-exposure",
        type=float,
        default=DEFAULT_MAX_ENTRY_EXPOSURE,
        help=(
            "Maximum target exposure for VPIN spread scale-ins. Used by --strategy vpin-spread. "
            f"Default: {DEFAULT_MAX_ENTRY_EXPOSURE:.1f}."
        ),
    )
    parser.add_argument(
        "--vwap-lookback",
        choices=["none", "1w", "1y"],
        default=DEFAULT_VWAP_LOOKBACK,
        help="VWAP basis: session-reset none, rolling one week, or rolling one year. Default: none.",
    )
    parser.add_argument(
        "--vwap-signal-mode",
        choices=["analog", "binary"],
        default=DEFAULT_VWAP_SIGNAL_MODE,
        help="VWAP signal model. Analog scales with z-score distance; binary uses fixed exposure. Default: analog.",
    )
    parser.add_argument(
        "--vwap-entry-sigma",
        type=float,
        default=DEFAULT_VWAP_ENTRY_SIGMA,
        help=f"VWAP sigma threshold that activates the signal. Default: {DEFAULT_VWAP_ENTRY_SIGMA:.1f}.",
    )
    parser.add_argument(
        "--vwap-base-exposure",
        type=float,
        default=DEFAULT_VWAP_BASE_EXPOSURE,
        help=f"Base target exposure for VWAP signals. Default: {DEFAULT_VWAP_BASE_EXPOSURE:.2f}.",
    )
    parser.add_argument(
        "--vwap-max-exposure",
        type=float,
        default=DEFAULT_VWAP_MAX_EXPOSURE,
        help=f"Maximum absolute target exposure for VWAP signals. Default: {DEFAULT_VWAP_MAX_EXPOSURE:.1f}.",
    )
    parser.add_argument(
        "--close-after-bars",
        type=int,
        default=DEFAULT_CLOSE_AFTER_BARS,
        help="Flatten an open strategy position after this many bars. Default: disabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data or default_data_path(args.instrument)
    output_path = args.output or default_output_path(args.instrument, "price")

    try:
        vwap_config = build_vwap_config(args)
        df = load_data(data_path, instrument=args.instrument)
        chart_data = prepare_chart_data(
            df,
            strategy=args.strategy,
            date=args.date,
            start=args.start,
            end=args.end,
            resample=args.resample,
            vwap_config=vwap_config,
        )
        chart_data, fills = run_portfolio_backtest(
            chart_data,
            strategy=args.strategy,
            instrument=args.instrument,
            initial_cash=args.initial_cash,
            entry_step_exposure=args.entry_step_exposure,
            max_entry_exposure=args.max_entry_exposure,
            vwap_signal_mode=args.vwap_signal_mode,
            vwap_base_exposure=args.vwap_base_exposure,
            vwap_max_exposure=args.vwap_max_exposure,
            close_after_bars=args.close_after_bars,
        )
        plot_backtest(
            chart_data,
            fills,
            output_path,
            strategy=args.strategy,
            instrument=args.instrument,
            vwap_config=vwap_config,
        )
        opened = False if args.no_open else open_chart(output_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df):,} {args.instrument} rows from {data_path}")
    print(
        "Plotted "
        f"{len(chart_data):,} points from {chart_data.index.min()} "
        f"to {chart_data.index.max()} to {output_path}"
    )
    print(f"Executed {len(fills):,} fills")
    print(f"Strategy: {args.strategy}")
    if args.strategy == STRATEGY_VWAP:
        print(
            "VWAP: "
            f"lookback={args.vwap_lookback}, "
            f"mode={args.vwap_signal_mode}, "
            f"entry_sigma={args.vwap_entry_sigma:g}, "
            f"base_exposure={args.vwap_base_exposure:g}, "
            f"max_exposure={args.vwap_max_exposure:g}"
        )
    print(format_performance_debug(chart_data, fills))
    if not args.no_open:
        print("Opened chart in default browser" if opened else "Chart written, but browser did not open")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
