from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PerformanceMetrics:
    label: str
    bars: int
    trading_days: int
    initial_equity: float
    final_equity: float
    total_pnl: float
    total_return: float
    cagr: float | None
    sharpe: float | None
    annualized_volatility: float | None
    max_drawdown: float
    max_drawdown_pct: float


def compute_performance_metrics(equity: pd.Series, label: str) -> PerformanceMetrics:
    clean = equity.dropna().astype(float)
    if len(clean) < 2:
        raise ValueError(f"{label} metrics require at least two equity points.")

    returns = clean.pct_change().dropna()
    trading_days = pd.Index(clean.index.date).nunique() if isinstance(clean.index, pd.DatetimeIndex) else 0
    periods_per_year = _infer_periods_per_year(clean)

    return PerformanceMetrics(
        label=label,
        bars=len(clean),
        trading_days=int(trading_days),
        initial_equity=float(clean.iloc[0]),
        final_equity=float(clean.iloc[-1]),
        total_pnl=float(clean.iloc[-1] - clean.iloc[0]),
        total_return=float(clean.iloc[-1] / clean.iloc[0] - 1.0),
        cagr=_compute_cagr(clean, trading_days),
        sharpe=_compute_sharpe(returns, periods_per_year),
        annualized_volatility=_compute_annualized_volatility(returns, periods_per_year),
        max_drawdown=_compute_max_drawdown(clean),
        max_drawdown_pct=_compute_max_drawdown_pct(clean),
    )


def format_performance_debug(
    chart_data: pd.DataFrame,
    fills: pd.DataFrame,
) -> str:
    strategy = compute_performance_metrics(chart_data["equity"], "Strategy")
    buy_hold = compute_performance_metrics(chart_data["buy_hold_equity"], "Buy & Hold")

    max_delta = chart_data["delta_exposure_pct"].abs().max()
    avg_delta = chart_data["delta_exposure_pct"].abs().mean()
    final_delta = chart_data["delta_exposure_pct"].iloc[-1]
    max_net = chart_data["net_exposure_pct"].abs().max()
    final_equity = chart_data["equity"].iloc[-1]

    realized = chart_data["realized_pnl"].iloc[-1]
    unrealized = chart_data["unrealized_pnl"].iloc[-1]
    fill_count = len(fills)
    entry_count = int((fills["quantity"] > 0).sum()) if not fills.empty else 0
    exit_count = int((fills["quantity"] < 0).sum()) if not fills.empty else 0
    gross_notional = float(fills["notional"].abs().sum()) if not fills.empty else 0.0

    return "\n".join(
        [
            "",
            "Performance Debug",
            f"  {_format_metrics(strategy)}",
            f"  {_format_metrics(buy_hold)}",
            "Risk / Exposure",
            f"  Max delta exposure: {_format_percent(max_delta)} | Avg abs delta: {_format_percent(avg_delta)} | Final delta: {_format_percent(final_delta)}",
            f"  Max net exposure: {_format_percent(max_net)} | Final equity: {_format_money(final_equity)}",
            "Trades / PnL",
            f"  Fills: {fill_count:,} | Entries: {entry_count:,} | Exits: {exit_count:,} | Gross notional: {_format_money(gross_notional)}",
            f"  Realized PnL: {_format_money(realized)} | Unrealized PnL: {_format_money(unrealized)}",
        ]
    )


def _format_metrics(metrics: PerformanceMetrics) -> str:
    return (
        f"{metrics.label}: "
        f"PnL {_format_money(metrics.total_pnl)} | "
        f"Return {_format_percent(metrics.total_return)} | "
        f"CAGR {_format_optional_percent(metrics.cagr)} | "
        f"Sharpe {_format_optional_number(metrics.sharpe)} | "
        f"Ann Vol {_format_optional_percent(metrics.annualized_volatility)} | "
        f"Max DD {_format_money(metrics.max_drawdown)} ({_format_percent(metrics.max_drawdown_pct)}) | "
        f"Days {metrics.trading_days:,}"
    )


def _compute_cagr(equity: pd.Series, trading_days: int) -> float | None:
    if trading_days < 2:
        return None

    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    if initial <= 0 or final <= 0:
        return None

    years = trading_days / TRADING_DAYS_PER_YEAR
    return (final / initial) ** (1.0 / years) - 1.0


def _compute_sharpe(returns: pd.Series, periods_per_year: float | None) -> float | None:
    if periods_per_year is None or returns.empty:
        return None

    volatility = float(returns.std(ddof=0))
    if volatility == 0.0:
        return None

    return float(returns.mean() / volatility * sqrt(periods_per_year))


def _compute_annualized_volatility(returns: pd.Series, periods_per_year: float | None) -> float | None:
    if periods_per_year is None or returns.empty:
        return None

    return float(returns.std(ddof=0) * sqrt(periods_per_year))


def _compute_max_drawdown(equity: pd.Series) -> float:
    drawdowns = equity - equity.cummax()
    return float(drawdowns.min())


def _compute_max_drawdown_pct(equity: pd.Series) -> float:
    drawdown_pct = equity / equity.cummax() - 1.0
    return float(drawdown_pct.min())


def _infer_periods_per_year(equity: pd.Series) -> float | None:
    if not isinstance(equity.index, pd.DatetimeIndex):
        return None

    counts_by_day = equity.groupby(equity.index.date).size()
    if counts_by_day.empty:
        return None

    return float(counts_by_day.median() * TRADING_DAYS_PER_YEAR)


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _format_percent(value: float) -> str:
    return f"{value * 100.0:,.2f}%"


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return _format_percent(value)


def _format_optional_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"
