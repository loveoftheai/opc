"""
Performance Analyser — computes all metrics visible in the reference image.

Metrics produced
----------------
  回测收益率          total_return
  年化收益率          annual_return
  超额年化收益率       excess_annual_return   (vs benchmark)
  基准年化收益率       benchmark_annual_return
  最大回撤            max_drawdown
  夏普比率            sharpe_ratio
  索提诺比率          sortino_ratio
  日均收益率          avg_daily_return
  每笔收益率          avg_trade_return
  胜率                win_rate
  盈亏比              profit_loss_ratio
  日均交易次数        avg_daily_trades
  最大连续盈利        max_consec_wins  (trading days)
  最大连续亏损        max_consec_losses
  最大单笔盈利        max_single_profit (RMB)
  最大单笔亏损        max_single_loss   (RMB)
  年化波动率          annual_volatility
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRADING_DAYS_PER_YEAR = 252


def _annual_return(cum_ret: float, n_days: int) -> float:
    """Convert cumulative return and holding days to annualised return."""
    if n_days <= 0:
        return 0.0
    return (1 + cum_ret) ** (TRADING_DAYS_PER_YEAR / n_days) - 1


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (as a positive fraction)."""
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    return float(-dd.min())


def _sharpe(daily_rets: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio."""
    excess = daily_rets - risk_free / TRADING_DAYS_PER_YEAR
    if excess.std(ddof=1) < 1e-9:
        return 0.0
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(daily_rets: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio (downside deviation in denominator)."""
    excess   = daily_rets - risk_free / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std(ddof=1) < 1e-9:
        return 0.0
    return float(excess.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_consecutive(win_flags: pd.Series) -> tuple[int, int]:
    """
    Return (max_consecutive_wins, max_consecutive_losses)
    where win_flags is a boolean Series (True = positive day).
    """
    max_wins = max_losses = cur_wins = cur_losses = 0
    for flag in win_flags:
        if flag:
            cur_wins  += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins   = 0
        max_wins   = max(max_wins,   cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


class PerformanceAnalyzer:
    """
    Compute and display the full set of metrics shown in the reference image.

    Parameters
    ----------
    result     : BacktestResult from backtest.py
    benchmark  : pd.Series of daily benchmark returns (index = date)
    risk_free  : annual risk-free rate (default 0.03 = 3 %)
    """

    def __init__(self, result, benchmark: Optional[pd.Series] = None,
                 risk_free: float = 0.03):
        self.result     = result
        self.benchmark  = benchmark
        self.risk_free  = risk_free
        self._metrics: dict = {}

    # ─── Main compute ─────────────────────────────────────────────────────────

    def compute(self) -> dict:
        daily  = self.result.daily
        trades = self.result.trades
        equity = self.result.equity_curve
        dr     = self.result.daily_returns

        n_days = len(dr)
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0

        m: dict = {}
        m["strategy_name"]   = self.result.cfg.STRATEGY_NAME
        m["start_date"]      = equity.index[0].strftime("%Y-%m-%d")
        m["end_date"]        = equity.index[-1].strftime("%Y-%m-%d")
        m["initial_capital"] = self.result.cfg.INITIAL_CAPITAL
        m["final_capital"]   = float(equity.iloc[-1])

        # ── Returns ───────────────────────────────────────────────────────────
        m["total_return"]    = total_return
        m["annual_return"]   = _annual_return(total_return, n_days)
        m["annual_volatility"] = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

        # ── Benchmark ─────────────────────────────────────────────────────────
        if self.benchmark is not None:
            bm = self.benchmark.reindex(equity.index).fillna(0)
            bm_cum    = float((1 + bm).prod() - 1)
            bm_annual = _annual_return(bm_cum, len(bm))
            m["benchmark_annual_return"] = bm_annual
            m["excess_annual_return"]    = m["annual_return"] - bm_annual
        else:
            m["benchmark_annual_return"] = np.nan
            m["excess_annual_return"]    = np.nan

        # ── Risk ──────────────────────────────────────────────────────────────
        m["max_drawdown"]  = _max_drawdown(equity)
        m["sharpe_ratio"]  = _sharpe(dr, self.risk_free)
        m["sortino_ratio"] = _sortino(dr, self.risk_free)

        # ── Daily stats ───────────────────────────────────────────────────────
        m["avg_daily_return"]  = float(dr.mean())
        m["avg_daily_trades"]  = float(daily["n_positions"].mean())

        # ── Trade-level stats ─────────────────────────────────────────────────
        if trades.empty:
            m["avg_trade_return"]   = 0.0
            m["win_rate"]           = 0.0
            m["profit_loss_ratio"]  = 0.0
            m["max_single_profit"]  = 0.0
            m["max_single_loss"]    = 0.0
        else:
            net_rets = trades["net_return"]
            m["avg_trade_return"]  = float(net_rets.mean())
            wins  = trades[trades["net_pnl"] > 0]
            losses = trades[trades["net_pnl"] < 0]
            m["win_rate"] = float(len(wins) / len(trades)) if len(trades) > 0 else 0.0
            avg_win  = wins["net_pnl"].mean()   if len(wins)   > 0 else 0.0
            avg_loss = losses["net_pnl"].abs().mean() if len(losses) > 0 else np.nan
            m["profit_loss_ratio"] = float(avg_win / avg_loss) if (avg_loss and avg_loss > 0) else np.nan
            m["max_single_profit"] = float(trades["net_pnl"].max())
            m["max_single_loss"]   = float(trades["net_pnl"].min())

        # ── Consecutive wins/losses (by trading day) ──────────────────────────
        win_days = dr > 0
        max_w, max_l = _max_consecutive(win_days)
        m["max_consec_wins"]   = max_w
        m["max_consec_losses"] = max_l

        self._metrics = m
        return m

    # ─── Display ─────────────────────────────────────────────────────────────

    def print_report(self, metrics: dict = None):
        m = metrics or self._metrics or self.compute()

        def pct(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v * 100:+.2f}%"

        def fmt(v, decimals=2):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v:,.{decimals}f}"

        lines = [
            "",
            "=" * 52,
            f"  策略名称: {m.get('strategy_name', 'N/A')}",
            f"  回测区间: {m.get('start_date')} 至 {m.get('end_date')}",
            "=" * 52,
            f"  初始资金:     {fmt(m.get('initial_capital'))}",
            f"  最终资金:     {fmt(m.get('final_capital'))}",
            "-" * 52,
            f"  回测收益率:   {pct(m.get('total_return'))}",
            f"  年化收益率:   {pct(m.get('annual_return'))}",
            f"  超额年化收益率:{pct(m.get('excess_annual_return'))}",
            f"  基准年化收益率:{pct(m.get('benchmark_annual_return'))}",
            "-" * 52,
            f"  最大回撤:     {pct(m.get('max_drawdown'))}",   # shown as positive
            f"  夏普比率:     {fmt(m.get('sharpe_ratio'))}",
            f"  索提诺比率:   {fmt(m.get('sortino_ratio'))}",
            f"  年化波动率:   {pct(m.get('annual_volatility'))}",
            "-" * 52,
            f"  日均收益率:   {pct(m.get('avg_daily_return'))}",
            f"  每笔收益率:   {pct(m.get('avg_trade_return'))}",
            f"  胜率:         {fmt(m.get('win_rate', 0) * 100)}%",
            f"  盈亏比:       {fmt(m.get('profit_loss_ratio'))}",
            "-" * 52,
            f"  日均交易次数: {fmt(m.get('avg_daily_trades'))}",
            f"  最大连续盈利: {m.get('max_consec_wins')} 天",
            f"  最大连续亏损: {m.get('max_consec_losses')} 天",
            f"  最大单笔盈利: {fmt(m.get('max_single_profit'))}",
            f"  最大单笔亏损: {fmt(m.get('max_single_loss'))}",
            "=" * 52,
            "",
        ]
        print("\n".join(lines))

    # ─── Export ───────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return metrics as a single-row DataFrame for easy storage."""
        m = self._metrics or self.compute()
        return pd.DataFrame([m])

    def drawdown_series(self) -> pd.Series:
        """Return the drawdown time series (values ≤ 0)."""
        equity = self.result.equity_curve
        rolling_max = equity.cummax()
        return (equity - rolling_max) / rolling_max
