"""
Weekly Walk-Forward Rolling Backtest for 2026.

Design
------
For every ISO calendar week in Config.ROLLING_YEAR (default 2026):

  1. Training cut-off = last trading day before the week starts.
     The LGBMSignal already trains strictly on past data (no look-ahead),
     and Config.RETRAIN_FREQ = 5 means the model updates every trading week.

  2. Test window = up to 5 trading days in the week.
     Capital carries over continuously from one week to the next.

  3. Execution = buy at close T, sell at VWAP_{T+1} (if USE_VWAP_EXIT).

  4. Per-week metrics are computed and reported:
     return, Sharpe, win-rate, avg positions, max drawdown, turnover.

Usage
-----
    rb = WeeklyRollingBacktest(config)
    weekly_df = rb.run(
        panel        = enriched_factor_panel,   # full date range
        ml_scores    = ml_scores_df,            # from LGBMSignal.fit_predict
        strategy     = fitted_strategy,         # CloseToCloseStrategy
        benchmark    = benchmark_returns,       # optional pd.Series
    )
    rb.print_weekly_report(weekly_df)
    rb.save_weekly_report(weekly_df, out_dir)
"""

import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from strategy import CloseToCloseStrategy
from backtest import BacktestEngine, BacktestResult
from performance import PerformanceAnalyzer, TRADING_DAYS_PER_YEAR

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class WeeklyRollingBacktest:
    """
    Walk-forward weekly rolling backtest.

    The backtest runs the entire year continuously so that capital is
    preserved across weeks. Results are then sliced by ISO week for
    per-week reporting.
    """

    def __init__(self, config: Config = None):
        self.cfg = config or Config()

    # ─── Main entry ───────────────────────────────────────────────────────────

    def run(
        self,
        panel:     pd.DataFrame,
        ml_scores: Optional[pd.DataFrame],
        strategy:  CloseToCloseStrategy,
        benchmark: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Run the full-year continuous backtest and slice by week.

        Parameters
        ----------
        panel      : enriched factor panel (MultiIndex date × code)
        ml_scores  : output of LGBMSignal.fit_predict (already walk-forward)
        strategy   : fitted CloseToCloseStrategy
        benchmark  : daily benchmark returns (optional)

        Returns
        -------
        weekly_df  : DataFrame indexed by "YYYY-Www" with per-week metrics
        """
        year = self.cfg.ROLLING_YEAR

        # ── 1. Select the test panel for the rolling year ─────────────────────
        year_panel = panel[
            panel.index.get_level_values("date").year == year
        ]
        if year_panel.empty:
            raise ValueError(
                f"No data found for year {year}. "
                "Check that data has been downloaded and the panel covers this year."
            )

        test_dates = sorted(year_panel.index.get_level_values("date").unique())
        log.info(f"Rolling backtest {year}: {len(test_dates)} trading days "
                 f"({test_dates[0].date()} → {test_dates[-1].date()})")

        # ── 2. Filter ML scores to the test year ─────────────────────────────
        year_ml = None
        if ml_scores is not None:
            year_ml = ml_scores[
                ml_scores.index.get_level_values("date").year == year
            ]

        # ── 3. Generate signals for the entire year ───────────────────────────
        signals = strategy.generate_signals(year_panel, year_ml)
        picks   = strategy.get_all_picks(signals, test_dates)

        nonempty = sum(1 for v in picks.values() if v)
        log.info(f"  Picks: {nonempty}/{len(picks)} non-empty days, "
                 f"avg {np.mean([len(v) for v in picks.values() if v]):.1f} stocks")

        # ── 4. Run the continuous backtest ────────────────────────────────────
        engine = BacktestEngine(
            picks          = picks,
            panel          = year_panel,
            config         = self.cfg,
            initial_capital= self.cfg.INITIAL_CAPITAL,
            use_vwap_exit  = getattr(self.cfg, "USE_VWAP_EXIT", True),
        )
        result = engine.run()
        log.info(f"  {result}")

        # ── 5. Slice by ISO week and compute per-week metrics ─────────────────
        weekly_df = self._weekly_metrics(result, benchmark)

        self._result = result        # store for downstream chart access
        self._benchmark = benchmark

        return weekly_df

    # ─── Slicing & per-week metrics ───────────────────────────────────────────

    def _weekly_metrics(
        self,
        result:    BacktestResult,
        benchmark: Optional[pd.Series],
    ) -> pd.DataFrame:
        """Group the continuous backtest result by ISO week."""
        daily = result.daily.copy()
        daily["iso_year"] = daily.index.isocalendar().year.astype(int)
        daily["iso_week"] = daily.index.isocalendar().week.astype(int)
        daily["week_key"] = daily.apply(
            lambda r: f"{int(r['iso_year'])}-W{int(r['iso_week']):02d}", axis=1
        )

        # Align benchmark
        bm_aligned = None
        if benchmark is not None:
            bm_aligned = benchmark.reindex(daily.index).fillna(0)

        rows = []
        for week_key, wdf in daily.groupby("week_key", sort=True):
            eq    = wdf["portfolio_value"]
            dr    = wdf["daily_return"]
            n     = len(wdf)

            week_ret   = float(eq.iloc[-1] / eq.iloc[0] - 1) if n > 0 else 0.0
            ann_ret    = (1 + week_ret) ** (TRADING_DAYS_PER_YEAR / max(n, 1)) - 1
            vol        = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if n > 1 else 0.0
            sharpe     = float(dr.mean() / dr.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if (n > 1 and dr.std() > 0) else 0.0
            mdd        = float(-(eq / eq.cummax() - 1).min())
            win_days   = int((dr > 0).sum())
            avg_pos    = float(wdf["n_positions"].mean())
            turnover   = float(wdf["turnover"].mean())

            # Benchmark comparison for the week
            bm_ret = np.nan
            if bm_aligned is not None:
                week_bm = bm_aligned.reindex(wdf.index)
                bm_ret  = float((1 + week_bm).prod() - 1)

            rows.append({
                "week":            week_key,
                "start_date":      wdf.index[0].strftime("%Y-%m-%d"),
                "end_date":        wdf.index[-1].strftime("%Y-%m-%d"),
                "n_days":          n,
                "week_return":     week_ret,
                "ann_return":      ann_ret,
                "ann_volatility":  vol,
                "sharpe":          sharpe,
                "max_drawdown":    mdd,
                "win_days":        win_days,
                "avg_positions":   avg_pos,
                "avg_turnover":    turnover,
                "benchmark_return": bm_ret,
                "excess_return":   week_ret - bm_ret if not np.isnan(bm_ret) else np.nan,
                "portfolio_value_end": float(eq.iloc[-1]),
            })

        return pd.DataFrame(rows).set_index("week")

    # ─── Reporting ───────────────────────────────────────────────────────────

    def print_weekly_report(self, weekly_df: pd.DataFrame):
        """Print a colour-coded weekly performance table."""
        total_ret  = (weekly_df["portfolio_value_end"].iloc[-1] /
                      self.cfg.INITIAL_CAPITAL - 1)
        ann_ret    = (1 + total_ret) ** (TRADING_DAYS_PER_YEAR /
                      max(weekly_df["n_days"].sum(), 1)) - 1
        avg_sharpe = weekly_df["sharpe"].mean()
        win_weeks  = (weekly_df["week_return"] > 0).sum()
        total_wks  = len(weekly_df)

        header = (
            f"\n{'='*72}\n"
            f"  {self.cfg.ROLLING_YEAR} 年周度滚动回测结果 — CloseToClose (VWAP标签)\n"
            f"{'='*72}\n"
            f"  总收益率: {total_ret*100:+.2f}%  |  "
            f"年化收益: {ann_ret*100:+.2f}%  |  "
            f"胜周率: {win_weeks}/{total_wks}  |  "
            f"平均Sharpe: {avg_sharpe:.2f}\n"
            f"{'='*72}"
        )
        print(header)

        col_fmt = (
            f"  {'周次':<12} {'开始':>11} {'结束':>11} "
            f"{'周收益':>9} {'夏普':>7} {'胜利天':>6} "
            f"{'平均仓':>7} {'基准':>9} {'超额':>9} {'末值':>14}"
        )
        print(col_fmt)
        print("  " + "-" * 70)

        for week, row in weekly_df.iterrows():
            wr    = row["week_return"] * 100
            sign  = "+" if wr >= 0 else ""
            bm    = f"{row['benchmark_return']*100:+.2f}%" if not np.isnan(row["benchmark_return"]) else "  N/A  "
            exc   = f"{row['excess_return']*100:+.2f}%" if not np.isnan(row["excess_return"])   else "  N/A  "
            print(
                f"  {week:<12} {row['start_date']:>11} {row['end_date']:>11} "
                f"  {sign}{wr:.2f}%  {row['sharpe']:>6.2f}  {int(row['win_days']):>5}d "
                f"  {row['avg_positions']:>5.1f}  {bm:>9}  {exc:>9}  "
                f"{row['portfolio_value_end']:>14,.0f}"
            )

        print(f"{'='*72}\n")

    def save_weekly_report(self, weekly_df: pd.DataFrame, out_dir: str):
        """Save weekly metrics CSV and per-week equity sub-plots."""
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "weekly_rolling.csv")
        weekly_df.to_csv(csv_path)
        log.info(f"Weekly report saved → {csv_path}")

    @property
    def result(self) -> Optional[BacktestResult]:
        return getattr(self, "_result", None)
