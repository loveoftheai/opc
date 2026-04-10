"""
CloseToClose Quantitative Trading System — Main Entry Point

Pipeline
--------
  1. Download A-share daily data (baostock, cached locally)
  2. Compute multi-factor panel (incl. VWAP + 盘口 features)
  3. Train rolling LightGBM signal (target = VWAP return)
  4. Combine IC-IR factor score + ML signal
  5a. Standard backtest (Dec 2025 – Mar 2026)   OR
  5b. Weekly rolling backtest for 2026 (--rolling-2026)
  6. Print performance report + weekly table
  7. Save charts (dashboard, weekly bars, factor IC, return dist)

Quick start
-----------
  python main.py                   # full pipeline (downloads data on first run)
  python main.py --skip-dl         # skip baostock download (use cache)
  python main.py --no-ml           # factor-only mode (no LightGBM)
  python main.py --rolling-2026    # weekly walk-forward for 2026
  python main.py --skip-dl --rolling-2026   # most common after first run
"""

import argparse
import logging
import os
import sys
import warnings

import pandas as pd

from config import Config
from data_fetcher import DataFetcher
from factors import FactorEngine
from ml_model import LGBMSignal
from strategy import CloseToCloseStrategy
from backtest import BacktestEngine
from performance import PerformanceAnalyzer
from rolling_backtest import WeeklyRollingBacktest
from visualize import (plot_dashboard, plot_factor_ic,
                       plot_return_distribution, plot_weekly_returns)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/run.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CloseToClose A-share quant strategy")
    p.add_argument("--skip-dl",       action="store_true",
                   help="Skip baostock download; use existing cache")
    p.add_argument("--no-ml",         action="store_true",
                   help="Disable LightGBM signal (factor-only)")
    p.add_argument("--n-stocks",      type=int,   default=None,
                   help="Override number of daily positions")
    p.add_argument("--test-start",    type=str,   default=None,
                   help="Override backtest start date (YYYY-MM-DD)")
    p.add_argument("--test-end",      type=str,   default=None,
                   help="Override backtest end date (YYYY-MM-DD)")
    p.add_argument("--train-only",    action="store_true",
                   help="Only train/compute signals; do not run backtest")
    p.add_argument("--rolling-2026",  action="store_true",
                   help="Run weekly walk-forward rolling backtest for 2026")
    p.add_argument("--rolling-year",  type=int,   default=None,
                   help="Override rolling backtest year (default 2026)")
    return p.parse_args()


# ─── Pipeline steps ───────────────────────────────────────────────────────────

def step_download(cfg: Config, skip: bool):
    """Step 1: Download (or load from cache) daily OHLCV data."""
    fetcher = DataFetcher(cfg)
    if not skip:
        log.info("=" * 60)
        log.info("STEP 1/6 — Downloading A-share data via baostock")
        log.info("=" * 60)
        with fetcher:
            fetcher.download_universe()
    else:
        log.info("STEP 1/6 — Skipping download (--skip-dl)")
    return fetcher


def step_load_panel(fetcher: DataFetcher, cfg: Config) -> pd.DataFrame:
    """Step 2: Load and filter the full panel."""
    log.info("=" * 60)
    log.info("STEP 2/6 — Loading panel data")
    log.info("=" * 60)
    with fetcher:
        panel = fetcher.load_panel_data(
            start=cfg.DATA_START,
            end=cfg.TEST_END,
        )
        benchmark = fetcher.load_benchmark(
            start=cfg.DATA_START,
            end=cfg.TEST_END,
        )
    return panel, benchmark


def step_compute_factors(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Step 3: Compute all alpha factors + target variable."""
    log.info("=" * 60)
    log.info("STEP 3/6 — Computing factors")
    log.info("=" * 60)
    fe = FactorEngine(cfg)
    enriched = fe.compute(panel)
    log.info(f"  Factors computed. Panel shape: {enriched.shape}")

    # Save factor panel for inspection
    panel_path = os.path.join(cfg.DATA_DIR, "factor_panel.parquet")
    enriched.to_parquet(panel_path)
    log.info(f"  Factor panel saved → {panel_path}")
    return enriched


def step_ml_signal(
    enriched: pd.DataFrame,
    cfg: Config,
    use_ml: bool,
) -> pd.DataFrame | None:
    """Step 4: Train rolling LightGBM and generate ML scores."""
    if not use_ml:
        log.info("STEP 4/6 — ML signal disabled (--no-ml)")
        return None

    log.info("=" * 60)
    log.info("STEP 4/6 — Training rolling LightGBM signal")
    log.info("=" * 60)
    ml = LGBMSignal(cfg)
    ml_scores = ml.fit_predict(enriched)

    # Save model
    model_path = os.path.join(cfg.OUTPUT_DIR, "lgbm_model.pkl")
    ml.save(model_path)

    # Feature importance
    try:
        imp = ml.feature_importance()
        log.info(f"  Top-10 features:\n{imp.head(10).to_string()}")
    except Exception:
        pass

    return ml_scores


def step_backtest(
    enriched: pd.DataFrame,
    ml_scores: pd.DataFrame | None,
    benchmark: pd.Series,
    cfg: Config,
) -> tuple:
    """Step 5: Strategy signal → picks → backtest."""
    log.info("=" * 60)
    log.info("STEP 5/6 — Running backtest")
    log.info("=" * 60)

    # Split train / test
    train_panel = enriched[
        enriched.index.get_level_values("date") <= pd.Timestamp(cfg.TRAIN_END)
    ]
    test_panel = enriched[
        enriched.index.get_level_values("date") >= pd.Timestamp(cfg.TEST_START)
    ]
    log.info(f"  Train: {len(train_panel):,} rows, "
             f"Test: {len(test_panel):,} rows")

    # Strategy
    strat = CloseToCloseStrategy(cfg)
    strat.fit(train_panel)

    # Factor IC report
    fe = FactorEngine(cfg)
    ic_df = fe.ic_summary(train_panel)
    ic_path = os.path.join(cfg.OUTPUT_DIR, "factor_ic.csv")
    ic_df.to_csv(ic_path)
    log.info(f"  IC summary saved → {ic_path}")

    # Generate full signal table (train + test combined for ML score alignment)
    test_ml = None
    if ml_scores is not None:
        test_ml_idx = ml_scores.index.get_level_values("date") >= pd.Timestamp(cfg.TEST_START)
        test_ml = ml_scores[test_ml_idx]

    signals = strat.generate_signals(test_panel, test_ml)

    # Trading dates in test window
    test_dates = sorted(
        signals.index.get_level_values("date").unique().tolist()
    )
    log.info(f"  Test trading dates: {len(test_dates)} days")

    # Daily picks
    picks = strat.get_all_picks(signals, test_dates, cfg.N_POSITIONS)
    non_empty = sum(1 for v in picks.values() if v)
    log.info(f"  Non-empty pick days: {non_empty} / {len(picks)}")

    # Backtest
    engine = BacktestEngine(picks, test_panel, cfg)
    result = engine.run()
    log.info(f"  {result}")
    return result, signals, ic_df


def step_rolling_backtest(
    enriched:  pd.DataFrame,
    ml_scores: pd.DataFrame | None,
    benchmark: pd.Series,
    cfg:       Config,
) -> tuple:
    """Step 5b: Weekly walk-forward rolling backtest for ROLLING_YEAR."""
    log.info("=" * 60)
    log.info(f"STEP 5/6 — Weekly rolling backtest ({cfg.ROLLING_YEAR})")
    log.info("=" * 60)

    # Fit strategy on data before the rolling year
    train_cutoff = pd.Timestamp(f"{cfg.ROLLING_YEAR - 1}-12-31")
    train_panel  = enriched[
        enriched.index.get_level_values("date") <= train_cutoff
    ]
    fe    = FactorEngine(cfg)
    strat = CloseToCloseStrategy(cfg)
    strat.fit(train_panel)

    ic_df = fe.ic_summary(train_panel)
    ic_path = os.path.join(cfg.OUTPUT_DIR, "factor_ic.csv")
    ic_df.to_csv(ic_path)
    log.info(f"  IC summary saved → {ic_path}")

    # Run rolling backtest
    rb      = WeeklyRollingBacktest(cfg)
    bm_year = (benchmark[benchmark.index.year == cfg.ROLLING_YEAR]
               if benchmark is not None else None)
    weekly_df = rb.run(enriched, ml_scores, strat, benchmark=bm_year)

    rb.print_weekly_report(weekly_df)

    # Save weekly report CSV
    rb.save_weekly_report(weekly_df, cfg.OUTPUT_DIR)

    # Weekly chart
    wk_path = os.path.join(cfg.OUTPUT_DIR, "weekly_rolling.png")
    plot_weekly_returns(weekly_df, rb.result, bm_year, save_path=wk_path)

    return rb.result, None, ic_df


def step_report(result, benchmark, cfg, ic_df, rolling_mode: bool = False):
    """Step 6: Performance report + charts."""
    log.info("=" * 60)
    log.info("STEP 6/6 — Performance report")
    log.info("=" * 60)

    if result is None:
        log.info("  (Rolling mode: detailed report already printed above)")
        return {}

    bm_test = None
    if benchmark is not None:
        start_ts = pd.Timestamp(f"{cfg.ROLLING_YEAR}-01-01" if rolling_mode
                                else cfg.TEST_START)
        bm_test  = benchmark[benchmark.index >= start_ts]

    analyser = PerformanceAnalyzer(result, bm_test, risk_free=0.03)
    metrics  = analyser.compute()
    analyser.print_report(metrics)

    # Save metrics CSV
    metrics_path = os.path.join(cfg.OUTPUT_DIR, "metrics.csv")
    analyser.to_dataframe().to_csv(metrics_path, index=False)
    log.info(f"  Metrics saved → {metrics_path}")

    # Save drawdown series
    dd_path = os.path.join(cfg.OUTPUT_DIR, "drawdown.csv")
    analyser.drawdown_series().to_csv(dd_path, header=["drawdown"])
    log.info(f"  Drawdown saved → {dd_path}")

    # Dashboard chart
    dash_path = os.path.join(cfg.OUTPUT_DIR, "dashboard.png")
    plot_dashboard(result, bm_test, metrics, save_path=dash_path)

    # Return distribution
    dist_path = os.path.join(cfg.OUTPUT_DIR, "return_dist.png")
    plot_return_distribution(result.daily_returns, save_path=dist_path)

    # Factor IC chart
    ic_path = os.path.join(cfg.OUTPUT_DIR, "factor_ic.png")
    plot_factor_ic(ic_df, save_path=ic_path)

    log.info("  All outputs written to ./output/")
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = Config()

    # Apply CLI overrides
    if args.n_stocks:
        cfg.N_POSITIONS = args.n_stocks
    if args.test_start:
        cfg.TEST_START = args.test_start
    if args.test_end:
        cfg.TEST_END = args.test_end
    if args.rolling_year:
        cfg.ROLLING_YEAR = args.rolling_year

    rolling_mode = args.rolling_2026 or (args.rolling_year is not None)
    if rolling_mode:
        # Extend test end to cover full rolling year if not already set
        year = cfg.ROLLING_YEAR
        cfg.TEST_END = f"{year}-12-31"

    cfg.ensure_dirs()
    os.makedirs("output", exist_ok=True)

    log.info("CloseToClose Quant System — Starting")
    log.info(f"  Initial capital : ¥{cfg.INITIAL_CAPITAL:,}")
    log.info(f"  Backtest period : {cfg.TEST_START} → {cfg.TEST_END}")
    log.info(f"  N positions     : {cfg.N_POSITIONS}")
    log.info(f"  ML target       : {cfg.TARGET_COL}")
    log.info(f"  VWAP exit       : {cfg.USE_VWAP_EXIT}")
    log.info(f"  ML signal       : {'disabled' if args.no_ml else 'enabled'}")
    log.info(f"  Rolling mode    : {'YES — ' + str(cfg.ROLLING_YEAR) if rolling_mode else 'no'}")

    # ── Step 1: Download ──────────────────────────────────────────────────────
    fetcher = step_download(cfg, skip=args.skip_dl)

    # ── Step 2: Load panel ────────────────────────────────────────────────────
    panel, benchmark = step_load_panel(fetcher, cfg)

    # ── Step 3: Factors ───────────────────────────────────────────────────────
    enriched = step_compute_factors(panel, cfg)

    if args.train_only:
        log.info("--train-only flag set; stopping after factor computation.")
        return

    # ── Step 4: ML signal ─────────────────────────────────────────────────────
    ml_scores = step_ml_signal(enriched, cfg, use_ml=not args.no_ml)

    # ── Step 5a/5b: Backtest or Rolling ───────────────────────────────────────
    if rolling_mode:
        result, signals, ic_df = step_rolling_backtest(
            enriched, ml_scores, benchmark, cfg
        )
    else:
        result, signals, ic_df = step_backtest(enriched, ml_scores, benchmark, cfg)

    # ── Step 6: Report ────────────────────────────────────────────────────────
    metrics = step_report(result, benchmark, cfg, ic_df,
                          rolling_mode=rolling_mode)

    log.info("Done.")
    return result, metrics


if __name__ == "__main__":
    main()
