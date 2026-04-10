"""
Quantitative Trading System Configuration
Strategy: CloseToClose (A-share market, close-price daily rotation)
"""

import os


class Config:
    # ─── Time periods ────────────────────────────────────────────────────────
    DATA_START    = "2022-01-01"   # earliest data to download
    TRAIN_START   = "2022-06-01"   # first date used for training (warm-up needed)
    TRAIN_END     = "2025-12-30"   # last training date (inclusive)
    TEST_START    = "2025-12-31"   # backtest start (matches image)
    TEST_END      = "2026-03-31"   # backtest end   (matches image)

    # ─── Universe ────────────────────────────────────────────────────────────
    MIN_PRICE         = 2.0        # exclude sub-penny stocks (RMB)
    MIN_AMOUNT        = 5_000_000  # min daily turnover (RMB) for liquidity
    EXCLUDE_ST        = True       # exclude ST / *ST stocks
    EXCLUDE_NEW_DAYS  = 60         # exclude stocks listed < N trading days
    EXCLUDE_AT_LIMIT  = True       # skip stocks locked at ±10 % price limit

    # ─── Strategy ────────────────────────────────────────────────────────────
    STRATEGY_NAME  = "closetoclose"
    N_POSITIONS    = 30            # stocks to hold each day (matches image ~30)
    HOLDING_DAYS   = 1             # close-to-close = 1 day hold
    SIGNAL_ALPHA   = 0.5           # blend: factor_score * α + ml_score * (1−α)

    # ─── Capital ─────────────────────────────────────────────────────────────
    INITIAL_CAPITAL = 1_280_000    # matches image

    # ─── A-share transaction costs ───────────────────────────────────────────
    BUY_COMMISSION  = 0.0003       # 0.03 % (charged by broker, one-way)
    SELL_COMMISSION = 0.0003       # 0.03 % (charged by broker, one-way)
    STAMP_DUTY      = 0.0005       # 0.05 % sell-side (reduced Aug 2023)
    SLIPPAGE        = 0.001        # 0.1 % estimated market-impact slippage
    LOT_SIZE        = 100          # minimum tradable lot in A-shares

    # ─── Factor windows ──────────────────────────────────────────────────────
    MOM_WINDOWS  = [1, 5, 10, 20, 60]   # momentum look-back windows (days)
    VOL_WINDOWS  = [5, 20]               # volatility windows
    RSI_WINDOWS  = [5, 14]               # RSI periods

    # ─── ML model (LightGBM) ─────────────────────────────────────────────────
    ROLLING_TRAIN_WINDOW = 252     # trading days used for each rolling fit
    RETRAIN_FREQ         = 21      # retrain every N trading days
    MIN_TRAIN_SAMPLES    = 5_000   # minimum rows before training
    N_ESTIMATORS         = 300
    MAX_DEPTH            = 6
    LEARNING_RATE        = 0.05
    SUBSAMPLE            = 0.8
    COLSAMPLE_BYTREE     = 0.8
    NUM_LEAVES           = 63
    RANDOM_STATE         = 42

    # ─── Benchmark ───────────────────────────────────────────────────────────
    BENCHMARK_CODE = "sh.000300"   # CSI 300 (baostock format)

    # ─── Paths ───────────────────────────────────────────────────────────────
    BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR      = os.path.join(BASE_DIR, "data")
    CACHE_DIR     = os.path.join(BASE_DIR, "data", "cache")
    OUTPUT_DIR    = os.path.join(BASE_DIR, "output")

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.DATA_DIR, cls.CACHE_DIR, cls.OUTPUT_DIR]:
            os.makedirs(d, exist_ok=True)
