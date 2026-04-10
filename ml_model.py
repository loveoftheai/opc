"""
LightGBM Rolling-Window Signal Model.

Approach:
  - Frame as a ranking problem: predict next-day VWAP return
    (Config.TARGET_COL = "target_vwap" by default).
  - VWAP = amount / volume; using it as the label is more robust than
    close-to-close because VWAP averages over the whole session and is
    harder to manipulate at the close.
  - Train on a rolling window (default 252 trading days).
  - Retrain every RETRAIN_FREQ trading days (default 5 = weekly).
  - At inference time, rank stocks by predicted score and select top-N.

The model operates purely on the factor panel produced by factors.py, so
there is no look-ahead: all features for date T use only data up to close T.
"""

import os
import logging
import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

from config import Config
from factors import FactorEngine

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class LGBMSignal:
    """
    Rolling LightGBM signal generator.

    Call `fit_predict(panel)` to obtain a DataFrame of daily predicted scores
    for each stock, covering the dates not used for the initial training burn-in.
    """

    FEATURE_COLS = FactorEngine.FACTOR_COLS   # use all factors as ML features

    def __init__(self, config: Config = None):
        self.cfg       = config or Config()
        self.target_col = getattr(self.cfg, "TARGET_COL", "target_vwap")
        self.model_:  Optional[lgb.Booster] = None
        self.scaler_: Optional[RobustScaler] = None
        self._train_dates: list = []

    # ─── LightGBM hyperparameters ─────────────────────────────────────────────

    @property
    def _lgb_params(self) -> dict:
        return {
            "objective":        "regression",
            "metric":           "rmse",
            "n_estimators":     self.cfg.N_ESTIMATORS,
            "max_depth":        self.cfg.MAX_DEPTH,
            "learning_rate":    self.cfg.LEARNING_RATE,
            "subsample":        self.cfg.SUBSAMPLE,
            "colsample_bytree": self.cfg.COLSAMPLE_BYTREE,
            "num_leaves":       self.cfg.NUM_LEAVES,
            "min_child_samples": 20,
            "reg_alpha":        0.1,
            "reg_lambda":       1.0,
            "random_state":     self.cfg.RANDOM_STATE,
            "n_jobs":           -1,
            "verbose":          -1,
        }

    # ─── Core API ─────────────────────────────────────────────────────────────

    def fit_predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling train-predict loop.

        For each date in [train_end+1, test_end]:
          - Train on the ROLLING_TRAIN_WINDOW trading days preceding this date.
          - Predict scores for all stocks available on this date.

        Returns
        -------
        DataFrame with (date, code) index and column "ml_score".
        """
        available_features = [c for c in self.FEATURE_COLS if c in panel.columns]
        trading_dates = sorted(panel.index.get_level_values("date").unique())

        # Determine which dates are in the test window
        test_start = pd.Timestamp(self.cfg.TEST_START)
        train_start = pd.Timestamp(self.cfg.TRAIN_START)

        # Build list of all dates; predictions start after sufficient training data
        all_preds = []
        last_train_date = None

        # Find index of first prediction date
        pred_dates = [d for d in trading_dates if d >= test_start]
        # Also include a few months before test_start to warm-up the backtest
        warmup_start = test_start - pd.Timedelta(days=90)
        pred_dates_full = [d for d in trading_dates if d >= warmup_start]

        log.info(f"Rolling ML: {len(pred_dates_full)} prediction dates, "
                 f"window={self.cfg.ROLLING_TRAIN_WINDOW} days, "
                 f"retrain every {self.cfg.RETRAIN_FREQ} days, "
                 f"label={self.target_col}")

        for i, pred_date in enumerate(pred_dates_full):
            # Retrain only at initial run or every RETRAIN_FREQ steps
            needs_train = (
                last_train_date is None
                or i % self.cfg.RETRAIN_FREQ == 0
            )

            if needs_train:
                train_end_idx = trading_dates.index(pred_date) - 1
                train_start_idx = max(0, train_end_idx - self.cfg.ROLLING_TRAIN_WINDOW)
                train_dates = trading_dates[train_start_idx:train_end_idx + 1]

                if len(train_dates) < 30:
                    continue

                train_panel = panel.loc[panel.index.get_level_values("date").isin(
                    set(train_dates)
                )]
                train_panel = train_panel.dropna(subset=available_features + [self.target_col])

                if len(train_panel) < self.cfg.MIN_TRAIN_SAMPLES:
                    log.debug(f"  Skipping train on {pred_date.date()}: "
                              f"only {len(train_panel)} samples")
                    if self.model_ is None:
                        continue
                else:
                    X_train = train_panel[available_features].values.astype(np.float32)
                    y_train = train_panel[self.target_col].values.astype(np.float32)

                    # Winsorise targets to ±10 %
                    y_train = np.clip(y_train, -0.10, 0.10)

                    self.scaler_ = RobustScaler()
                    X_train = self.scaler_.fit_transform(X_train)

                    model = lgb.LGBMRegressor(**self._lgb_params)
                    model.fit(
                        X_train, y_train,
                        callbacks=[lgb.early_stopping(30, verbose=False),
                                   lgb.log_evaluation(-1)],
                        eval_set=[(X_train, y_train)],
                    )
                    self.model_ = model
                    last_train_date = pred_date

            if self.model_ is None:
                continue

            # Predict on all stocks available on pred_date
            try:
                day_data = panel.xs(pred_date, level="date")
            except KeyError:
                continue

            day_data = day_data.dropna(subset=available_features)
            if day_data.empty:
                continue

            X_pred = day_data[available_features].values.astype(np.float32)
            X_pred = self.scaler_.transform(X_pred)

            scores = self.model_.predict(X_pred)
            pred_df = pd.DataFrame({
                "date":     pred_date,
                "code":     day_data.index.tolist(),
                "ml_score": scores,
            })
            all_preds.append(pred_df)

        if not all_preds:
            raise RuntimeError("ML model produced no predictions. "
                               "Check training data size.")

        result = pd.concat(all_preds, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        result = result.set_index(["date", "code"]).sort_index()

        # Cross-sectional rank normalise the ml_score per date
        result["ml_score"] = (
            result.groupby(level="date")["ml_score"]
            .rank(pct=True)
        )
        log.info(f"ML predictions: {len(result):,} rows across "
                 f"{result.index.get_level_values('date').nunique()} dates")
        return result

    # ─── Feature importance ───────────────────────────────────────────────────

    def feature_importance(self) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Model not trained yet.")
        available = [c for c in self.FEATURE_COLS if c in self.FEATURE_COLS]
        imp = self.model_.feature_importances_
        return pd.Series(imp, index=available[:len(imp)]).sort_values(ascending=False)

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model_, "scaler": self.scaler_}, f)
        log.info(f"Model saved → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.model_  = obj["model"]
        self.scaler_ = obj["scaler"]
        log.info(f"Model loaded ← {path}")
