"""
CloseToClose Strategy — daily stock-selection signal.

Design
------
On each trading day T (after market close):
  1. Compute a composite score for every stock in the universe.
  2. Apply hard filters (price limits, liquidity, ST).
  3. Rank stocks by composite score.
  4. Return the top-N candidates to buy at close T (hold until close T+1).

Composite score = α × factor_rank + (1 − α) × ml_score
  - factor_rank : IC-IR weighted combination of all alpha factors.
  - ml_score    : LightGBM predicted next-day return (cross-sectionally ranked).
  - α           : Config.SIGNAL_ALPHA (default 0.5)

The factor weights are computed once during `fit()` using IC-IR on the
training panel, making the weighting data-driven rather than hand-tuned.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from config import Config
from factors import FactorEngine

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class CloseToCloseStrategy:
    """
    Blended factor + ML signal for daily close-to-close stock selection.

    Usage
    -----
    strat = CloseToCloseStrategy(config)
    strat.fit(train_panel)                          # compute IC-IR weights
    signals = strat.generate_signals(panel, ml_scores)  # full signal table
    daily_picks = strat.daily_picks(signals, date)  # codes for one day
    """

    def __init__(self, config: Config = None):
        self.cfg     = config or Config()
        self.weights_: pd.Series | None = None      # IC-IR factor weights
        self._fe     = FactorEngine(self.cfg)

    # ─── Training: compute factor weights via IC-IR ───────────────────────────

    def fit(self, train_panel: pd.DataFrame) -> "CloseToCloseStrategy":
        """
        Compute IC-IR weighted factor combination on the training panel.
        Stores factor weights in self.weights_.
        """
        log.info("Computing IC-IR factor weights on training data …")
        ic_df = self._fe.ic_summary(train_panel)

        # Use IC-IR as weight; clip negative IC-IR to zero (don't invert bad factors)
        weights = ic_df["ic_ir"].clip(lower=0)

        # Exclude factors with essentially zero IC-IR (noise)
        weights = weights[weights > 0.05]

        if weights.sum() < 1e-9:
            # Fallback: equal weights on all factors
            weights = pd.Series(
                1.0, index=self._fe.FACTOR_COLS
            )

        self.weights_ = weights / weights.sum()
        log.info(f"  Top-5 factors by IC-IR:\n{ic_df.head()}")
        return self

    # ─── Signal generation ───────────────────────────────────────────────────

    def generate_signals(
        self,
        panel: pd.DataFrame,
        ml_scores: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build the composite signal table for all (date, code) rows.

        Parameters
        ----------
        panel     : enriched panel from FactorEngine (factor columns present)
        ml_scores : DataFrame with (date, code) index and "ml_score" column.
                    If None, only factor scores are used (α=1).

        Returns
        -------
        DataFrame with (date, code) index and columns:
          [factor_score, ml_score (opt), composite_score,
           close, amount, high, low, pctChg, isST]
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before generate_signals().")

        available_factors = [c for c in self.weights_.index if c in panel.columns]
        w = self.weights_.reindex(available_factors).fillna(0)
        w = w / w.sum()

        # Factor composite: weighted sum of per-date rank-normalised factor cols
        factor_matrix = panel[available_factors]
        factor_scores  = (factor_matrix * w.values).sum(axis=1)
        factor_scores  = factor_scores.groupby(level="date").rank(pct=True)

        signals = panel[["close", "amount", "high", "low", "pctChg", "isST",
                         "open"]].copy()
        signals["factor_score"] = factor_scores

        if ml_scores is not None and not ml_scores.empty:
            signals = signals.join(ml_scores[["ml_score"]], how="left")
            signals["ml_score"] = signals["ml_score"].fillna(0.5)
            alpha = self.cfg.SIGNAL_ALPHA
            signals["composite_score"] = (
                alpha * signals["factor_score"]
                + (1 - alpha) * signals["ml_score"]
            )
        else:
            signals["ml_score"] = np.nan
            signals["composite_score"] = signals["factor_score"]

        return signals.sort_index()

    # ─── Daily picks ─────────────────────────────────────────────────────────

    def daily_picks(
        self,
        signals: pd.DataFrame,
        date: pd.Timestamp,
        n: int | None = None,
    ) -> list[str]:
        """
        Return the top-N stock codes to buy at close on *date*.

        Applies hard filters:
          - Exclude stocks at or near the 10 % upper price limit (can't buy)
          - Exclude ST stocks
          - Require minimum liquidity
        """
        n = n or self.cfg.N_POSITIONS

        try:
            day = signals.xs(date, level="date")
        except KeyError:
            return []

        # Hard filter 1: not at upper price limit (pctChg ≥ 9.5 %)
        if "pctChg" in day.columns:
            day = day[day["pctChg"] < 9.5]

        # Hard filter 2: exclude ST
        if "isST" in day.columns:
            day = day[day["isST"] != "1"]

        # Hard filter 3: minimum amount
        if "amount" in day.columns:
            day = day[day["amount"] >= self.cfg.MIN_AMOUNT]

        if day.empty:
            return []

        top = day.nlargest(n, "composite_score")
        return top.index.tolist()

    # ─── Convenience: full backtest signals ───────────────────────────────────

    def get_all_picks(
        self,
        signals: pd.DataFrame,
        trading_dates: list[pd.Timestamp],
        n: int | None = None,
    ) -> dict[pd.Timestamp, list[str]]:
        """Return {date: [codes]} dict for every date in trading_dates."""
        return {
            d: self.daily_picks(signals, d, n)
            for d in trading_dates
            if d in signals.index.get_level_values("date")
        }
