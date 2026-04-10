"""
Multi-Factor Alpha Engine for A-share CloseToClose strategy.

Factor categories:
  1. Momentum     — cross-sectional price momentum at multiple horizons
  2. Reversal     — short-term mean-reversion
  3. Volume       — volume-price relationship & turnover
  4. Technical    — RSI, MACD, Bollinger Bands, moving-average crossovers
  5. Candlestick  — intraday price pattern (close position, body, shadows)
  6. Volatility   — realized volatility at multiple horizons

All factors are computed per-stock via groupby, then cross-sectionally
z-scored (rank-based) so they are comparable across stocks and time.
"""

import numpy as np
import pandas as pd
import warnings

from config import Config

warnings.filterwarnings("ignore")


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _cs_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0, 1]."""
    return s.rank(pct=True)


def _cs_zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score (mean-zero, std-one)."""
    mu, sd = s.mean(), s.std(ddof=0)
    if sd < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up   = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    gain = up.ewm(com=window - 1, min_periods=window).mean()
    loss = down.ewm(com=window - 1, min_periods=window).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = close.ewm(span=fast,   min_periods=fast).mean()
    ema_slow   = close.ewm(span=slow,   min_periods=slow).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, window=20, num_std=2):
    ma  = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    # Position within bands: 0 = at lower, 1 = at upper
    band_width = upper - lower
    position = (close - lower) / band_width.replace(0, np.nan)
    return position.clip(0, 1)


# ─── Main factor engine ───────────────────────────────────────────────────────

class FactorEngine:
    """
    Compute all alpha factors on a stacked panel DataFrame.

    Input  : panel with MultiIndex (date, code) and columns
             [open, high, low, close, volume, amount, turn, pctChg]
    Output : panel augmented with factor columns + target column
    """

    def __init__(self, config: Config = None):
        self.cfg = config or Config()

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry: compute all factors, add target, cross-sectionally
        normalise, and return the enriched panel.
        """
        panel = panel.copy()
        panel = panel.sort_index()

        # Per-stock rolling factors (applied via groupby)
        panel = self._add_per_stock_factors(panel)

        # Cross-sectional normalisation (applied per date)
        panel = self._normalise_cs(panel)

        # Target variable: next-day close-to-close return
        panel = self._add_target(panel)

        return panel

    # ─── Per-stock rolling factors ────────────────────────────────────────────

    def _add_per_stock_factors(self, panel: pd.DataFrame) -> pd.DataFrame:
        result_frames = []

        for code, grp in panel.groupby(level="code", sort=False):
            grp = grp.sort_index(level="date").copy()
            c = grp["close"]
            o = grp["open"]
            h = grp["high"]
            lo = grp["low"]
            v = grp["volume"]
            amt = grp["amount"]
            turn = grp.get("turn", pd.Series(np.nan, index=grp.index))

            # ── 1. Momentum ──────────────────────────────────────────────────
            for w in self.cfg.MOM_WINDOWS:
                grp[f"mom_{w}d"] = c.pct_change(w)

            # ── 2. Short-term reversal ────────────────────────────────────────
            grp["rev_1d"]  = -c.pct_change(1)
            grp["rev_5d"]  = -c.pct_change(5)

            # ── 3. Volume factors ─────────────────────────────────────────────
            for w in [5, 10, 20]:
                avg_vol = v.rolling(w, min_periods=w).mean()
                avg_amt = amt.rolling(w, min_periods=w).mean()
                grp[f"vol_ratio_{w}d"] = v / avg_vol.replace(0, np.nan)
                grp[f"amt_ratio_{w}d"] = amt / avg_amt.replace(0, np.nan)

            grp["turn_5d_avg"] = turn.rolling(5, min_periods=3).mean()

            # ── 4. Moving-average crossover ────────────────────────────────────
            for w in [5, 10, 20, 60]:
                ma = c.rolling(w, min_periods=w).mean()
                grp[f"price_vs_ma{w}"] = (c - ma) / ma.replace(0, np.nan)

            # ── 5. RSI ────────────────────────────────────────────────────────
            for w in self.cfg.RSI_WINDOWS:
                grp[f"rsi_{w}"] = _rsi(c, w)

            # ── 6. MACD ───────────────────────────────────────────────────────
            _, _, grp["macd_hist"] = _macd(c)
            # Normalise MACD histogram by price level
            grp["macd_hist"] = grp["macd_hist"] / c.replace(0, np.nan)

            # ── 7. Bollinger Band position ─────────────────────────────────────
            grp["bb_pos"] = _bollinger(c, window=20)

            # ── 8. Candlestick pattern factors ────────────────────────────────
            rng = (h - lo).replace(0, np.nan)
            grp["close_pos"]    = (c - lo) / rng          # 0=low end, 1=high end
            body                = (c - o).abs()
            grp["body_ratio"]   = body / rng
            upper_shadow        = h - pd.concat([c, o], axis=1).max(axis=1)
            lower_shadow        = pd.concat([c, o], axis=1).min(axis=1) - lo
            grp["upper_shadow"] = upper_shadow / rng
            grp["lower_shadow"] = lower_shadow / rng
            grp["bullish_body"] = ((c > o).astype(float) * 2 - 1)  # +1 up, -1 down

            # ── 9. Overnight gap ──────────────────────────────────────────────
            grp["open_gap"] = (o - c.shift(1)) / c.shift(1).replace(0, np.nan)

            # ── 10. Volatility ────────────────────────────────────────────────
            log_ret = np.log(c / c.shift(1))
            for w in self.cfg.VOL_WINDOWS:
                grp[f"vol_{w}d"] = log_ret.rolling(w, min_periods=w).std() * np.sqrt(252)
            # Volatility ratio: short-term vs long-term
            grp["vol_ratio"] = (grp[f"vol_{self.cfg.VOL_WINDOWS[0]}d"] /
                                grp[f"vol_{self.cfg.VOL_WINDOWS[-1]}d"].replace(0, np.nan))

            # ── 11. Amihud illiquidity proxy ──────────────────────────────────
            abs_ret = c.pct_change().abs()
            grp["illiq_5d"] = (abs_ret / amt.replace(0, np.nan)).rolling(5, min_periods=3).mean()

            result_frames.append(grp)

        return pd.concat(result_frames).sort_index()

    # ─── Cross-sectional normalisation ────────────────────────────────────────

    FACTOR_COLS = [
        # momentum
        "mom_1d", "mom_5d", "mom_10d", "mom_20d", "mom_60d",
        # reversal
        "rev_1d", "rev_5d",
        # volume
        "vol_ratio_5d", "vol_ratio_10d", "vol_ratio_20d",
        "amt_ratio_5d", "amt_ratio_10d", "amt_ratio_20d",
        "turn_5d_avg",
        # MA crossover
        "price_vs_ma5", "price_vs_ma10", "price_vs_ma20", "price_vs_ma60",
        # technical
        "rsi_5", "rsi_14",
        "macd_hist", "bb_pos",
        # candlestick
        "close_pos", "body_ratio", "upper_shadow", "lower_shadow", "bullish_body",
        # overnight
        "open_gap",
        # volatility
        "vol_5d", "vol_20d", "vol_ratio",
        # liquidity
        "illiq_5d",
    ]

    def _normalise_cs(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-sectional rank normalisation (per date) to all factor cols."""
        available = [c for c in self.FACTOR_COLS if c in panel.columns]

        def _rank_group(grp):
            for col in available:
                grp[col] = _cs_rank(grp[col])
            return grp

        panel = panel.groupby(level="date", group_keys=False).apply(_rank_group)
        return panel

    # ─── Target variable ──────────────────────────────────────────────────────

    def _add_target(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        target_ret: next-day close-to-close return (what we trade).
        Computed per-stock to avoid look-ahead via groupby shift.
        """
        def _next_ret(grp):
            grp["target_ret"] = grp["close"].pct_change(1).shift(-1)
            return grp

        panel = panel.groupby(level="code", group_keys=False).apply(_next_ret)
        return panel

    # ─── Factor IC analysis ───────────────────────────────────────────────────

    def compute_ic(self, panel: pd.DataFrame, factor_col: str,
                   target_col: str = "target_ret") -> pd.Series:
        """
        Return daily IC (rank correlation between factor and next-day return).
        Useful for evaluating individual factor quality.
        """
        def _daily_ic(grp):
            return grp[factor_col].corr(grp[target_col], method="spearman")

        ic = panel.dropna(subset=[factor_col, target_col]).groupby(
            level="date").apply(_daily_ic)
        return ic

    def ic_summary(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Compute IC mean / IC_IR for all factor columns."""
        records = []
        available = [c for c in self.FACTOR_COLS if c in panel.columns]
        for col in available:
            ic = self.compute_ic(panel, col)
            records.append({
                "factor":  col,
                "ic_mean": ic.mean(),
                "ic_std":  ic.std(),
                "ic_ir":   ic.mean() / ic.std() if ic.std() > 0 else 0,
                "ic_pos_ratio": (ic > 0).mean(),
            })
        return pd.DataFrame(records).set_index("factor").sort_values(
            "ic_ir", ascending=False
        )
