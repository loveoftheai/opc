"""
Visualisation — replicates the dark-themed dashboard shown in the reference image.

Charts produced
---------------
  1. Top panel   : strategy vs benchmark cumulative return curve
  2. Middle panel: rolling drawdown (shaded red)
  3. Lower panel : daily P&L bar chart (green/red)
  4. Bottom panel: trade P&L bar chart
  5. Standalone  : factor IC bar chart (optional)
"""

import os
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from config import Config

warnings.filterwarnings("ignore")

# ─── Theme ────────────────────────────────────────────────────────────────────

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
GRID_COL  = "#2a2a4a"
BLUE      = "#4fc3f7"
ORANGE    = "#ffb74d"
RED       = "#ef5350"
GREEN     = "#66bb6a"
TEXT_COL  = "#e0e0e0"
ACCENT    = "#7c4dff"


def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_COL,
        "axes.labelcolor":   TEXT_COL,
        "axes.grid":         True,
        "grid.color":        GRID_COL,
        "grid.linewidth":    0.5,
        "xtick.color":       TEXT_COL,
        "ytick.color":       TEXT_COL,
        "text.color":        TEXT_COL,
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  GRID_COL,
        "font.family":       "DejaVu Sans",
        "font.size":         9,
    })


# ─── Main dashboard ───────────────────────────────────────────────────────────

def plot_dashboard(
    result,
    benchmark: Optional[pd.Series] = None,
    metrics:   Optional[dict]      = None,
    save_path: Optional[str]       = None,
):
    """
    Generate the full performance dashboard (dark theme).

    Parameters
    ----------
    result     : BacktestResult
    benchmark  : daily benchmark returns (Series, index=date)
    metrics    : dict from PerformanceAnalyzer.compute()
    save_path  : if given, save PNG to this path; otherwise show interactively
    """
    _apply_dark_style()

    daily   = result.daily
    equity  = result.equity_curve
    dr      = result.daily_returns
    trades  = result.trades

    # Normalised equity (start = 1.0)
    norm_eq = equity / equity.iloc[0]

    # Benchmark cumulative return
    bm_cum = None
    if benchmark is not None:
        bm = benchmark.reindex(equity.index).fillna(0)
        bm_cum = (1 + bm).cumprod()

    # Drawdown series
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max

    fig = plt.figure(figsize=(14, 12), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[3, 1, 1.2, 1.2],
        hspace=0.08,
    )

    ax1 = fig.add_subplot(gs[0])   # equity curve
    ax2 = fig.add_subplot(gs[1], sharex=ax1)   # drawdown
    ax3 = fig.add_subplot(gs[2], sharex=ax1)   # daily P&L
    ax4 = fig.add_subplot(gs[3], sharex=ax1)   # trade P&L

    dates = equity.index

    # ── Panel 1: equity curve ─────────────────────────────────────────────────
    ax1.plot(dates, norm_eq,  color=BLUE,   lw=1.8, label="策略收益")
    if bm_cum is not None:
        ax1.plot(dates, bm_cum, color=ORANGE, lw=1.2, label="基准收益", linestyle="--")
    ax1.axhline(1.0, color=GRID_COL, lw=0.8, linestyle=":")
    ax1.set_ylabel("倍数", color=TEXT_COL)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Annotate final return
    final_ret = float(norm_eq.iloc[-1] - 1) * 100
    ax1.annotate(
        f"+{final_ret:.2f}%",
        xy=(dates[-1], float(norm_eq.iloc[-1])),
        xytext=(-60, 10), textcoords="offset points",
        fontsize=9, color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
    )

    # ── Panel 2: drawdown ─────────────────────────────────────────────────────
    ax2.fill_between(dates, drawdown * 100, 0, color=RED, alpha=0.6)
    ax2.plot(dates, drawdown * 100, color=RED, lw=0.8)
    ax2.set_ylabel("回撤(%)", color=TEXT_COL)
    ax2.axhline(0, color=GRID_COL, lw=0.5)

    # ── Panel 3: daily P&L ────────────────────────────────────────────────────
    pnl_values = dr * equity.shift(1).fillna(equity.iloc[0])
    colors3 = [GREEN if v >= 0 else RED for v in pnl_values]
    ax3.bar(dates, pnl_values, color=colors3, width=0.8, alpha=0.85)
    ax3.axhline(0, color=GRID_COL, lw=0.5)
    ax3.set_ylabel("日盈亏", color=TEXT_COL)
    ax3.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e4:.0f}万" if abs(x) >= 1e4 else f"{x:.0f}")
    )

    # ── Panel 4: per-trade P&L bar ────────────────────────────────────────────
    if not trades.empty:
        trade_dates  = pd.to_datetime(trades["sell_date"])
        trade_pnl    = trades["net_pnl"].values
        colors4      = [GREEN if p >= 0 else RED for p in trade_pnl]
        ax4.bar(trade_dates, trade_pnl, color=colors4, width=0.6, alpha=0.75)
        ax4.axhline(0, color=GRID_COL, lw=0.5)
    ax4.set_ylabel("每笔盈亏", color=TEXT_COL)
    ax4.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e4:.1f}万" if abs(x) >= 1e4 else f"{x:.0f}")
    )
    ax4.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
    ax4.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval=2))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # ── Title + metrics sidebar ────────────────────────────────────────────────
    title_str = (
        f"CloseToClose 策略回测  |  "
        f"{equity.index[0].strftime('%Y-%m-%d')} → {equity.index[-1].strftime('%Y-%m-%d')}"
    )
    if metrics:
        ar   = metrics.get("annual_return", 0) * 100
        mdd  = metrics.get("max_drawdown",  0) * 100
        sr   = metrics.get("sharpe_ratio",  0)
        wr   = metrics.get("win_rate",      0) * 100
        title_str += (
            f"\n年化收益 {ar:+.2f}%  |  最大回撤 {mdd:.2f}%  "
            f"|  夏普 {sr:.2f}  |  胜率 {wr:.1f}%"
        )
    fig.suptitle(title_str, color=TEXT_COL, fontsize=10, y=0.98)

    # Remove x-tick labels from top panels
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Dashboard saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─── Factor IC bar chart ──────────────────────────────────────────────────────

def plot_factor_ic(ic_df: pd.DataFrame, save_path: Optional[str] = None, top_n: int = 20):
    """Plot IC-IR bar chart for top factors."""
    _apply_dark_style()

    df = ic_df.head(top_n).sort_values("ic_ir")
    colors = [GREEN if v >= 0 else RED for v in df["ic_ir"]]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.barh(df.index, df["ic_ir"], color=colors, alpha=0.85)
    ax.axvline(0, color=TEXT_COL, lw=0.8)
    ax.set_xlabel("IC-IR", color=TEXT_COL)
    ax.set_title(f"因子 IC-IR 排名 (Top {top_n})", color=TEXT_COL)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
        print(f"IC chart saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─── Return distribution ──────────────────────────────────────────────────────

def plot_return_distribution(daily_returns: pd.Series, save_path: Optional[str] = None):
    """Histogram + KDE of daily returns."""
    _apply_dark_style()

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    ax.hist(daily_returns * 100, bins=40, color=BLUE, alpha=0.7, density=True, label="日收益率")
    daily_returns.mul(100).plot.kde(ax=ax, color=ORANGE, lw=2)
    ax.axvline(0, color=RED, lw=1, linestyle="--")
    ax.set_xlabel("日收益率 (%)", color=TEXT_COL)
    ax.set_title("日收益率分布", color=TEXT_COL)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Distribution chart saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
