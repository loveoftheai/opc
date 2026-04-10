"""
Backtest Engine — T+1 aware close-to-close simulation for A-shares.

Execution model
---------------
  Day T (after market close):
    1. Exit all positions held from Day T-1  → sell at Day T close price.
    2. Generate signal for Day T.
    3. Enter new positions                   → buy at Day T close price.

T+1 constraint is satisfied: stocks bought at close T are sold at close T+1
(the very next trading day), which is the earliest allowed exit.

Transaction costs (A-share realistic):
  - Buy  : commission = BUY_COMMISSION × notional
  - Sell : commission = SELL_COMMISSION × notional + STAMP_DUTY × notional
  - Both : slippage   = SLIPPAGE × notional (applied to each side)

Lot-size rounding: A-shares require minimum 100-share lots.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Position:
    code:       str
    shares:     int
    buy_price:  float
    buy_date:   pd.Timestamp
    buy_cost:   float          # total cost incl. commission + slippage


@dataclass
class Trade:
    code:        str
    buy_date:    pd.Timestamp
    sell_date:   pd.Timestamp
    buy_price:   float
    sell_price:  float
    shares:      int
    gross_pnl:   float         # (sell - buy) × shares
    net_pnl:     float         # gross_pnl - all costs
    net_return:  float         # net_pnl / total_buy_cost


@dataclass
class DailyRecord:
    date:            pd.Timestamp
    portfolio_value: float
    cash:            float
    holdings_value:  float
    daily_return:    float
    n_positions:     int
    turnover:        float     # traded notional / portfolio value


# ─── Engine ───────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven backtest simulator for the CloseToClose strategy.

    Parameters
    ----------
    picks      : dict  {date: [code, …]}  — daily selected stocks
    panel      : MultiIndex DataFrame (date, code) with at least 'close'
    config     : Config object
    """

    def __init__(
        self,
        picks:  Dict[pd.Timestamp, List[str]],
        panel:  pd.DataFrame,
        config: Config = None,
    ):
        self.picks  = picks
        self.panel  = panel
        self.cfg    = config or Config()

        # Runtime state
        self._cash:       float             = self.cfg.INITIAL_CAPITAL
        self._positions:  Dict[str, Position] = {}
        self._daily:      List[DailyRecord]   = []
        self._trades:     List[Trade]          = []

    # ─── Public interface ─────────────────────────────────────────────────────

    def run(self) -> "BacktestResult":
        """Execute the backtest and return a BacktestResult."""
        trading_dates = sorted(self.picks.keys())
        if not trading_dates:
            raise ValueError("No trading dates in picks dict.")

        log.info(f"Backtest: {trading_dates[0].date()} → {trading_dates[-1].date()}, "
                 f"{len(trading_dates)} days")

        prev_value = self.cfg.INITIAL_CAPITAL

        for date in trading_dates:
            prices = self._prices_on(date)

            # ── Step 1: sell all holdings from previous day ───────────────────
            sell_notional = self._sell_all(date, prices)

            # ── Step 2: buy today's picks ─────────────────────────────────────
            today_codes = self.picks.get(date, [])
            buy_notional = self._buy_positions(date, today_codes, prices)

            # ── Step 3: mark-to-market ────────────────────────────────────────
            holdings_val = self._holdings_value(prices)
            total_val    = self._cash + holdings_val
            daily_ret    = (total_val - prev_value) / prev_value if prev_value > 0 else 0
            turnover     = (sell_notional + buy_notional) / prev_value if prev_value > 0 else 0

            self._daily.append(DailyRecord(
                date            = date,
                portfolio_value = total_val,
                cash            = self._cash,
                holdings_value  = holdings_val,
                daily_return    = daily_ret,
                n_positions     = len(self._positions),
                turnover        = turnover,
            ))
            prev_value = total_val

        return BacktestResult(
            daily   = pd.DataFrame([vars(d) for d in self._daily]).set_index("date"),
            trades  = pd.DataFrame([vars(t) for t in self._trades]),
            config  = self.cfg,
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _prices_on(self, date: pd.Timestamp) -> Dict[str, dict]:
        """Return {code: {close, high, low, pctChg, amount}} for date."""
        try:
            day_df = self.panel.xs(date, level="date")
        except KeyError:
            return {}
        result = {}
        for code, row in day_df.iterrows():
            result[code] = {
                "close":  float(row.get("close", np.nan)),
                "high":   float(row.get("high",  np.nan)),
                "low":    float(row.get("low",   np.nan)),
                "pctChg": float(row.get("pctChg", 0.0)),
                "amount": float(row.get("amount", 0.0)),
            }
        return result

    def _sell_all(self, date: pd.Timestamp, prices: Dict[str, dict]) -> float:
        """Close every open position; record trades; return total sell notional."""
        total_notional = 0.0
        for code, pos in list(self._positions.items()):
            p = prices.get(code)
            if p is None or np.isnan(p["close"]):
                # Can't sell — no price today; hold over (shouldn't happen often)
                log.debug(f"  No price for {code} on {date.date()}, holding over.")
                continue

            sell_price = p["close"]

            # If stock is at lower limit (−9.5 %), we're forced to sell anyway
            # (in reality you may be stuck; we assume you can sell at limit price)
            gross_sell   = sell_price * pos.shares
            sell_comm    = gross_sell * (self.cfg.SELL_COMMISSION + self.cfg.STAMP_DUTY)
            sell_slip    = gross_sell * self.cfg.SLIPPAGE
            net_proceeds = gross_sell - sell_comm - sell_slip

            gross_pnl = (sell_price - pos.buy_price) * pos.shares
            net_pnl   = net_proceeds - pos.buy_cost

            self._cash += net_proceeds
            total_notional += gross_sell

            self._trades.append(Trade(
                code       = code,
                buy_date   = pos.buy_date,
                sell_date  = date,
                buy_price  = pos.buy_price,
                sell_price = sell_price,
                shares     = pos.shares,
                gross_pnl  = gross_pnl,
                net_pnl    = net_pnl,
                net_return = net_pnl / pos.buy_cost if pos.buy_cost > 0 else 0.0,
            ))
            del self._positions[code]

        return total_notional

    def _buy_positions(
        self,
        date: pd.Timestamp,
        codes: List[str],
        prices: Dict[str, dict],
    ) -> float:
        """Open equal-weight positions in selected codes. Return buy notional."""
        # Filter out codes without price data or at upper limit
        buyable = []
        for code in codes:
            p = prices.get(code)
            if p is None or np.isnan(p["close"]):
                continue
            if self.cfg.EXCLUDE_AT_LIMIT and p["pctChg"] >= 9.5:
                continue    # locked at upper limit — can't buy
            buyable.append(code)

        if not buyable:
            return 0.0

        n = len(buyable)
        per_position_cash = self._cash / n     # equal weight
        total_notional = 0.0

        for code in buyable:
            buy_price = prices[code]["close"]
            if buy_price <= 0:
                continue

            # Round down to nearest lot (100 shares)
            raw_shares = int(per_position_cash / buy_price)
            shares = (raw_shares // self.cfg.LOT_SIZE) * self.cfg.LOT_SIZE
            if shares < self.cfg.LOT_SIZE:
                continue

            gross_cost  = buy_price * shares
            buy_comm    = gross_cost * self.cfg.BUY_COMMISSION
            buy_slip    = gross_cost * self.cfg.SLIPPAGE
            total_cost  = gross_cost + buy_comm + buy_slip

            if total_cost > self._cash:
                # Try smaller lot
                shares -= self.cfg.LOT_SIZE
                if shares < self.cfg.LOT_SIZE:
                    continue
                gross_cost  = buy_price * shares
                buy_comm    = gross_cost * self.cfg.BUY_COMMISSION
                buy_slip    = gross_cost * self.cfg.SLIPPAGE
                total_cost  = gross_cost + buy_comm + buy_slip
                if total_cost > self._cash:
                    continue

            self._cash -= total_cost
            total_notional += gross_cost

            self._positions[code] = Position(
                code      = code,
                shares    = shares,
                buy_price = buy_price,
                buy_date  = date,
                buy_cost  = total_cost,
            )

        return total_notional

    def _holdings_value(self, prices: Dict[str, dict]) -> float:
        val = 0.0
        for code, pos in self._positions.items():
            p = prices.get(code)
            if p and not np.isnan(p["close"]):
                val += p["close"] * pos.shares
            else:
                val += pos.buy_price * pos.shares   # use cost basis as fallback
        return val


# ─── Result container ─────────────────────────────────────────────────────────

class BacktestResult:
    """
    Container for backtest output. Provides convenient accessors
    for daily returns, equity curve, and trade log.
    """

    def __init__(
        self,
        daily:   pd.DataFrame,
        trades:  pd.DataFrame,
        config:  Config,
    ):
        self.daily  = daily
        self.trades = trades
        self.cfg    = config

    @property
    def equity_curve(self) -> pd.Series:
        return self.daily["portfolio_value"]

    @property
    def daily_returns(self) -> pd.Series:
        return self.daily["daily_return"]

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def __repr__(self):
        if self.daily.empty:
            return "BacktestResult(empty)"
        start = self.daily.index[0].date()
        end   = self.daily.index[-1].date()
        total = (self.equity_curve.iloc[-1] / self.cfg.INITIAL_CAPITAL - 1) * 100
        return f"BacktestResult({start} → {end}, return={total:.2f}%)"
