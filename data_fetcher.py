"""
Data Fetcher — baostock-based A-share market data pipeline.

Provides:
  - Stock universe (all A-share stocks)
  - Daily OHLCV + turnover data with local parquet caching
  - Index daily data (benchmark)
  - Incremental updates (only downloads missing date ranges)

Usage:
    fetcher = DataFetcher(config)
    fetcher.download_universe()              # download all stocks
    panel = fetcher.load_panel_data()        # returns MultiIndex DataFrame
    benchmark = fetcher.load_benchmark()     # CSI 300 returns
"""

import os
import logging
import warnings
from datetime import datetime, timedelta

import baostock as bs
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import Config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


DAILY_FIELDS = (
    "date,code,open,high,low,close,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,isST"
)


class DataFetcher:
    """Download and cache A-share daily data via baostock."""

    def __init__(self, config: Config = None):
        self.cfg = config or Config()
        self.cfg.ensure_dirs()
        self._logged_in = False

    # ─── baostock session ────────────────────────────────────────────────────

    def _login(self):
        if not self._logged_in:
            result = bs.login()
            if result.error_code != "0":
                raise RuntimeError(f"baostock login failed: {result.error_msg}")
            self._logged_in = True

    def _logout(self):
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def __enter__(self):
        self._login()
        return self

    def __exit__(self, *args):
        self._logout()

    # ─── Universe ────────────────────────────────────────────────────────────

    def get_stock_list(self, date: str = None) -> list[str]:
        """Return baostock stock codes for all A-share stocks on *date*."""
        self._login()
        date = date or self.cfg.TRAIN_END
        rs = bs.query_all_stock(day=date)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        # Keep only SH/SZ A-shares (6xxxxx SH or 0/3xxxxx SZ), exclude index/fund codes
        mask = (
            df["code"].str.startswith(("sh.6", "sz.0", "sz.3"))
            & ~df["code"].str.startswith("sh.688")   # keep STAR market too
        )
        # Actually include STAR (sh.688xxx) — they are A-shares
        mask = df["code"].str.match(r"^(sh\.6|sz\.0|sz\.3|sh\.688)")
        return df.loc[mask, "code"].tolist()

    # ─── Single-stock download ────────────────────────────────────────────────

    def _cache_path(self, code: str) -> str:
        return os.path.join(self.cfg.CACHE_DIR, f"{code.replace('.', '_')}.parquet")

    def _fetch_one(self, code: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily bars for one stock from baostock."""
        rs = bs.query_history_k_data_plus(
            code, DAILY_FIELDS,
            start_date=start, end_date=end,
            frequency="d", adjustflag="2"   # qfq (forward-adjusted)
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=rs.fields)
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = ["open", "high", "low", "close", "volume", "amount",
                        "turn", "pctChg"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["isST"] = df["isST"].astype(str).str.strip()
        df["tradestatus"] = df["tradestatus"].astype(str).str.strip()
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df

    def _load_cached(self, code: str) -> pd.DataFrame:
        path = self._cache_path(code)
        if os.path.exists(path):
            return pd.read_parquet(path)
        return pd.DataFrame()

    def _save_cached(self, code: str, df: pd.DataFrame):
        if df.empty:
            return
        df.to_parquet(self._cache_path(code))

    def download_stock(self, code: str, start: str = None, end: str = None) -> pd.DataFrame:
        """Download (with incremental caching) daily data for one stock."""
        self._login()
        start = start or self.cfg.DATA_START
        end   = end   or self.cfg.TEST_END

        cached = self._load_cached(code)
        if cached.empty:
            new_data = self._fetch_one(code, start, end)
        else:
            last_cached = cached.index.max().strftime("%Y-%m-%d")
            if last_cached >= end:
                return cached
            next_day = (cached.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
            new_data = self._fetch_one(code, next_day, end)
            if not new_data.empty:
                new_data = pd.concat([cached, new_data])
                new_data = new_data[~new_data.index.duplicated(keep="last")]
                new_data.sort_index(inplace=True)
            else:
                new_data = cached

        self._save_cached(code, new_data)
        return new_data

    # ─── Bulk download ───────────────────────────────────────────────────────

    def download_universe(self, codes: list[str] = None):
        """Bulk-download all A-share stocks with progress bar."""
        self._login()
        if codes is None:
            log.info("Fetching stock universe …")
            codes = self.get_stock_list()
            log.info(f"  → {len(codes)} stocks found")

        failed = []
        for code in tqdm(codes, desc="Downloading stocks", ncols=80):
            try:
                self.download_stock(code)
            except Exception as e:
                failed.append((code, str(e)))

        if failed:
            log.warning(f"{len(failed)} stocks failed: {failed[:5]} …")
        log.info("Download complete.")

    # ─── Load as panel ───────────────────────────────────────────────────────

    def load_panel_data(
        self,
        start: str = None,
        end: str = None,
        min_close: float = None,
        min_amount: float = None,
    ) -> pd.DataFrame:
        """
        Load all cached stock data and return a stacked panel DataFrame
        indexed by (date, code).
        """
        start  = start      or self.cfg.DATA_START
        end    = end        or self.cfg.TEST_END
        min_close  = min_close  or self.cfg.MIN_PRICE
        min_amount = min_amount or self.cfg.MIN_AMOUNT

        frames = []
        cache_files = [f for f in os.listdir(self.cfg.CACHE_DIR)
                       if f.endswith(".parquet")]
        if not cache_files:
            raise FileNotFoundError(
                "No cached data found. Run download_universe() first."
            )

        for fname in tqdm(cache_files, desc="Loading panel", ncols=80):
            code = fname.replace(".parquet", "").replace("_", ".", 1)
            path = os.path.join(self.cfg.CACHE_DIR, fname)
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if df.empty:
                continue
            df = df[(df.index >= pd.Timestamp(start)) &
                    (df.index <= pd.Timestamp(end))].copy()
            if df.empty:
                continue
            df["code"] = code
            frames.append(df)

        if not frames:
            raise ValueError("No data loaded — check cache directory.")

        panel = pd.concat(frames)
        panel.index.name = "date"
        panel = panel.reset_index().set_index(["date", "code"]).sort_index()

        # Basic quality filters
        panel = panel[panel["tradestatus"] == "1"]          # only trading days
        panel = panel[panel["close"] >= min_close]          # price filter
        panel = panel[panel["amount"] >= min_amount]        # liquidity filter
        if self.cfg.EXCLUDE_ST:
            panel = panel[panel["isST"] != "1"]             # exclude ST

        log.info(f"Panel loaded: {panel.shape[0]:,} rows, "
                 f"{panel.index.get_level_values('code').nunique():,} stocks, "
                 f"dates {panel.index.get_level_values('date').min().date()} – "
                 f"{panel.index.get_level_values('date').max().date()}")
        return panel

    # ─── Benchmark ───────────────────────────────────────────────────────────

    def load_benchmark(self, start: str = None, end: str = None) -> pd.Series:
        """Return CSI-300 daily close-to-close returns as a Series."""
        self._login()
        start = start or self.cfg.DATA_START
        end   = end   or self.cfg.TEST_END

        cache_path = os.path.join(self.cfg.CACHE_DIR, "benchmark_csi300.parquet")
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            last = df.index.max().strftime("%Y-%m-%d")
            if last < end:
                new_end = end
                rs = bs.query_history_k_data_plus(
                    self.cfg.BENCHMARK_CODE,
                    "date,close",
                    start_date=(df.index.max() + timedelta(days=1)).strftime("%Y-%m-%d"),
                    end_date=new_end, frequency="d", adjustflag="3"
                )
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    new_df = pd.DataFrame(rows, columns=rs.fields)
                    new_df["date"] = pd.to_datetime(new_df["date"])
                    new_df["close"] = pd.to_numeric(new_df["close"], errors="coerce")
                    new_df.set_index("date", inplace=True)
                    df = pd.concat([df, new_df]).sort_index()
                    df.to_parquet(cache_path)
        else:
            rs = bs.query_history_k_data_plus(
                self.cfg.BENCHMARK_CODE,
                "date,close",
                start_date=start, end_date=end,
                frequency="d", adjustflag="3"
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df.set_index("date", inplace=True)
            df.to_parquet(cache_path)

        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        returns = df["close"].pct_change().rename("benchmark")
        return returns

    # ─── Trading calendar ────────────────────────────────────────────────────

    def get_trading_dates(self, start: str = None, end: str = None) -> list:
        """Return sorted list of A-share trading dates in range."""
        self._login()
        start = start or self.cfg.DATA_START
        end   = end   or self.cfg.TEST_END
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        df = df[df["is_trading_day"] == "1"]
        return sorted(pd.to_datetime(df["calendar_date"]).tolist())
