# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data acquisition layer.

Primary source: Yahoo Finance via yfinance. Fallback: Stooq daily CSV.
All series are cached locally (parquet) so the app works offline after
first fetch and so backtests are reproducible.
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(os.environ.get("FRACTAL_CACHE", Path.home() / ".fractal_model_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 6 * 3600  # refresh cached data every 6 hours

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(ticker: str) -> Path:
    safe = ticker.upper().replace("/", "_").replace("^", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _from_cache(ticker: str, max_age: float = CACHE_TTL_SECONDS) -> pd.DataFrame | None:
    p = _cache_path(ticker)
    if p.exists() and (time.time() - p.stat().st_mtime) < max_age:
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


def _to_cache(ticker: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(ticker))
    except Exception:
        pass  # cache is best-effort


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # yfinance sometimes returns MultiIndex columns for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in REQUIRED_COLS if c in df.columns]]
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _fetch_yfinance(ticker: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        df = yf.download(ticker, period="max", interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        if df is not None and len(df) > 0:
            return _clean(df)
    except Exception:
        pass
    return None


def _fetch_stooq(ticker: str) -> pd.DataFrame | None:
    """Stooq fallback. US equities use e.g. 'nflx.us'; BTC-USD -> 'btcusd'."""
    t = ticker.lower()
    if t.endswith("-usd"):
        symbol = t.replace("-usd", "usd")
    elif "." in t or "^" in t:
        symbol = t.replace("^", "")
    else:
        symbol = f"{t}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200 or "Date" not in r.text[:200]:
            return None
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
        if len(df) == 0:
            return None
        return _clean(df)
    except Exception:
        return None


def get_history(ticker: str, min_rows: int = 250) -> pd.DataFrame:
    """Return daily OHLCV history for ticker, longest available.

    Raises ValueError if no source yields at least `min_rows` rows.
    """
    cached = _from_cache(ticker)
    if cached is not None and len(cached) >= min_rows:
        return cached

    for fetch in (_fetch_yfinance, _fetch_stooq):
        df = fetch(ticker)
        if df is not None and len(df) >= min_rows:
            _to_cache(ticker, df)
            return df

    # last resort: stale cache of any age
    stale = _from_cache(ticker, max_age=float("inf"))
    if stale is not None and len(stale) >= min_rows:
        return stale
    raise ValueError(
        f"Could not fetch at least {min_rows} daily bars for '{ticker}' "
        f"from Yahoo Finance or Stooq."
    )
