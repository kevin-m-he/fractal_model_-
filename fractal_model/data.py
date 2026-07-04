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
SHARES_TTL_SECONDS = 7 * 24 * 3600  # share counts move on buybacks/dilution, not daily

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _safe_name(ticker: str) -> str:
    return ticker.upper().replace("/", "_").replace("^", "_")


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{_safe_name(ticker)}.parquet"


def _shares_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{_safe_name(ticker)}_shares.parquet"


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


def _split_adjust(s: pd.Series, tk) -> pd.Series:
    """Rescale historical share counts to today's split basis.

    Yahoo reports raw share counts as filed, but prices come back
    split-adjusted — without this, a 10:1 split shows as a fake 10x
    cliff in the shares axis and understates pre-split market cap.
    """
    try:
        splits = tk.splits
        if splits is not None and len(splits) > 0:
            splits = splits[splits > 0]
            splits.index = pd.to_datetime(splits.index).tz_localize(None).normalize()
            for dt, ratio in splits.items():
                s.loc[s.index < dt] *= float(ratio)
    except Exception:
        pass
    return s


def _fetch_shares_series(ticker: str) -> pd.Series | None:
    """Historical shares-outstanding series from Yahoo (usually ~2y of
    filings-derived points, normalized to the current split basis),
    falling back to the current count as a single-point series. Crypto
    uses circulating supply."""
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        try:
            s = tk.get_shares_full(start="1900-01-01")
        except Exception:
            s = None
        if s is not None and len(s) > 0:
            s = s.astype(float)
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[s > 0]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if len(s) > 0:
                return _split_adjust(s, tk)
        info = tk.get_info()
        for key in ("sharesOutstanding", "impliedSharesOutstanding",
                    "circulatingSupply"):
            v = info.get(key)
            if v:
                return pd.Series([float(v)], index=[pd.Timestamp.now().normalize()])
    except Exception:
        pass
    return None


def get_shares(ticker: str, index: pd.DatetimeIndex) -> pd.Series | None:
    """Shares outstanding aligned to `index` (one value per bar), or None.

    Values between reported counts are forward-filled; dates before the
    earliest report are back-filled with it (Yahoo's share history rarely
    reaches back further than a couple of years). Cached like prices, but
    with a weekly TTL since counts only move on buybacks and dilution.
    """
    s: pd.Series | None = None
    p = _shares_cache_path(ticker)
    if p.exists() and (time.time() - p.stat().st_mtime) < SHARES_TTL_SECONDS:
        try:
            s = pd.read_parquet(p)["Shares"]
        except Exception:
            s = None
    if s is None:
        s = _fetch_shares_series(ticker)
        if s is not None:
            try:
                s.rename("Shares").to_frame().to_parquet(p)
            except Exception:
                pass  # cache is best-effort
        elif p.exists():  # stale cache beats nothing
            try:
                s = pd.read_parquet(p)["Shares"]
            except Exception:
                s = None
    if s is None or len(s) == 0:
        return None
    aligned = s.reindex(s.index.union(index)).ffill().bfill().reindex(index)
    return aligned.astype(float)
