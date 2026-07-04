# SPDX-License-Identifier: AGPL-3.0-or-later
"""Universe scanner: ranks tickers by best fractal-projection confidence
and emits the Top-10 'following fractals' table with buy price, sell
target, timeframe, and confidence."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .data import get_history
from .projection import project_all_scales

DEFAULT_UNIVERSE = [
    # large-cap tech & growth
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "AMD", "AVGO", "CRM", "ADBE", "ORCL", "INTC", "QCOM", "SHOP",
    # financials / industrials / consumer
    "JPM", "BAC", "GS", "V", "MA", "WMT", "COST", "HD", "MCD", "NKE",
    "DIS", "BA", "CAT", "XOM", "CVX", "UNH", "JNJ", "PFE", "LLY", "KO",
    # crypto & indices via Yahoo symbols
    "BTC-USD", "ETH-USD", "SPY", "QQQ", "IWM",
]


def scan_ticker(ticker: str) -> dict | None:
    try:
        df = get_history(ticker)
        projs = project_all_scales(ticker, df["Close"])
        if not projs:
            return None
        best = projs[0]
        return {
            "ticker": ticker,
            "last": float(df["Close"].iloc[-1]),
            "scale": best.scale_label,
            "buy_price": best.buy_price,
            "buy_by": best.buy_day.date(),
            "sell_target": best.sell_price,
            "sell_by": best.sell_day.date(),
            "exp_return": best.expected_return,
            "confidence": best.confidence,
            "H": best.hurst.get("H_dfa"),
            "n_motifs": best.n_matches,
        }
    except Exception:
        return None


def top_following_fractals(universe: list[str] | None = None,
                           n: int = 10, workers: int = 6) -> pd.DataFrame:
    universe = universe or DEFAULT_UNIVERSE
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(scan_ticker, t): t for t in universe}
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("confidence", ascending=False)
    return df.head(n).reset_index(drop=True)
