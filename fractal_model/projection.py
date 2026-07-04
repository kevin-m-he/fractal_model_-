# SPDX-License-Identifier: AGPL-3.0-or-later
"""Forward projection from matched motifs.

For each MotifMatch we take the *continuation*: the bars that followed
the historical occurrence, over a horizon proportional to the historical
window length. The continuation is transported to the present:

    time is stretched by the match's time_ratio
    log-price increments are scaled by the fractal law  s_t ** H
      (blended with the match's observed amp_ratio)

The ensemble of transported continuations gives a median projected path
plus dispersion bands. Buy/sell zones fall out of the median path:
buy = deepest early trough of the projection, sell = highest later peak.

Confidence is an explicit, documented composite in [0,1]:
    match quality x ensemble agreement x fractal richness x fit quality
Nothing here is a guarantee; see backtest.py for measured hit rates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .hurst import hurst_summary
from .motif import MotifMatch, find_motifs
from .multifractal import mfdfa


@dataclass
class Projection:
    ticker: str
    live_len: int
    horizon: int
    dates: pd.DatetimeIndex
    median_path: np.ndarray
    lo_band: np.ndarray
    hi_band: np.ndarray
    buy_price: float
    buy_day: pd.Timestamp
    sell_price: float
    sell_day: pd.Timestamp
    expected_return: float
    confidence: float
    n_matches: int
    hurst: dict = field(default_factory=dict)
    delta_alpha: float = float("nan")
    matches: list[MotifMatch] = field(default_factory=list)
    scale_label: str = ""


def _transport_continuation(log_p: np.ndarray, m: MotifMatch, h: float,
                            horizon: int) -> np.ndarray | None:
    """Rescale the bars following a historical match into today's frame."""
    hist_len = m.hist_end - m.hist_start
    cont_len = int(np.ceil(horizon / m.time_ratio))
    if m.hist_end + cont_len > len(log_p) - 1 or cont_len < 2:
        return None
    cont = log_p[m.hist_end - 1: m.hist_end + cont_len]
    inc = np.diff(cont)
    # fractal amplitude law with empirical blend
    law = m.time_ratio ** h if np.isfinite(h) else m.amp_ratio
    amp = float(np.sqrt(max(law, 1e-9) * max(m.amp_ratio, 1e-9)))  # geometric mean
    scaled = inc * amp
    # stretch to horizon in time
    t_old = np.linspace(0.0, 1.0, len(scaled))
    t_new = np.linspace(0.0, 1.0, horizon)
    stretched = np.interp(t_new, t_old, np.cumsum(scaled))
    return stretched  # cumulative log-return path of length `horizon`


def project(ticker: str, close: pd.Series, live_len: int,
            horizon: int | None = None, top_k: int = 8) -> Projection | None:
    if horizon is None:
        horizon = max(20, live_len // 2)
    log_p = np.log(close.values.astype(float))
    hs = hurst_summary(log_p)
    h = hs["H_dfa"]
    mf = mfdfa(log_p)
    matches = find_motifs(close, live_len=live_len, hurst=h, top_k=top_k)
    if not matches:
        return None

    paths, weights = [], []
    for m in matches:
        p = _transport_continuation(log_p, m, h, horizon)
        if p is not None:
            paths.append(p)
            weights.append(m.score)
    if len(paths) < 2:
        return None
    paths = np.vstack(paths)
    weights = np.array(weights)
    weights = weights / weights.sum()

    last_price = float(close.iloc[-1])
    price_paths = last_price * np.exp(paths)
    # weighted median via sorting per column
    order = np.argsort(price_paths, axis=0)
    cum_w = np.cumsum(weights[order], axis=0)
    med_idx = np.argmax(cum_w >= 0.5, axis=0)
    median_path = price_paths[order[med_idx, np.arange(horizon)],
                              np.arange(horizon)]
    lo = np.quantile(price_paths, 0.2, axis=0)
    hi = np.quantile(price_paths, 0.8, axis=0)

    # buy = lowest point in first 60% of horizon on the median path,
    # sell = highest point after the buy day
    cut = max(2, int(horizon * 0.6))
    b_i = int(np.argmin(median_path[:cut]))
    s_rel = int(np.argmax(median_path[b_i:]))
    s_i = b_i + s_rel
    buy_price, sell_price = float(median_path[b_i]), float(median_path[s_i])
    exp_ret = sell_price / buy_price - 1.0

    # ---- confidence composite (each term in [0,1]) ----
    match_q = float(np.average([m.score for m in matches], weights=None))
    spread = float(np.median((hi - lo) / np.maximum(median_path, 1e-9)))
    agreement = float(np.exp(-2.0 * spread))          # tight band -> ~1
    richness = float(np.clip((mf["delta_alpha"] or 0) / 1.0, 0, 1)) \
        if np.isfinite(mf["delta_alpha"]) else 0.4
    fit_q = float(np.clip(hs["dfa_fit_r2"], 0, 1))    # is it even a fractal?
    confidence = float(np.clip(
        0.40 * match_q + 0.25 * agreement + 0.15 * richness + 0.20 * fit_q,
        0, 1))

    freq = pd.infer_freq(close.index[-10:]) or "B"
    dates = pd.date_range(close.index[-1], periods=horizon + 1, freq="B")[1:]

    return Projection(
        ticker=ticker, live_len=live_len, horizon=horizon, dates=dates,
        median_path=median_path, lo_band=lo, hi_band=hi,
        buy_price=buy_price, buy_day=dates[b_i],
        sell_price=sell_price, sell_day=dates[s_i],
        expected_return=exp_ret, confidence=confidence,
        n_matches=len(paths), hurst=hs,
        delta_alpha=mf["delta_alpha"], matches=matches,
    )


# scale ladder used across the app: (label, live window bars, horizon bars)
SCALES = [
    ("short  ~3mo pattern", 63, 42),
    ("medium ~6mo pattern", 126, 84),
    ("long   ~1yr pattern", 252, 168),
    ("macro  ~2yr pattern", 504, 336),
    ("secular ~4yr pattern", 1008, 672),
]


def project_all_scales(ticker: str, close: pd.Series,
                       progress=None) -> list[Projection]:
    """Run projection at every scale that the data length allows and
    return them ranked by confidence (the 'all scales' mode).

    progress, if given, is called as progress(done, total, label) after
    each scale — the app uses it to drive the percent loading bar.
    """
    out = []
    runnable = [s for s in SCALES if len(close) >= s[1] * 2]
    for i, (label, live_len, horizon) in enumerate(runnable):
        if progress is not None:
            progress(i, len(runnable), f"matching motifs — {label.strip()}")
        p = project(ticker, close, live_len=live_len, horizon=horizon)
        if p is not None:
            p.scale_label = label
            out.append(p)
    if progress is not None:
        progress(len(runnable), len(runnable), "ranking scales")
    out.sort(key=lambda p: p.confidence, reverse=True)
    return out
