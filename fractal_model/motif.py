# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scale-invariant motif detection — the heart of the fractal model.

Definition used throughout this project (per project owner): a *fractal*
is a price pattern that recurs at a magnified scale over time — larger
amplitude, longer duration, same shape. The reference BTC / NFLX charts
show these as nested colored boxes: the pattern inside the small box
reappears, blown up, inside the big box.

Algorithm
---------
1. Take the most recent window of length L_r (the "live" pattern).
2. Normalize it to shape space:  log price -> min-max [0,1],
   time -> resampled to K points. Shape space erases scale, so a
   2015 pattern at $10 over 90 days and a 2024 pattern at $600 over
   500 days become directly comparable curves.
3. Slide historical candidate windows of *different* lengths L_h across
   the past (candidates must end before the live window starts).
4. Score each candidate:
     shape_sim   : Pearson correlation + normalized RMSE in shape space
     hurst_consistency : a true self-affine fractal must obey the series'
         own scaling law  amplitude_ratio ~ time_ratio ** H.
         We penalize matches whose (s_a, s_t) pair violates the measured
         Hurst exponent — this is what separates "fractal recurrence"
         from coincidental shape lookalikes.
5. Return top-k matches with their scale ratios; projection.py replays
   what followed each match, rescaled, to form the forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SHAPE_POINTS = 64  # resolution of shape space


@dataclass
class MotifMatch:
    hist_start: int          # integer index into series
    hist_end: int            # exclusive
    live_start: int
    live_end: int
    time_ratio: float        # L_live / L_hist  (>1 => magnified in time)
    amp_ratio: float         # live log-range / hist log-range
    shape_corr: float
    shape_rmse: float
    hurst_consistency: float # in [0,1], 1 = perfectly obeys s_a = s_t^H
    score: float
    dates: dict = field(default_factory=dict)
    hist_shape: np.ndarray | None = None  # shape-space curve of the hist window
    live_shape: np.ndarray | None = None  # shape-space curve of the live window


def to_shape(log_p: np.ndarray, k: int = SHAPE_POINTS) -> tuple[np.ndarray, float]:
    """Min-max normalize a log-price window and resample to k points.

    Returns (shape, log_range). log_range is the amplitude in log space.
    """
    log_p = np.asarray(log_p, dtype=float)
    rng = float(log_p.max() - log_p.min())
    if rng <= 0:
        return np.zeros(k), 0.0
    u = (log_p - log_p.min()) / rng
    t_old = np.linspace(0.0, 1.0, len(u))
    t_new = np.linspace(0.0, 1.0, k)
    return np.interp(t_new, t_old, u), rng


def _shape_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.std() == 0 or b.std() == 0:
        return 0.0, 1.0
    corr = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))  # both already in [0,1]
    return corr, rmse


def _hurst_consistency(time_ratio: float, amp_ratio: float, h: float) -> float:
    """1.0 when amp_ratio == time_ratio**H exactly, decaying with the
    log-space distance from the scaling law. Tolerance of one natural-log
    unit (~e-fold) gives ~0.37 consistency."""
    if not np.isfinite(h) or time_ratio <= 0 or amp_ratio <= 0:
        return 0.5  # neutral when H unknown
    expected = time_ratio ** h
    d = abs(np.log(amp_ratio / expected))
    return float(np.exp(-d))


def group_families(shapes: list[np.ndarray], min_corr: float = 0.85) -> list[int]:
    """Cluster shape-space curves into pattern families.

    Greedy leader clustering: each shape joins the first family whose
    leader it correlates with at >= min_corr, else founds a new family.
    Returns a family id per input shape (order-stable, so pass shapes in
    descending match-score order and leaders are the strongest examples).
    Everything in one family is 'the same fractal' for display purposes —
    the UI gives each family one color.
    """
    leaders: list[np.ndarray] = []
    ids: list[int] = []
    for s in shapes:
        fid = None
        for i, leader in enumerate(leaders):
            corr, _ = _shape_similarity(leader, s)
            if corr >= min_corr:
                fid = i
                break
        if fid is None:
            leaders.append(s)
            fid = len(leaders) - 1
        ids.append(fid)
    return ids


def find_motifs(
    close: pd.Series,
    live_len: int,
    hist_lens: list[int] | None = None,
    top_k: int = 8,
    stride_frac: float = 0.1,
    hurst: float | None = None,
    min_corr: float = 0.60,
) -> list[MotifMatch]:
    """Find historical windows whose shape matches the last `live_len` bars.

    hist_lens defaults to a geometric ladder from live_len/6 up to
    live_len (fractals magnify: past occurrences are usually *shorter*).
    """
    log_p = np.log(close.values.astype(float))
    n = len(log_p)
    if n < live_len + 40:
        return []
    live_start, live_end = n - live_len, n
    live_shape, live_rng = to_shape(log_p[live_start:live_end])
    if live_rng <= 0:
        return []

    if hist_lens is None:
        lo = max(20, live_len // 6)
        hist_lens = sorted({int(x) for x in np.geomspace(lo, live_len, 6)})

    matches: list[MotifMatch] = []
    for L in hist_lens:
        stride = max(1, int(L * stride_frac))
        # candidate must end at or before the live window begins
        for start in range(0, live_start - L + 1, stride):
            end = start + L
            cand = log_p[start:end]
            shape, rng = to_shape(cand)
            if rng <= 0:
                continue
            corr, rmse = _shape_similarity(live_shape, shape)
            if corr < min_corr:
                continue
            t_ratio = live_len / L
            a_ratio = live_rng / rng
            hc = _hurst_consistency(t_ratio, a_ratio, hurst)
            score = (0.55 * max(corr, 0.0)
                     + 0.25 * (1.0 - min(rmse / 0.5, 1.0))
                     + 0.20 * hc)
            matches.append(MotifMatch(
                hist_start=start, hist_end=end,
                live_start=live_start, live_end=live_end,
                time_ratio=t_ratio, amp_ratio=a_ratio,
                shape_corr=corr, shape_rmse=rmse,
                hurst_consistency=hc, score=score,
                dates={
                    "hist_start": close.index[start],
                    "hist_end": close.index[end - 1],
                    "live_start": close.index[live_start],
                    "live_end": close.index[-1],
                },
                hist_shape=shape, live_shape=live_shape,
            ))

    matches.sort(key=lambda m: m.score, reverse=True)
    # non-maximum suppression: drop matches overlapping a better one >50%
    kept: list[MotifMatch] = []
    for m in matches:
        clash = False
        for k in kept:
            inter = min(m.hist_end, k.hist_end) - max(m.hist_start, k.hist_start)
            if inter > 0.5 * (m.hist_end - m.hist_start):
                clash = True
                break
        if not clash:
            kept.append(m)
        if len(kept) >= top_k:
            break
    return kept
