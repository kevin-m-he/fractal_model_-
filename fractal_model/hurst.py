# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hurst exponent estimation.

The Hurst exponent H is the single most important number in this project.
It is the scaling law of the fractal: over a window of length s, typical
price excursion grows like s**H. H = 0.5 is a random walk; H > 0.5 means
persistent (trending, self-reinforcing) structure; H < 0.5 means
anti-persistent (mean-reverting) structure.

Two independent estimators are provided:
  * DFA  — detrended fluctuation analysis (robust to non-stationarity)
  * R/S  — classical rescaled-range (Hurst 1951, Mandelbrot & Wallis 1969)

The model uses DFA as primary and reports both.
"""
from __future__ import annotations

import numpy as np


def _log_windows(n: int, min_s: int = 8, n_scales: int = 20) -> np.ndarray:
    max_s = max(min_s + 1, n // 4)
    scales = np.unique(np.geomspace(min_s, max_s, n_scales).astype(int))
    return scales[scales >= min_s]


def dfa_hurst(x: np.ndarray, order: int = 1) -> tuple[float, float]:
    """DFA-`order` Hurst estimate on increments of log-price series x.

    Parameters
    ----------
    x : 1-D array of log prices (the *profile* is built internally).

    Returns
    -------
    (H, r2) : estimated exponent and the R^2 of the log-log fit
              (fit quality = how cleanly the series obeys a power law,
               i.e. how "fractal" it actually is).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 64:
        return float("nan"), 0.0
    inc = np.diff(x)
    profile = np.cumsum(inc - inc.mean())
    n = len(profile)
    scales = _log_windows(n)
    flucts = []
    for s in scales:
        n_seg = n // s
        if n_seg < 2:
            continue
        segs = profile[: n_seg * s].reshape(n_seg, s)
        t = np.arange(s)
        # vectorized polynomial detrend per segment
        coeffs = np.polynomial.polynomial.polyfit(t, segs.T, order)
        trend = np.polynomial.polynomial.polyval(t, coeffs)
        f2 = np.mean((segs - trend) ** 2, axis=1)
        flucts.append((s, np.sqrt(np.mean(f2))))
    if len(flucts) < 4:
        return float("nan"), 0.0
    s_arr = np.log([f[0] for f in flucts])
    f_arr = np.log([f[1] for f in flucts])
    slope, intercept = np.polyfit(s_arr, f_arr, 1)
    pred = slope * s_arr + intercept
    ss_res = np.sum((f_arr - pred) ** 2)
    ss_tot = np.sum((f_arr - f_arr.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(max(0.0, r2))


def rs_hurst(x: np.ndarray) -> float:
    """Classical rescaled-range Hurst estimate on log-price series x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 64:
        return float("nan")
    inc = np.diff(x)
    n = len(inc)
    scales = _log_windows(n, min_s=10)
    out = []
    for s in scales:
        n_seg = n // s
        if n_seg < 1:
            continue
        rs_vals = []
        for i in range(n_seg):
            seg = inc[i * s: (i + 1) * s]
            z = np.cumsum(seg - seg.mean())
            r = z.max() - z.min()
            sd = seg.std(ddof=1)
            if sd > 0:
                rs_vals.append(r / sd)
        if rs_vals:
            out.append((s, np.mean(rs_vals)))
    if len(out) < 4:
        return float("nan")
    slope, _ = np.polyfit(np.log([o[0] for o in out]), np.log([o[1] for o in out]), 1)
    return float(slope)


def hurst_summary(log_close: np.ndarray) -> dict:
    h_dfa, r2 = dfa_hurst(log_close)
    h_rs = rs_hurst(log_close)
    regime = ("persistent" if h_dfa > 0.55 else
              "anti-persistent" if h_dfa < 0.45 else "near-random-walk")
    return {"H_dfa": h_dfa, "dfa_fit_r2": r2, "H_rs": h_rs, "regime": regime}
