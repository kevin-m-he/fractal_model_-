# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multifractal detrended fluctuation analysis (MF-DFA).

A monofractal series has a single scaling exponent H. Real markets are
*multifractal*: small moves and large moves scale differently. MF-DFA
(Kantelhardt et al., 2002) computes a generalized exponent h(q) for a
range of moment orders q. The width of the resulting singularity
spectrum, delta_alpha, measures how rich the fractal structure is:

  * delta_alpha ~ 0      -> monofractal / featureless (weak candidate)
  * delta_alpha large    -> strongly multifractal (rich nested structure,
                            i.e. exactly the "patterns inside patterns"
                            visible in the BTC / NFLX reference charts)

The model uses delta_alpha as one input to motif confidence.
"""
from __future__ import annotations

import numpy as np

from .hurst import _log_windows


def mfdfa(x: np.ndarray, q_values: np.ndarray | None = None,
          order: int = 1) -> dict:
    """Return generalized Hurst h(q) and singularity spectrum width.

    x : 1-D log-price array.
    """
    if q_values is None:
        q_values = np.array([-5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5], dtype=float)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 200:
        return {"q": q_values, "h_q": np.full_like(q_values, np.nan),
                "delta_alpha": float("nan")}
    inc = np.diff(x)
    profile = np.cumsum(inc - inc.mean())
    n = len(profile)
    scales = _log_windows(n, min_s=16)

    # segment fluctuations for every scale (both directions to use all data)
    fq = np.zeros((len(scales), len(q_values)))
    for si, s in enumerate(scales):
        n_seg = n // s
        if n_seg < 4:
            fq[si] = np.nan
            continue
        t = np.arange(s)
        segs_f = profile[: n_seg * s].reshape(n_seg, s)
        segs_b = profile[n - n_seg * s:].reshape(n_seg, s)
        segs = np.vstack([segs_f, segs_b])
        coeffs = np.polynomial.polynomial.polyfit(t, segs.T, order)
        trend = np.polynomial.polynomial.polyval(t, coeffs)
        f2 = np.mean((segs - trend) ** 2, axis=1)
        f2 = f2[f2 > 0]
        if len(f2) == 0:
            fq[si] = np.nan
            continue
        for qi, q in enumerate(q_values):
            if abs(q) < 1e-9:
                fq[si, qi] = np.exp(0.5 * np.mean(np.log(f2)))
            else:
                fq[si, qi] = np.mean(f2 ** (q / 2.0)) ** (1.0 / q)

    h_q = np.full(len(q_values), np.nan)
    log_s = np.log(scales)
    for qi in range(len(q_values)):
        col = fq[:, qi]
        mask = np.isfinite(col) & (col > 0)
        if mask.sum() >= 4:
            h_q[qi], _ = np.polyfit(log_s[mask], np.log(col[mask]), 1)

    # singularity spectrum: tau(q) = q*h(q) - 1 ; alpha = d tau / d q
    valid = np.isfinite(h_q)
    delta_alpha = float("nan")
    if valid.sum() >= 4:
        q_v, h_v = q_values[valid], h_q[valid]
        tau = q_v * h_v - 1.0
        alpha = np.gradient(tau, q_v)
        delta_alpha = float(np.nanmax(alpha) - np.nanmin(alpha))
    return {"q": q_values, "h_q": h_q, "delta_alpha": delta_alpha}
