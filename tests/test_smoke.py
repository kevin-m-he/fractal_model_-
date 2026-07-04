# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline smoke tests using synthetic data (no network)."""
import numpy as np
import pandas as pd
from fractal_model.hurst import dfa_hurst, hurst_summary
from fractal_model.motif import to_shape
from fractal_model.projection import project_all_scales


def test_dfa_on_random_walk_near_half():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(4000)) * 0.01
    H, r2 = dfa_hurst(x)
    assert 0.35 < H < 0.65, H
    assert r2 > 0.9


def test_shape_space_scale_invariant():
    base = np.log(np.linspace(10, 20, 100))
    big = np.log(np.linspace(100, 400, 300))
    s1, _ = to_shape(base); s2, _ = to_shape(big)
    assert np.corrcoef(s1, s2)[0, 1] > 0.99


def test_pipeline_runs_on_synthetic():
    n = 2000
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01)),
                      index=pd.bdate_range("2010-01-01", periods=n))
    projs = project_all_scales("SYN", close)
    assert isinstance(projs, list)


def test_hurst_summary_keys():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.standard_normal(2000)) * 0.01
    s = hurst_summary(x)
    assert {"H_dfa", "dfa_fit_r2", "H_rs", "regime"} <= set(s)
