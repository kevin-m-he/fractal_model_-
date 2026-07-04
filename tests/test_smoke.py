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


def _synthetic_ohlcv(n=800, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    return pd.DataFrame({"Open": close, "High": close * 1.01,
                         "Low": close * 0.99, "Close": close,
                         "Volume": rng.integers(1e6, 5e6, n)}, index=idx)


def test_viz3d_shares_axis():
    from fractal_model.viz3d import fractal_figure_3d
    df = _synthetic_ohlcv()
    # step function: a buyback partway through, like a real share count
    shares = pd.Series(np.where(np.arange(len(df)) < 400, 5e8, 4.6e8),
                       index=df.index)
    fig = fractal_figure_3d("SYN", df, shares=shares)
    path = fig.data[0]
    assert np.isclose(path.y[0], np.log10(5e8))
    assert np.isclose(path.y[-1], np.log10(4.6e8))
    assert fig.layout.scene.yaxis.title.text == "log10 shares outstanding"
    # market cap in hover data: close * shares formatted with B suffix
    assert "B" in path.customdata[0][3]


def test_viz3d_without_shares_falls_back_flat():
    from fractal_model.viz3d import fractal_figure_3d
    df = _synthetic_ohlcv(seed=6)
    fig = fractal_figure_3d("SYN", df, shares=None)
    assert np.allclose(fig.data[0].y, 0.0)  # log10(1) plane
    assert "no data" in fig.layout.scene.yaxis.title.text


def test_get_shares_alignment_no_network(tmp_path, monkeypatch):
    import fractal_model.data as data
    idx = pd.bdate_range("2020-01-01", periods=100)
    reported = pd.Series([1e9, 1.1e9],
                         index=[idx[30], idx[70]]).rename("Shares")
    monkeypatch.setattr(data, "_fetch_shares_series", lambda t: reported)
    monkeypatch.setattr(data, "_shares_cache_path",
                        lambda t: tmp_path / "x_shares.parquet")
    s = data.get_shares("SYN", idx)
    assert len(s) == len(idx)
    assert s.iloc[0] == 1e9      # back-filled before first report
    assert s.iloc[50] == 1e9     # forward-filled between reports
    assert s.iloc[-1] == 1.1e9   # latest count carried forward
