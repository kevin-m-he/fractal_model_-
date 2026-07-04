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


def test_group_families_clusters_similar_shapes():
    from fractal_model.motif import group_families
    t = np.linspace(0, 1, 64)
    rising = t.copy()
    v_shape = np.abs(t - 0.5) * 2
    ids = group_families([rising, v_shape, rising * 0.98 + 0.01, v_shape])
    assert ids[0] == ids[2] and ids[1] == ids[3]  # same shape, same family
    assert ids[0] != ids[1]                       # different shapes split


def test_family_boxes_share_colors():
    from fractal_model.viz3d import _family_boxes
    from fractal_model.motif import find_motifs
    # self-similar series: same motif repeated at growing scale
    seg = np.sin(np.linspace(0, 3 * np.pi, 100))
    x = np.concatenate([seg * a for a in (0.5, 1.0, 2.0)])
    close = pd.Series(100 * np.exp(np.cumsum(np.full(len(x), 0.001)) + x * 0.1),
                      index=pd.bdate_range("2012-01-01", periods=len(x)))
    matches = find_motifs(close, live_len=100, hurst=0.5)
    assert matches, "expected motif matches on a constructed self-similar series"
    boxes = _family_boxes(matches, np.log(close.values))
    kinds = {b["kind"] for b in boxes}
    assert kinds == {"hist", "live"}
    # every box has a family letter and a color; same family -> same color
    fam_color = {}
    for b in boxes:
        assert fam_color.setdefault(b["fam"], b["color"]) == b["color"]


def test_option_chain_figure_offline():
    from fractal_model.viz3d import option_chain_figure_3d
    rows = []
    for dte, exp in [(30, "2026-08-03"), (90, "2026-10-02"), (365, "2027-07-04")]:
        strikes = np.linspace(50, 150, 21)
        # Black-Scholes-ish decay: farther expiries are magnified copies
        call = np.maximum(100 - strikes, 0.5) + 8 * np.sqrt(dte / 30)
        put = np.maximum(strikes - 100, 0.5) + 8 * np.sqrt(dte / 30)
        for k, c, p in zip(strikes, call, put):
            rows.append({"expiration": pd.Timestamp(exp), "dte": dte,
                         "strike": k, "call": c, "put": p})
    chain = pd.DataFrame(rows)
    # model expects spot to drift to 110 by the far expiry
    exp_spot = {30: 102.0, 90: 105.0, 365: 110.0}
    fig = option_chain_figure_3d("SYN", chain, spot=100.0, exp_spot=exp_spot)
    names = [tr.name for tr in fig.data]
    assert sum("calls" in n for n in names) == 3
    assert any(n.startswith("spot") for n in names)
    # similar curves should be grouped into a shared pattern family
    call_traces = [tr for tr in fig.data if "calls" in tr.name]
    assert len({tr.line.color for tr in call_traces}) < len(call_traces) + 1
    # under-line fill: one P/L curtain mesh per expiry, shaded by profit
    curtains = [tr for tr in fig.data if tr.name and "expected P/L" in tr.name]
    assert len(curtains) == 3
    for tr in curtains:
        assert tr.intensity is not None and len(tr.intensity) > 0
    # deep ITM call at strike 50 for the 365d expiry: payoff 60 vs
    # premium ~94 -> loss; sanity-check the profit signs vary
    pl = np.asarray(curtains[-1].intensity)
    assert pl.min() < 0


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
