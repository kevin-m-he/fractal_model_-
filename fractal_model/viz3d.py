# SPDX-License-Identifier: AGPL-3.0-or-later
"""3D fractal rendering: time x shares outstanding x price.

The price curve is drawn as a 3D ribbon through (t, log shares, log
price). Because log(market cap) = log(price) + log(shares), the ribbon's
height off the time axis reads as company valuation: buybacks pull the
path toward the viewer, dilution pushes it away, and the diagonal
(z + y) is log market cap. Detected motifs are drawn as translucent 3D
boxes spanning the time/price extent of each occurrence, with box depth
spanning the share-count range inside the window. Matched pairs
(historical occurrence <-> live pattern) share a color from the
annotation palette.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .motif import MotifMatch, group_families, to_shape
from .projection import Projection

# palette lifted from the project owner's hand-annotated reference charts
BOX_COLORS = ["#E8C400", "#2E9E5B", "#9B4FC0", "#8B1A2B",
              "#E07B39", "#3D7DD8", "#C23B80", "#5BA8A0"]
BG = "#0E1220"
GRID = "#232A40"
TRACE = "#7FB4E6"
PROJ = "#F2E9DC"


def _box_mesh(x0, x1, y0, y1, z0, z1, color: str, name: str,
              opacity: float = 0.16) -> go.Mesh3d:
    xs = [x0, x0, x1, x1, x0, x0, x1, x1]
    ys = [y0, y1, y1, y0, y0, y1, y1, y0]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color,
                     opacity=opacity, name=name, hoverinfo="name",
                     showlegend=True, flatshading=True)


def _family_boxes(matches: list[MotifMatch], log_p: np.ndarray,
                  max_matches: int = 24) -> list[dict]:
    """Assign every motif occurrence a pattern family so same shape =
    same color everywhere.

    Historical windows and live windows are clustered together in shape
    space (greedy leader clustering on the stored shape curves), so the
    user can read correspondence off the colors: every box sharing a
    color is the same recurring fractal, and the translucent box of that
    color is the live pattern it refers to. Returns dicts with keys
    a, b (index span), kind ('hist'|'live'), fam (letter), color, m.
    """
    ms = matches[:max_matches]
    items = []  # (shape, a, b, kind, match)
    for m in ms:
        shape = (m.hist_shape if m.hist_shape is not None
                 else to_shape(log_p[m.hist_start:m.hist_end])[0])
        items.append((shape, m.hist_start, m.hist_end, "hist", m))
    seen_live = set()
    for m in ms:  # one live item per distinct live window (scale)
        key = (m.live_start, m.live_end)
        if key in seen_live:
            continue
        seen_live.add(key)
        shape = (m.live_shape if m.live_shape is not None
                 else to_shape(log_p[m.live_start:m.live_end])[0])
        items.append((shape, m.live_start, m.live_end, "live", m))
    fams = group_families([it[0] for it in items])
    boxes = []
    for (shape, a, b, kind, m), f in zip(items, fams):
        boxes.append(dict(a=a, b=b, kind=kind,
                          fam=chr(ord("A") + f % 26),
                          color=BOX_COLORS[f % len(BOX_COLORS)], m=m))
    return boxes


def _fmt_cap(x: float) -> str:
    for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
        if abs(x) >= div:
            return f"${x / div:,.2f}{suf}"
    return f"${x:,.0f}"


def fractal_figure_3d(ticker: str, df: pd.DataFrame,
                      matches: list[MotifMatch] | None = None,
                      projection: Projection | None = None,
                      shares: pd.Series | None = None) -> go.Figure:
    close = df["Close"].values.astype(float)
    if shares is not None:
        sh = shares.values.astype(float)
        sh = np.where(np.isfinite(sh) & (sh > 0), sh, np.nan)
        sh = pd.Series(sh).ffill().bfill().values
        if not np.isfinite(sh).all():
            shares = None
    if shares is None:  # no share data (index, delisted, …): flat valuation axis
        sh = np.ones(len(df))
        y_title = "log10 shares (no data)"
    else:
        y_title = "log10 shares outstanding"
    log_s = np.log10(sh)
    log_p = np.log10(close)
    mcap = close * sh
    t = np.arange(len(df))
    dates = df.index

    val_lines = ("<br>shares %{customdata[2]:,.0f}<br>mkt cap %{customdata[3]}"
                 if shares is not None else "")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=t, y=log_s, z=log_p, mode="lines",
        line=dict(color=TRACE, width=3.5),
        name=f"{ticker} valuation path",
        customdata=np.stack([dates.strftime("%Y-%m-%d"), close, sh,
                             [_fmt_cap(c) for c in mcap]], axis=-1),
        hovertemplate="%{customdata[0]}<br>price $%{customdata[1]:,.2f}"
                      f"{val_lines}<extra></extra>",
    ))

    if matches:
        boxes = _family_boxes(matches, np.log(close))
        legend_done = set()
        for box in boxes:
            a, b, m = box["a"], box["b"], box["m"]
            z0, z1 = log_p[a:b].min(), log_p[a:b].max()
            y0, y1 = np.nanmin(log_s[a:b]), np.nanmax(log_s[a:b])
            # share counts barely move inside a window — give the box a
            # visible minimum depth so it doesn't collapse to a plane
            pad_y = max(0.012, 0.05 * (y1 - y0))
            y0, y1 = y0 - pad_y, y1 + pad_y
            pad_z = 0.02 * (z1 - z0 + 1e-6)
            if box["kind"] == "hist":
                label = (f"pattern {box['fam']} · {dates[a].year}–"
                         f"{dates[b-1].year} (x{m.time_ratio:.1f} time, "
                         f"x{m.amp_ratio:.1f} amp)")
                op = 0.18
            else:
                label = f"pattern {box['fam']} · live window"
                op = 0.09
            mesh = _box_mesh(a, b, y0, y1, z0 - pad_z, z1 + pad_z,
                             box["color"], label, opacity=op)
            # one legend entry per family; hover still shows per-box detail
            mesh.legendgroup = f"fam-{box['fam']}"
            mesh.showlegend = box["fam"] not in legend_done
            legend_done.add(box["fam"])
            fig.add_trace(mesh)

    if projection is not None:
        t_f = np.arange(len(df), len(df) + projection.horizon)
        v_last = log_s[-1]
        fig.add_trace(go.Scatter3d(
            x=t_f, y=np.full_like(t_f, v_last, dtype=float),
            z=np.log10(projection.median_path), mode="lines",
            line=dict(color=PROJ, width=5, dash="dash"),
            name="projected median path",
            hovertemplate="proj $%{text}<extra></extra>",
            text=[f"{p:,.2f}" for p in projection.median_path],
        ))
        for band, nm in [(projection.lo_band, "20th pct"),
                         (projection.hi_band, "80th pct")]:
            fig.add_trace(go.Scatter3d(
                x=t_f, y=np.full_like(t_f, v_last, dtype=float),
                z=np.log10(band), mode="lines",
                line=dict(color=PROJ, width=1.5),
                opacity=0.45, name=nm, showlegend=False,
                hoverinfo="skip"))

    # year tick labels on the time axis
    years = pd.Series(dates.year)
    tickvals = [int(years[years == y].index[0]) for y in sorted(years.unique())][::max(1, len(years.unique()) // 8)]
    ticktext = [str(dates[v].year) for v in tickvals]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        scene=dict(
            xaxis=dict(title="time", tickvals=tickvals, ticktext=ticktext,
                       backgroundcolor=BG, gridcolor=GRID),
            yaxis=dict(title=y_title, backgroundcolor=BG, gridcolor=GRID),
            zaxis=dict(title="log10 price", backgroundcolor=BG, gridcolor=GRID),
            aspectratio=dict(x=2.2, y=0.8, z=1.0),
            camera=dict(eye=dict(x=1.6, y=-1.9, z=0.7)),
        ),
        legend=dict(bgcolor="rgba(14,18,32,0.7)", font=dict(size=10)),
        margin=dict(l=0, r=0, t=30, b=0),
        height=640,
    )
    return fig


def fractal_figure_2d(ticker: str, df: pd.DataFrame,
                      matches: list[MotifMatch] | None = None,
                      projection: Projection | None = None) -> go.Figure:
    """Classic 2D view with motif rectangles — for direct comparison with
    the hand-annotated reference charts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines",
                             line=dict(color=TRACE, width=1.4),
                             name=f"{ticker} close"))
    if matches:
        boxes = _family_boxes(matches, np.log(df["Close"].values.astype(float)))
        for box in boxes:
            a, b = box["a"], box["b"]
            seg = df["Close"].iloc[a:b]
            dash = "solid" if box["kind"] == "hist" else "dot"
            y1 = float(seg.max()) * 1.015
            fig.add_shape(type="rect",
                          x0=df.index[a], x1=df.index[b - 1],
                          y0=float(seg.min()) * 0.985, y1=y1,
                          line=dict(color=box["color"], width=2, dash=dash),
                          fillcolor="rgba(0,0,0,0)")
            fig.add_annotation(x=df.index[a], y=np.log10(y1), yanchor="bottom",
                               xanchor="left", showarrow=False,
                               text=box["fam"] + ("’" if box["kind"] == "live"
                                                  else ""),
                               font=dict(color=box["color"], size=11,
                                         family="monospace"))
    if projection is not None:
        fig.add_trace(go.Scatter(
            x=projection.dates, y=projection.median_path, mode="lines",
            line=dict(color=PROJ, width=2, dash="dash"), name="projection"))
        fig.add_trace(go.Scatter(
            x=list(projection.dates) + list(projection.dates[::-1]),
            y=list(projection.hi_band) + list(projection.lo_band[::-1]),
            fill="toself", fillcolor="rgba(242,233,220,0.10)",
            line=dict(width=0), name="20–80% band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[projection.buy_day], y=[projection.buy_price], mode="markers+text",
            marker=dict(color="#2E9E5B", size=11, symbol="triangle-up"),
            text=[f"buy zone ${projection.buy_price:,.2f}"],
            textposition="bottom center", name="buy zone"))
        fig.add_trace(go.Scatter(
            x=[projection.sell_day], y=[projection.sell_price], mode="markers+text",
            marker=dict(color="#C23B80", size=11, symbol="triangle-down"),
            text=[f"sell zone ${projection.sell_price:,.2f}"],
            textposition="top center", name="sell zone"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                      yaxis=dict(type="log", gridcolor=GRID, title="price ($, log)"),
                      xaxis=dict(gridcolor=GRID),
                      legend=dict(bgcolor="rgba(14,18,32,0.7)", font=dict(size=10)),
                      margin=dict(l=10, r=10, t=30, b=10), height=440)
    return fig


def option_chain_figure_3d(ticker: str, chain: pd.DataFrame,
                           spot: float | None = None) -> go.Figure:
    """3D option-chain fractal: days-to-expiry × log10 strike × log10 price.

    Each expiration's call curve C(K) is one ribbon through the surface.
    The curves are clustered into shape families with the same machinery
    as the price motifs (min-max shape space over log price), so expiries
    whose price-vs-strike geometry is the same fractal shape share a
    color — the term structure's self-similarity made visible. Puts are
    drawn dimmer in the family color of their expiry.
    """
    fig = go.Figure()
    expiries = sorted(pd.to_datetime(chain["expiration"].unique()))
    curves = []
    for exp in expiries:
        sub = chain[chain["expiration"] == exp].sort_values("strike")
        sub = sub[sub["strike"] > 0]
        c = sub[(sub["call"] > 0) & np.isfinite(sub["call"])]
        shape = None
        if len(c) >= 8:
            s, rng = to_shape(np.log10(c["call"].values))
            if rng > 0:
                shape = s
        curves.append((exp, sub, c, shape))

    shaped = [i for i, (_, _, _, s) in enumerate(curves) if s is not None]
    fams = group_families([curves[i][3] for i in shaped], min_corr=0.90)
    fam_of = {i: f for i, f in zip(shaped, fams)}

    z_min = np.inf
    legend_done = set()
    for i, (exp, sub, c, shape) in enumerate(curves):
        if len(c) == 0:
            continue
        f = fam_of.get(i)
        color = BOX_COLORS[f % len(BOX_COLORS)] if f is not None else "#5A6478"
        letter = chr(ord("A") + f % 26) if f is not None else "?"
        dte = int(sub["dte"].iloc[0])
        z = np.log10(c["call"].values)
        z_min = min(z_min, z.min())
        fig.add_trace(go.Scatter3d(
            x=np.full(len(c), dte), y=np.log10(c["strike"].values), z=z,
            mode="lines", line=dict(color=color, width=4),
            name=f"{exp.date()} calls · pattern {letter}",
            legendgroup=f"fam-{letter}",
            customdata=np.stack([c["strike"].values, c["call"].values], axis=-1),
            hovertemplate=(f"{exp.date()} · {dte}d"
                           "<br>strike $%{customdata[0]:,.2f}"
                           "<br>call $%{customdata[1]:,.2f}<extra></extra>"),
        ))
        p = sub[(sub["put"] > 0) & np.isfinite(sub["put"])]
        if len(p) > 0:
            zp = np.log10(p["put"].values)
            z_min = min(z_min, zp.min())
            fig.add_trace(go.Scatter3d(
                x=np.full(len(p), dte), y=np.log10(p["strike"].values), z=zp,
                mode="lines", line=dict(color=color, width=2), opacity=0.35,
                name=f"{exp.date()} puts", legendgroup=f"fam-{letter}",
                showlegend=False,
                customdata=np.stack([p["strike"].values, p["put"].values],
                                    axis=-1),
                hovertemplate=(f"{exp.date()} · {dte}d"
                               "<br>strike $%{customdata[0]:,.2f}"
                               "<br>put $%{customdata[1]:,.2f}<extra></extra>"),
            ))

    if spot and spot > 0 and np.isfinite(z_min):
        dtes = sorted(chain["dte"].unique())
        fig.add_trace(go.Scatter3d(
            x=[dtes[0], dtes[-1]], y=[np.log10(spot)] * 2, z=[z_min] * 2,
            mode="lines", line=dict(color=PROJ, width=3, dash="dot"),
            name=f"spot ${spot:,.2f}",
            hoverinfo="name",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        scene=dict(
            xaxis=dict(title="days to expiry", backgroundcolor=BG,
                       gridcolor=GRID),
            yaxis=dict(title="log10 strike", backgroundcolor=BG,
                       gridcolor=GRID),
            zaxis=dict(title="log10 option price", backgroundcolor=BG,
                       gridcolor=GRID),
            aspectratio=dict(x=1.6, y=1.0, z=0.9),
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.8)),
        ),
        legend=dict(bgcolor="rgba(14,18,32,0.7)", font=dict(size=10)),
        margin=dict(l=0, r=0, t=30, b=0),
        height=640,
    )
    return fig
