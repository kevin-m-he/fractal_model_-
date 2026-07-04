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

from .motif import MotifMatch
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
        # live window box (white outline via high-opacity thin box)
        m0 = matches[0]
        seen = set()
        for ci, m in enumerate(matches[:6]):
            color = BOX_COLORS[ci % len(BOX_COLORS)]
            for (a, b, tag) in [(m.hist_start, m.hist_end, "motif"),
                                (m.live_start, m.live_end, "live")]:
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                z0, z1 = log_p[a:b].min(), log_p[a:b].max()
                y0, y1 = np.nanmin(log_s[a:b]), np.nanmax(log_s[a:b])
                # share counts barely move inside a window — give the box a
                # visible minimum depth so it doesn't collapse to a plane
                pad_y = max(0.012, 0.05 * (y1 - y0))
                y0, y1 = y0 - pad_y, y1 + pad_y
                pad_z = 0.02 * (z1 - z0 + 1e-6)
                label = (f"{tag} {dates[a].year}–{dates[b-1].year} "
                         f"(x{m.time_ratio:.1f} time, x{m.amp_ratio:.1f} amp)"
                         if tag == "motif" else "live pattern")
                op = 0.10 if tag == "live" else 0.18
                fig.add_trace(_box_mesh(a, b, y0, y1, z0 - pad_z, z1 + pad_z,
                                        color, label, opacity=op))

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
        for ci, m in enumerate(matches[:6]):
            color = BOX_COLORS[ci % len(BOX_COLORS)]
            for (a, b, dash) in [(m.hist_start, m.hist_end, "solid"),
                                 (m.live_start, m.live_end, "dot")]:
                seg = df["Close"].iloc[a:b]
                fig.add_shape(type="rect",
                              x0=df.index[a], x1=df.index[b - 1],
                              y0=float(seg.min()) * 0.985,
                              y1=float(seg.max()) * 1.015,
                              line=dict(color=color, width=2, dash=dash),
                              fillcolor="rgba(0,0,0,0)")
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
