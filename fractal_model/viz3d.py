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


def _fmt_qty(x: float) -> str:
    for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(x) >= div:
            return f"{x / div:,.1f}{suf}"
    return f"{x:,.0f}"


def _fmt_px(x: float) -> str:
    return f"${x:,.2f}" if x < 10 else f"${x:,.0f}"


def _dollar_ticks(lo: float, hi: float, n: int = 6) -> tuple[list, list]:
    """Tick positions in log10 space labeled with actual dollar prices,
    so the log axes read in real money."""
    vals = list(np.linspace(lo, hi, n))
    return vals, [_fmt_px(10 ** v) for v in vals]


def family_summaries(matches: list[MotifMatch], close: pd.Series,
                     max_matches: int = 24) -> list[dict]:
    """One plain-English sentence per pattern family, computed from what
    actually followed that family's historical occurrences — e.g.
    'Pattern A: … a downtrend may follow once the live fractal completes.'
    Forward window is half the occurrence's own length (min 10 bars).
    """
    closev = close.values.astype(float)
    boxes = _family_boxes(matches, np.log(closev), max_matches)
    out = []
    for fam in sorted({b["fam"] for b in boxes}):
        hist = [b for b in boxes if b["fam"] == fam and b["kind"] == "hist"]
        if not hist:
            continue
        rets, spans = [], []
        for b in hist:
            fwd_len = max(10, (b["b"] - b["a"]) // 2)
            end = min(len(closev) - 1, b["b"] - 1 + fwd_len)
            if end > b["b"] - 1:
                rets.append(closev[end] / closev[b["b"] - 1] - 1.0)
                spans.append(end - (b["b"] - 1))
        if not rets:
            continue
        avg = float(np.mean(rets))
        up_frac = float(np.mean([r > 0 for r in rets]))
        n = len(rets)
        if avg > 0.02:
            trend = "an uptrend may follow once the live fractal completes"
        elif avg < -0.02:
            trend = "a downtrend may follow once the live fractal completes"
        else:
            trend = ("little net movement is implied once the live fractal "
                     "completes")
        out.append({
            "fam": fam, "color": hist[0]["color"],
            "text": (f"Pattern {fam}: after its {n} historical occurrence"
                     f"{'s' if n != 1 else ''} price averaged {avg:+.0%} "
                     f"over the next ~{int(np.mean(spans))} trading days "
                     f"(up {up_frac:.0%} of the time) — {trend}."),
        })
    return out


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
                hovertemplate=f"{nm} $%{{text}}<extra></extra>",
                text=[f"{p:,.2f}" for p in band]))

        # buy zone / sell target markers on the projected path itself
        b_i = int(np.argmin(np.abs(projection.dates - projection.buy_day)))
        s_i = int(np.argmin(np.abs(projection.dates - projection.sell_day)))
        for i_d, px, day, nm, col, tpos in [
                (b_i, projection.buy_price, projection.buy_day,
                 "buy zone", "#2E9E5B", "bottom center"),
                (s_i, projection.sell_price, projection.sell_day,
                 "sell target", "#C23B80", "top center")]:
            fig.add_trace(go.Scatter3d(
                x=[len(df) + i_d], y=[v_last], z=[np.log10(px)],
                mode="markers+text",
                marker=dict(color=col, size=7, symbol="diamond"),
                text=[f"{nm} ${px:,.2f}"], textposition=tpos,
                textfont=dict(color=col, size=11),
                name=f"{nm} ${px:,.2f} by {day.date()}",
                hovertemplate=(f"{nm} ${px:,.2f}"
                               f"<br>{day.date()}<extra></extra>"),
            ))

    # year tick labels on the time axis
    years = pd.Series(dates.year)
    tickvals = [int(years[years == y].index[0]) for y in sorted(years.unique())][::max(1, len(years.unique()) // 8)]
    ticktext = [str(dates[v].year) for v in tickvals]

    # label the log axes in real dollars / share counts
    z_span = [log_p]
    if projection is not None:
        z_span.append(np.log10(projection.median_path))
    z_all = np.concatenate(z_span)
    z_ticks, z_text = _dollar_ticks(float(z_all.min()), float(z_all.max()))
    yaxis = dict(title=y_title, backgroundcolor=BG, gridcolor=GRID)
    if shares is not None:
        y_ticks = list(np.linspace(float(log_s.min()), float(log_s.max()), 4))
        yaxis.update(title="shares outstanding (log scale)",
                     tickvals=y_ticks,
                     ticktext=[_fmt_qty(10 ** v) for v in y_ticks])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        scene=dict(
            xaxis=dict(title="time", tickvals=tickvals, ticktext=ticktext,
                       backgroundcolor=BG, gridcolor=GRID),
            yaxis=yaxis,
            zaxis=dict(title="price ($, log scale)", tickvals=z_ticks,
                       ticktext=z_text, backgroundcolor=BG, gridcolor=GRID),
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
                           spot: float | None = None,
                           exp_spot: dict[int, float] | None = None
                           ) -> go.Figure:
    """3D option-chain fractal: days-to-expiry × log10 strike × log10 price.

    Each expiration's call curve C(K) is one ribbon through the surface.
    The curves are clustered into shape families with the same machinery
    as the price motifs (min-max shape space over log price), so expiries
    whose price-vs-strike geometry is the same fractal shape share a
    color — the term structure's self-similarity made visible. Puts are
    drawn dimmer in the family color of their expiry.

    The space under each call ribbon is filled with a curtain mesh shaded
    by **model-expected profit at expiry**: buy the call at today's price,
    settle at the fractal projection's spot for that date (`exp_spot`,
    days-to-expiry -> projected price; falls back to today's spot). Green
    = the model expects the contract to finish worth more than it costs,
    red = expected loss. The same P/L is reported on hover.
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
        dte = int(sub["dte"].iloc[0]) if len(sub) else 0
        s_exp = (exp_spot or {}).get(dte, spot)
        profit = None
        if s_exp is not None and len(c) > 0:
            profit = (np.maximum(s_exp - c["strike"].values, 0.0)
                      - c["call"].values)
        curves.append((exp, dte, sub, c, shape, profit))

    shaped = [i for i, cv in enumerate(curves) if cv[4] is not None]
    fams = group_families([curves[i][4] for i in shaped], min_corr=0.90)
    fam_of = {i: f for i, f in zip(shaped, fams)}

    # shared floor and symmetric P/L color range across all expiries
    z_all = []
    for (_, _, sub, c, _, _) in curves:
        if len(c):
            z_all.append(np.log10(c["call"].values))
        p = sub[(sub["put"] > 0) & np.isfinite(sub["put"])]
        if len(p):
            z_all.append(np.log10(p["put"].values))
    if not z_all:
        return fig
    z_floor = float(min(z.min() for z in z_all)) - 0.08
    z_top = float(max(z.max() for z in z_all))
    strikes_pos = chain.loc[chain["strike"] > 0, "strike"]
    k_lo, k_hi = np.log10(strikes_pos.min()), np.log10(strikes_pos.max())
    # curtain color = return on premium, not $ P/L: a symmetric $ range let
    # deep-ITM contracts (huge absolute swings) wash every ordinary strike
    # into the gray midpoint. Return per dollar spent is comparable across
    # strikes AND expiries; clip at ±100% and sign-sqrt boost so ±25%
    # already reads clearly green/red.
    PL_SCALE = [[0.0, "#FF2E55"], [0.5, "#20263B"], [1.0, "#12E27C"]]

    for i, (exp, dte, sub, c, shape, profit) in enumerate(curves):
        if len(c) == 0:
            continue
        f = fam_of.get(i)
        color = BOX_COLORS[f % len(BOX_COLORS)] if f is not None else "#5A6478"
        letter = chr(ord("A") + f % 26) if f is not None else "?"
        log_k = np.log10(c["strike"].values)
        z = np.log10(c["call"].values)

        if profit is not None:
            ret = profit / c["call"].values          # return on premium
            boosted = np.sign(ret) * np.sqrt(np.minimum(np.abs(ret), 1.0))
            pl_col, ret_col = profit, ret
        else:
            ret, boosted = None, None
            pl_col = ret_col = np.full(len(c), np.nan)
        pl_line = ("<br>model P/L %{customdata[2]:+,.2f} "
                   "(%{customdata[3]:+.0%} on premium)"
                   if profit is not None else "")
        fig.add_trace(go.Scatter3d(
            x=np.full(len(c), dte), y=log_k, z=z,
            mode="lines", line=dict(color=color, width=4),
            name=f"{exp.date()} calls · pattern {letter}",
            legendgroup=f"fam-{letter}",
            customdata=np.stack([c["strike"].values, c["call"].values,
                                 pl_col, ret_col], axis=-1),
            hovertemplate=(f"{exp.date()} · {dte}d"
                           "<br>strike $%{customdata[0]:,.2f}"
                           "<br>call $%{customdata[1]:,.2f}"
                           f"{pl_line}<extra></extra>"),
        ))

        # curtain fill under the call ribbon: expected profit over time
        n_pts = len(c)
        if n_pts >= 2:
            xs = np.concatenate([np.full(n_pts, dte)] * 2)
            ys = np.concatenate([log_k, log_k])
            zs = np.concatenate([z, np.full(n_pts, z_floor)])
            tri_i, tri_j, tri_k = [], [], []
            for q in range(n_pts - 1):
                tri_i += [q, q + 1]
                tri_j += [q + 1, n_pts + q + 1]
                tri_k += [n_pts + q, n_pts + q]
            mesh_kw = dict(x=xs, y=ys, z=zs, i=tri_i, j=tri_j, k=tri_k,
                           opacity=0.45, hoverinfo="skip",
                           name=f"{exp.date()} expected P/L",
                           legendgroup=f"fam-{letter}", showlegend=False,
                           flatshading=True,
                           lighting=dict(ambient=0.95, diffuse=0.3,
                                         specular=0.0))
            if boosted is not None:
                mesh_kw.update(intensity=np.concatenate([boosted, boosted]),
                               colorscale=PL_SCALE, cmin=-1.0,
                               cmax=1.0, cmid=0.0, showscale=False)
            else:
                mesh_kw.update(color=color, opacity=0.30)
            fig.add_trace(go.Mesh3d(**mesh_kw))

        p = sub[(sub["put"] > 0) & np.isfinite(sub["put"])]
        if len(p) > 0:
            fig.add_trace(go.Scatter3d(
                x=np.full(len(p), dte), y=np.log10(p["strike"].values),
                z=np.log10(p["put"].values),
                mode="lines", line=dict(color=color, width=2), opacity=0.35,
                name=f"{exp.date()} puts", legendgroup=f"fam-{letter}",
                showlegend=False,
                customdata=np.stack([p["strike"].values, p["put"].values],
                                    axis=-1),
                hovertemplate=(f"{exp.date()} · {dte}d"
                               "<br>strike $%{customdata[0]:,.2f}"
                               "<br>put $%{customdata[1]:,.2f}<extra></extra>"),
            ))

    if spot and spot > 0:
        dtes = sorted(chain["dte"].unique())
        fig.add_trace(go.Scatter3d(
            x=[dtes[0], dtes[-1]], y=[np.log10(spot)] * 2, z=[z_floor] * 2,
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
            yaxis=dict(title="strike ($, log scale)",
                       tickvals=_dollar_ticks(float(k_lo), float(k_hi), 5)[0],
                       ticktext=_dollar_ticks(float(k_lo), float(k_hi), 5)[1],
                       backgroundcolor=BG, gridcolor=GRID),
            zaxis=dict(title="option price ($, log scale)",
                       tickvals=_dollar_ticks(z_floor, z_top, 5)[0],
                       ticktext=_dollar_ticks(z_floor, z_top, 5)[1],
                       backgroundcolor=BG, gridcolor=GRID),
            aspectratio=dict(x=1.6, y=1.0, z=0.9),
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.8)),
        ),
        legend=dict(bgcolor="rgba(14,18,32,0.7)", font=dict(size=10)),
        margin=dict(l=0, r=0, t=30, b=0),
        height=640,
    )
    return fig
