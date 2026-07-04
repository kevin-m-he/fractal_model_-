# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fractal Model — local desktop application (Dash + Plotly 3D).

Run:  python -m fractal_model.app   then open http://127.0.0.1:8050

Tabs:
  Visualizer  — search any ticker, render the 3D price/time/volume
                fractal with detected motif boxes and the projected
                buy/sell path, all scales ranked by confidence.
  Top 10      — scan the default universe, rank by best-scale
                confidence, show buy price / sell target / timeframe.
  Backtest    — walk-forward falsification for the current ticker,
                with the baseline-comparison verdict shown plainly.

Nothing here is financial advice. Projections are pattern-implied
scenarios with explicit confidence and measured error, not forecasts.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update

from .data import get_history
from .projection import project_all_scales
from .backtest import walk_forward
from .scanner import top_following_fractals
from .viz3d import fractal_figure_3d, fractal_figure_2d, BG

BLANK = {"paper_bgcolor": BG, "plot_bgcolor": BG, "template": "plotly_dark"}

app = Dash(__name__, title="Fractal Model")
server = app.server

DISCLAIMER = ("Pattern-implied scenarios with measured error — not financial "
              "advice, not a forecast. Always read the Backtest tab.")

CARD = {"backgroundColor": "#131829", "border": "1px solid #232A40",
        "borderRadius": "10px", "padding": "16px", "marginBottom": "14px"}


def _header():
    return html.Div([
        html.Div([
            html.Span("FRACTAL", style={"color": "#E8C400", "fontWeight": 800}),
            html.Span("MODEL", style={"color": "#F2E9DC", "fontWeight": 300}),
        ], style={"fontSize": "26px", "letterSpacing": "3px",
                  "fontFamily": "monospace"}),
        html.Div("self-similar price structure across scale · "
                 "time × volume × price", style={"color": "#6B7488",
                 "fontSize": "12px", "fontFamily": "monospace"}),
        html.Div(DISCLAIMER, style={"color": "#8B6A2B", "fontSize": "11px",
                 "marginTop": "6px", "fontFamily": "monospace"}),
    ], style={"padding": "18px 24px", "borderBottom": "1px solid #232A40"})


app.layout = html.Div([
    dcc.Store(id="cur-ticker", data="NFLX"),
    _header(),
    dcc.Tabs(id="tabs", value="viz", children=[
        dcc.Tab(label="Visualizer", value="viz"),
        dcc.Tab(label="Top 10 Fractals", value="top"),
        dcc.Tab(label="Backtest", value="bt"),
    ], colors={"border": "#232A40", "primary": "#E8C400", "background": "#0E1220"}),
    html.Div(id="tab-body", style={"padding": "20px 24px"}),
], style={"backgroundColor": "#0E1220", "minHeight": "100vh",
          "color": "#D7DCE8", "fontFamily": "system-ui, sans-serif"})


# ---------------- tab router ----------------
@app.callback(Output("tab-body", "children"), Input("tabs", "value"),
              State("cur-ticker", "data"))
def render_tab(tab, ticker):
    if tab == "viz":
        return _viz_layout(ticker or "NFLX")
    if tab == "top":
        return _top_layout()
    return _bt_layout(ticker or "NFLX")


def _viz_layout(ticker):
    return html.Div([
        html.Div([
            dcc.Input(id="ticker-in", value=ticker, type="text",
                      placeholder="ticker, e.g. AAPL or BTC-USD",
                      style={"backgroundColor": "#0E1220", "color": "#F2E9DC",
                             "border": "1px solid #232A40", "padding": "10px",
                             "borderRadius": "6px", "width": "260px",
                             "fontFamily": "monospace", "fontSize": "15px"}),
            html.Button("render fractal", id="go-btn", n_clicks=0,
                        style={"marginLeft": "10px", "backgroundColor": "#E8C400",
                               "color": "#0E1220", "border": "none",
                               "padding": "10px 18px", "borderRadius": "6px",
                               "fontWeight": 700, "cursor": "pointer"}),
        ], style={"marginBottom": "14px"}),
        dcc.Loading(html.Div(id="viz-content"), color="#E8C400", type="dot"),
    ])


@app.callback(
    Output("viz-content", "children"), Output("cur-ticker", "data"),
    Input("go-btn", "n_clicks"), State("ticker-in", "value"),
    prevent_initial_call=False)
def render_viz(_n, ticker):
    ticker = (ticker or "NFLX").strip().upper()
    try:
        df = get_history(ticker)
    except Exception as e:
        return html.Div(f"Could not load '{ticker}': {e}",
                        style={"color": "#C23B80"}), no_update
    projs = project_all_scales(ticker, df["Close"])
    if not projs:
        return html.Div(f"No fractal motifs found for {ticker} at any scale.",
                        style={"color": "#8B6A2B"}), ticker
    best = projs[0]
    fig3d = fractal_figure_3d(ticker, df, best.matches, best)
    fig2d = fractal_figure_2d(ticker, df, best.matches, best)

    cards = []
    for p in projs:
        arrow = "▲" if p.expected_return >= 0 else "▼"
        col = "#2E9E5B" if p.expected_return >= 0 else "#C23B80"
        cards.append(html.Div([
            html.Div(p.scale_label, style={"color": "#6B7488",
                     "fontFamily": "monospace", "fontSize": "11px"}),
            html.Div(f"conf {p.confidence:.0%}", style={"fontSize": "20px",
                     "fontWeight": 700, "color": "#E8C400"}),
            html.Div([html.Span(f"buy ${p.buy_price:,.2f}"),
                      html.Span(" → ", style={"color": "#6B7488"}),
                      html.Span(f"sell ${p.sell_price:,.2f}", style={"color": col})],
                     style={"fontFamily": "monospace", "fontSize": "13px"}),
            html.Div(f"{arrow} {p.expected_return:+.1%} by {p.sell_day.date()}",
                     style={"color": col, "fontSize": "12px"}),
            html.Div(f"H={p.hurst['H_dfa']:.2f} · {p.hurst['regime']} · "
                     f"{p.n_matches} motifs", style={"color": "#6B7488",
                     "fontSize": "11px", "fontFamily": "monospace"}),
        ], style={**CARD, "flex": "1", "minWidth": "180px", "marginRight": "10px"}))

    return html.Div([
        html.Div(cards, style={"display": "flex", "flexWrap": "wrap"}),
        html.Div([html.Div("3D fractal — drag to rotate", style={"color": "#6B7488",
                 "fontSize": "12px", "marginBottom": "4px",
                 "fontFamily": "monospace"}),
                 dcc.Graph(figure=fig3d, config={"displModeBar": True})],
                 style=CARD),
        html.Div([html.Div("2D view with motif boxes (compare to your "
                 "annotated charts)", style={"color": "#6B7488",
                 "fontSize": "12px", "marginBottom": "4px",
                 "fontFamily": "monospace"}),
                 dcc.Graph(figure=fig2d)], style=CARD),
    ]), ticker


def _top_layout():
    return html.Div([
        html.Button("scan universe", id="scan-btn", n_clicks=0,
                    style={"backgroundColor": "#E8C400", "color": "#0E1220",
                           "border": "none", "padding": "10px 18px",
                           "borderRadius": "6px", "fontWeight": 700,
                           "cursor": "pointer"}),
        html.Span("  scans ~45 tickers across all scales — takes a minute",
                  style={"color": "#6B7488", "fontSize": "12px"}),
        dcc.Loading(html.Div(id="top-content"), color="#E8C400", type="dot"),
    ])


@app.callback(Output("top-content", "children"), Input("scan-btn", "n_clicks"),
              prevent_initial_call=True)
def run_scan(_n):
    df = top_following_fractals(n=10)
    if df.empty:
        return html.Div("Scan returned nothing — check network.",
                        style={"color": "#C23B80"})
    disp = df.copy()
    disp["last"] = disp["last"].map(lambda x: f"${x:,.2f}")
    disp["buy_price"] = disp["buy_price"].map(lambda x: f"${x:,.2f}")
    disp["sell_target"] = disp["sell_target"].map(lambda x: f"${x:,.2f}")
    disp["exp_return"] = disp["exp_return"].map(lambda x: f"{x:+.1%}")
    disp["confidence"] = disp["confidence"].map(lambda x: f"{x:.0%}")
    disp["H"] = disp["H"].map(lambda x: f"{x:.2f}")
    cols = ["ticker", "last", "scale", "buy_price", "buy_by",
            "sell_target", "sell_by", "exp_return", "confidence", "H"]
    return html.Div([
        html.Div("Top 10 following fractals — ranked by best-scale confidence",
                 style={"color": "#E8C400", "fontFamily": "monospace",
                        "margin": "14px 0"}),
        dash_table.DataTable(
            data=disp[cols].to_dict("records"),
            columns=[{"name": c, "id": c} for c in cols],
            style_header={"backgroundColor": "#232A40", "color": "#E8C400",
                          "fontWeight": "bold", "fontFamily": "monospace"},
            style_cell={"backgroundColor": "#131829", "color": "#D7DCE8",
                        "fontFamily": "monospace", "fontSize": "13px",
                        "border": "1px solid #232A40", "padding": "8px"},
        ),
    ])


def _bt_layout(ticker):
    return html.Div([
        html.Div(f"Walk-forward falsification — {ticker}",
                 style={"color": "#E8C400", "fontFamily": "monospace",
                        "marginBottom": "10px"}),
        html.Button("run backtest", id="bt-btn", n_clicks=0,
                    style={"backgroundColor": "#E8C400", "color": "#0E1220",
                           "border": "none", "padding": "10px 18px",
                           "borderRadius": "6px", "fontWeight": 700,
                           "cursor": "pointer"}),
        dcc.Store(id="bt-ticker", data=ticker),
        dcc.Loading(html.Div(id="bt-content"), color="#E8C400", type="dot"),
    ])


@app.callback(Output("bt-content", "children"), Input("bt-btn", "n_clicks"),
              State("bt-ticker", "data"), prevent_initial_call=True)
def run_bt(_n, ticker):
    try:
        df = get_history(ticker)
        projs = project_all_scales(ticker, df["Close"])
        if not projs:
            return html.Div("No motifs to backtest.", style={"color": "#8B6A2B"})
        p = projs[0]
        bt = walk_forward(ticker, df["Close"], live_len=p.live_len,
                          horizon=p.horizon)
    except Exception as e:
        return html.Div(f"Backtest failed: {e}", style={"color": "#C23B80"})
    if bt is None:
        return html.Div("Not enough history for a walk-forward test.",
                        style={"color": "#8B6A2B"})
    verdict_col = "#2E9E5B" if bt.beats_baseline else "#C23B80"
    verdict = ("Fractal signal BEATS naive-drift baseline at this scale."
               if bt.beats_baseline else
               "Fractal signal does NOT beat naive drift at this scale — "
               "treat projections with heavy skepticism.")

    def stat(label, val):
        return html.Div([html.Div(label, style={"color": "#6B7488",
                        "fontSize": "11px", "fontFamily": "monospace"}),
                        html.Div(val, style={"fontSize": "18px",
                        "fontWeight": 700})], style={**CARD, "flex": 1,
                        "minWidth": "150px", "marginRight": "10px"})

    return html.Div([
        html.Div(verdict, style={"color": verdict_col, "fontWeight": 700,
                 "fontSize": "15px", "padding": "12px", "border":
                 f"1px solid {verdict_col}", "borderRadius": "8px",
                 "marginBottom": "14px"}),
        html.Div([
            stat("out-of-sample trials", f"{bt.n_trials}"),
            stat("direction hit rate", f"{bt.direction_hit_rate:.0%}"),
            stat("baseline hit rate", f"{bt.baseline_direction_hit_rate:.0%}"),
            stat("trade hit rate", f"{bt.trade_hit_rate:.0%}"
                 if bt.trade_hit_rate == bt.trade_hit_rate else "n/a"),
        ], style={"display": "flex", "flexWrap": "wrap"}),
        html.Div([
            stat("MAPE @ horizon", f"{bt.mape_at_horizon:.1%}"),
            stat("baseline MAPE", f"{bt.baseline_mape:.1%}"),
            stat("mean trade return", f"{bt.mean_trade_return:+.1%}"
                 if bt.mean_trade_return == bt.mean_trade_return else "n/a"),
            stat("confidence↔accuracy", f"{bt.confidence_correlation:+.2f}"),
        ], style={"display": "flex", "flexWrap": "wrap"}),
        html.Div("Direction hit rate is measured on data the model never saw "
                 "when each forecast was made. If it can't beat naive drift, "
                 "the fractal edge isn't real at that scale — the model tells "
                 "you so rather than hiding it.", style={"color": "#6B7488",
                 "fontSize": "12px", "marginTop": "12px",
                 "fontFamily": "monospace"}),
    ])


def main():
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
