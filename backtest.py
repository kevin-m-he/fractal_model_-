# SPDX-License-Identifier: AGPL-3.0-or-later
"""Walk-forward backtesting — the falsification layer.

A pattern model validated only on the charts that inspired it is not a
model. This module answers, on data the model never saw when forming
each forecast:

  * direction hit rate  — did price move the way the projection said?
  * trade hit rate      — did the buy-then-sell plan end profitable?
  * MAPE at horizon     — how far off was the median path endpoint?
  * baseline comparison — same stats for a naive "price drifts at its
    trailing mean return" forecaster. If the fractal model cannot beat
    the naive drift, its signal is not real at that scale. The app
    displays this verdict verbatim.

Protocol: anchored walk-forward. At each anchor date T (stepped every
`step` bars), the model sees only bars <= T, produces a projection, and
is graded against bars (T, T + horizon].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .projection import project


@dataclass
class BacktestReport:
    ticker: str
    live_len: int
    horizon: int
    n_trials: int
    direction_hit_rate: float
    trade_hit_rate: float
    mean_trade_return: float
    mape_at_horizon: float
    baseline_direction_hit_rate: float
    baseline_mape: float
    beats_baseline: bool
    mean_confidence: float
    confidence_correlation: float  # corr(confidence, |error| * -1)


def walk_forward(ticker: str, close: pd.Series, live_len: int,
                 horizon: int, step: int | None = None,
                 min_history: int | None = None) -> BacktestReport | None:
    if step is None:
        step = max(10, horizon // 2)
    if min_history is None:
        min_history = max(live_len * 3, 500)
    n = len(close)
    anchors = list(range(min_history, n - horizon, step))
    if len(anchors) < 5:
        return None

    dir_hits, base_dir_hits = [], []
    trade_rets, mapes, base_mapes, confs, abs_errs = [], [], [], [], []

    for T in anchors:
        seen = close.iloc[:T]
        future = close.iloc[T: T + horizon]
        if len(future) < horizon:
            continue
        p = project(ticker, seen, live_len=live_len, horizon=horizon)
        if p is None:
            continue
        last = float(seen.iloc[-1])
        actual_end = float(future.iloc[-1])
        pred_end = float(p.median_path[-1])

        # direction
        pred_up = pred_end > last
        act_up = actual_end > last
        dir_hits.append(pred_up == act_up)

        # naive drift baseline: continue trailing mean daily log return
        drift = np.mean(np.diff(np.log(seen.values[-live_len:])))
        base_end = last * np.exp(drift * horizon)
        base_dir_hits.append((base_end > last) == act_up)
        base_mapes.append(abs(base_end - actual_end) / actual_end)

        # trade: buy at first touch of buy_price (or day-1 open if the
        # projection says buy immediately), sell at sell target or horizon end
        fut = future.values.astype(float)
        b_day = int(np.argmin(np.abs(p.dates - p.buy_day))) if len(p.dates) else 0
        entered = None
        for i, px in enumerate(fut):
            if px <= p.buy_price or i >= b_day:
                entered = (i, px if px <= p.buy_price else fut[i])
                break
        if entered is not None:
            i0, entry = entered
            exit_px = fut[-1]
            for j in range(i0 + 1, len(fut)):
                if fut[j] >= p.sell_price:
                    exit_px = fut[j]
                    break
            trade_rets.append(exit_px / entry - 1.0)

        err = abs(pred_end - actual_end) / actual_end
        mapes.append(err)
        abs_errs.append(err)
        confs.append(p.confidence)

    if len(mapes) < 5:
        return None
    conf_corr = float(np.corrcoef(confs, -np.array(abs_errs))[0, 1]) \
        if np.std(confs) > 0 else 0.0
    mape = float(np.mean(mapes))
    base_mape = float(np.mean(base_mapes))
    dir_rate = float(np.mean(dir_hits))
    base_rate = float(np.mean(base_dir_hits))
    return BacktestReport(
        ticker=ticker, live_len=live_len, horizon=horizon,
        n_trials=len(mapes),
        direction_hit_rate=dir_rate,
        trade_hit_rate=float(np.mean([r > 0 for r in trade_rets])) if trade_rets else float("nan"),
        mean_trade_return=float(np.mean(trade_rets)) if trade_rets else float("nan"),
        mape_at_horizon=mape,
        baseline_direction_hit_rate=base_rate,
        baseline_mape=base_mape,
        beats_baseline=(dir_rate > base_rate) and (mape < base_mape),
        mean_confidence=float(np.mean(confs)),
        confidence_correlation=conf_corr,
    )
