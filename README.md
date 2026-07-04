# Fractal Model

A local desktop application that detects **self-similar price patterns across
scale** and projects them forward — working *solely* on fractal structure
(price, time, volume), with no fundamentals, sentiment, or macro inputs.

It renders each stock as a **3D fractal** (time × volume × price), boxes the
recurring motifs it finds, and — critically — **backtests itself out-of-sample
against a naive baseline and shows you the verdict**, because a pattern model
that can't be falsified is astrology with a nicer chart.

![concept](assets/concept.svg)

## What it does

- **3D fractal visualizer** — search any ticker, rotate its price/time/volume
  fractal, see detected self-affine motifs as colored boxes (a 3D generalization
  of hand-drawn chart annotations).
- **All-scales projection** — from ~3-month to ~4-year patterns, each with a
  buy zone, sell target, timeframe, and an explicit confidence score, ranked by
  confidence.
- **Top-10 following fractals** — scans a universe and lists the strongest
  current fractal setups with buy/sell/timeframe.
- **Walk-forward backtest** — measures direction hit rate, trade hit rate, and
  error on data the model never saw, versus a naive-drift baseline. The verdict
  is shown plainly, green or red.

## The math (short version)

Prices are treated as **self-affine**: rescaling time by `a` and price by `a^H`
leaves the process statistically unchanged, where `H` is the **Hurst exponent**
(estimated by DFA and R/S). Windows are compared in **shape space** (log-price
normalized, time resampled), and a candidate recurrence is only trusted if its
time/amplitude magnification obeys the series' own scaling law `a_p ≈ a_t^H`.
**Multifractal** richness (MF-DFA singularity width `Δα`) and DFA fit quality
feed a transparent confidence score. Full derivation:
[`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md).

## Install & run

```bash
pip install -r requirements.txt
python -m fractal_model.app
# open http://127.0.0.1:8050
```

First run fetches history from Yahoo Finance (Stooq fallback) and caches it
locally.

## Honest status

On the tickers tested so far, the fractal signal does **not** yet beat a naive
drift baseline at every scale — see the Backtest tab and §0/§9 of the spec. The
promising sign is a positive confidence↔accuracy correlation. This is a research
instrument for finding and stress-testing fractal structure, **not** investment
advice and **not** a forecast.

## License

GNU AGPLv3 — see [`LICENSE`](LICENSE). Network use is distribution: if you run a
modified version as a service, you must offer users its source.
