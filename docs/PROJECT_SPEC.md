# Fractal Model — Project Specification & Handoff

**Status:** working end-to-end prototype. Data → Hurst/multifractal → motif
detection → projection → 3D app → walk-forward falsification all run on live
market data (verified on NFLX, BTC-USD, AAPL).

**Purpose of this document:** everything a successor (human or model) needs to
understand, trust, extend, or correct the model — the math first, then the
architecture, then the open problems. If you are picking this up cold, read
sections 1–4 before touching code.

---

## 0. The core claim, stated honestly

The owner's definition: *a fractal is a price pattern that recurs at a magnified
scale over time — larger amplitude, longer duration, same shape.* The hand-
annotated BTC/NFLX charts show this as nested boxes: a small early pattern
reappears, blown up, later on.

This project takes that seriously **and tests it**. The model detects recurring
self-affine motifs and projects them forward, using **only** price/time/share
structure — no fundamentals, no sentiment, no macro. But a pattern model
validated only on the charts that inspired it is not a model. So the falsual
core deliverable is the **walk-forward backtest with a naive-drift baseline**
(§6). Early result: on NFLX at its best scale the fractal signal did **not** beat
naive drift (54% vs 58% direction). That is not a bug — it is the model
refusing to flatter itself. The honest edge so far is a **positive
confidence↔accuracy correlation** (the confidence score is higher when the model
is in fact more accurate), which suggests real but weak signal worth developing.
Do not remove or hide the backtest to make the product look better. That failure
mode is the whole reason this section is first.

---

## 1. Mathematical foundations

### 1.1 Self-affinity and the Hurst exponent

A time series `X(t)` is **self-affine** with exponent `H` if rescaling time by a
factor `a` and price by `a^H` yields a statistically identical process:

```
X(at) =_d  a^H · X(t)
```

`H` is the **Hurst exponent**, the single most important number in the project:

- `H = 0.5` → random walk (increments independent)
- `H > 0.5` → **persistent**: trends reinforce; a fractal that magnifies in a
  consistent direction (the BTC/NFLX case)
- `H < 0.5` → **anti-persistent**: mean-reverting

The self-affine law is what lets us compare a 2015 pattern ($10, 90 days) with a
2024 pattern ($600, 500 days): in **shape space** (log-price min-max normalized,
time resampled to K points) they become the same curve, and their *scale ratios*
`(a_t, a_p)` should obey `a_p ≈ a_t^H` if the recurrence is a true fractal rather
than a coincidental lookalike. This constraint is enforced in scoring (§3.3).

### 1.2 Estimating H — two independent estimators

**DFA (Detrended Fluctuation Analysis)** — primary, robust to non-stationarity.
Given log-price `x`, form increments, integrate to a profile
`Y(k) = Σ_{i≤k}(Δx_i − mean)`. For each scale `s`, split `Y` into segments of
length `s`, fit and remove an order-1 polynomial per segment, and compute the RMS
fluctuation `F(s)`. Then `F(s) ∝ s^H`, and `H` is the slope of `log F` vs
`log s`. The **R² of that log-log fit** doubles as a "how fractal is this
really?" score — a clean power law means clean scaling. Code: `hurst.dfa_hurst`.

**R/S (Rescaled Range)** — classical Hurst 1951 / Mandelbrot–Wallis 1969, kept
as an independent cross-check. Code: `hurst.rs_hurst`.

### 1.3 Multifractality — MF-DFA

Real markets are not monofractal: small and large moves scale with different
exponents. **MF-DFA** (Kantelhardt et al., 2002) generalizes DFA to moment order
`q`:

```
F_q(s) = { (1/N_s) Σ_v [F²(v,s)]^{q/2} }^{1/q}  ∝  s^{h(q)}
```

`h(q)` is the generalized Hurst exponent. From `τ(q) = q·h(q) − 1` and
`α = dτ/dq`, the **singularity spectrum width** `Δα = α_max − α_min` measures
fractal *richness*:

- `Δα ≈ 0` → monofractal / featureless (weak fractal candidate)
- `Δα` large → strongly multifractal — "patterns inside patterns," exactly the
  nested structure the owner drew

`Δα` feeds the confidence composite (§4). Code: `multifractal.mfdfa`.

### 1.4 Shape space (scale-invariant matching)

A window of log-prices `p[a:b]` maps to shape space by:

```
u = (p − min p) / (max p − min p)      # amplitude → [0,1], records log-range R
û = interp(u, K points)                # time → fixed length K = 64
```

Two windows are compared by **Pearson correlation** (shape agreement) and
**normalized RMSE** (pointwise deviation), both in `[0,1]`-ish ranges. Scale
information is *not* discarded — it is carried as `(R, length)` and re-checked
against the Hurst law. Code: `motif.to_shape`, `motif._shape_similarity`.

---

## 2. Data layer (`data.py`)

- Primary: Yahoo Finance (`yfinance`, `period=max`, daily, auto-adjusted).
- Fallback: Stooq daily CSV (`nflx.us`, `btcusd`, …).
- Local parquet cache (`~/.fractal_model_cache`, 6h TTL) → offline + reproducible
  backtests. Stale cache used as last resort.
- Cleaning: flatten MultiIndex columns, drop non-positive/NaN closes, de-dupe
  index, tz-strip. Raises `ValueError` if no source yields ≥ `min_rows` bars.

To add a source, implement `_fetch_x(ticker) -> DataFrame|None` with columns
`Open/High/Low/Close/Volume` and insert into the `get_history` fallback chain.

---

## 3. Motif detection (`motif.py`) — the engine

For a **live window** of the last `L_r` bars:

1. Map live window → shape space, record log-range `R_r`.
2. Build a geometric ladder of historical lengths `L_h ∈ [L_r/6, L_r]` (fractals
   usually recur *shorter* in the past and magnify toward the present).
3. Slide candidate windows (stride ∝ length) over all history **ending before the
   live window starts** (no look-ahead).
4. Score each candidate:

   ```
   shape term   = 0.55·max(corr,0) + 0.25·(1 − min(rmse/0.5, 1))
   hurst term   = 0.20·C,   C = exp(−|ln(a_p / a_t^H)|)      # §3.3
   score        = shape term + hurst term
   ```

5. Non-maximum suppression: drop any match overlapping a higher-scoring one by
   >50%. Return top-k.

### 3.3 Hurst-consistency `C`

`a_t = L_r / L_h` (time magnification), `a_p = R_r / R_h` (amplitude
magnification). A true self-affine recurrence satisfies `a_p = a_t^H`. We penalize
by log-distance from that law, tolerance ~1 e-fold. `C=1` at perfect obedience;
`C→0` far away; `C=0.5` (neutral) when `H` is unavailable. **This term is what
distinguishes fractal recurrence from ordinary chart-pattern lookalikes** and is
the model's main novelty.

---

## 4. Projection (`projection.py`)

For each match, transport the **continuation** (bars that followed the historical
occurrence) into the present:

- horizon in the past = `horizon_now / a_t`
- rescale log-return increments by the **geometric mean** of the fractal law
  `a_t^H` and the observed `a_p` (blends theory with what actually happened)
- stretch in time to the forecast horizon via interpolation

Ensemble the transported continuations (score-weighted) → **weighted-median path**
+ 20/80 percentile **bands**. Buy zone = lowest median point in the first 60% of
the horizon; sell zone = highest median point after the buy day.

### 4.1 Confidence composite (each term ∈ [0,1])

```
confidence = 0.40·match_quality      # mean motif score
           + 0.25·ensemble_agreement # exp(−2·median band width);  tight → 1
           + 0.15·fractal_richness   # Δα clipped to [0,1]
           + 0.20·dfa_fit_r2         # is the series even a clean fractal?
```

Scales run via `project_all_scales` over a 5-rung ladder (≈3mo → 4yr), returned
ranked by confidence — this is the "all scales, ranked by confidence" mode.

---

## 5. 3D visualization (`viz3d.py`)

Axes: **time × log₁₀ shares outstanding × log₁₀ price**. Price path = 3D ribbon.
Each motif = translucent `Mesh3d` box spanning its time/price extent with depth
over the window's share-count range (padded to a visible minimum, since share
counts barely move inside a window) — the 3D generalization of the hand-drawn
rectangles. A 2D view (`fractal_figure_2d`) reproduces the owner's boxed-chart
style for direct comparison.

**Pattern-family coloring.** Color now encodes *identity*, not match rank:
every motif occurrence (historical windows from all five scales, plus each
scale's live window) is clustered in shape space by greedy leader clustering
(`motif.group_families`, Pearson ≥ 0.85 to the family leader), and each family
gets one palette color. Boxes sharing a color are the same recurring fractal;
the translucent box of that color is the live pattern the solid boxes refer
to. The 2D view letters the families (`A` = historical occurrence, `A′` = live
window). One legend entry per family (grouped via `legendgroup`), per-box
detail on hover.

The depth axis was switched from volume to shares outstanding so the geometry
models **company valuation instead of order flow**: `log₁₀ cap = log₁₀ price +
log₁₀ shares`, so the ribbon's combined height is log market cap, and buybacks/
dilution bend the path where pure price cannot see them. The cost is geometry —
shares outstanding is a near-flat step function, hence the minimum box depth —
and coverage: Yahoo's filings-derived share history (`get_shares_full`) rarely
reaches back more than ~2 years, so earlier dates are back-filled with the
oldest known count, and tickers with no share data at all (indices, some
crypto) render with a flat shares axis. Hover text reports per-day price,
shares, and market cap. Fetching lives in `data.get_shares` with a weekly-TTL
parquet cache beside the price cache.

The volume-vs-shares question was revisited when family coloring landed and
the choice is **shares outstanding, affirmed**: motif detection runs on price
shape alone, so the depth axis carries context rather than signal, and stable
valuation context beats daily volume noise, which log-smoothing had mostly
flattened into fake geometry anyway. Volume remains in the cached OHLCV data
for any future analysis that wants it.

**Option-chain fractal view** (`option_chain_figure_3d`). A button in the
Visualizer swaps the entire chart for the listed option chain in
(days-to-expiry × log₁₀ strike × log₁₀ option price) space. Each expiration's
call curve C(K) is one ribbon; ribbons are clustered into shape families with
the same `to_shape`/`group_families` machinery (Pearson ≥ 0.90 over log-price
shape), so expiries whose strike-geometry is the same fractal shape share a
color — the near-homogeneity of the option surface across maturities, made
visible. Puts render dimmer in their expiry's family color; the spot price is
a dotted floor line. Data comes from `data.get_option_chain`: up to 10
expirations sampled evenly across the listed curve (front week to LEAPS),
bid/ask midpoint where two-sided, last trade otherwise, hourly-TTL parquet
cache. Tickers without listed options get a friendly message instead of a
chart.

The space under each call ribbon is filled with a curtain `Mesh3d` shaded by
**model-expected profit at expiry**: buy the call at today's premium, settle
at the fractal projection's spot for that expiry (`app._model_prices_at_expiries`
maps each expiry's calendar days to business days on the best-confidence
median path, using the longest-horizon scale beyond it and carrying the last
value flat past every horizon; with no projections it falls back to today's
spot, i.e. intrinsic − premium). Curtain intensity is per-strike P/L on a
red→neutral→green diverging scale centered at 0, symmetric range across all
expiries so color is comparable; the same P/L appears on the call ribbons'
hover. This is the projection layer's opinion made volumetric — and it
inherits all of the projection layer's measured error (§6 still applies).

**Percent loading bars.** Chart renders report real stage-based progress
instead of an indeterminate spinner: `project_all_scales`, `walk_forward`,
`top_following_fractals`, and `get_option_chain` all take an optional
`progress(done, total, label)` callback; the app writes stages into a
module-level dict which a 350 ms `dcc.Interval` polls to draw the bar
(Flask runs threaded, so polling proceeds while a render computes — fine for
a single-user local app).

---

## 6. Falsification (`backtest.py`) — do not skip

Anchored walk-forward: at each anchor `T` (stepped `horizon/2`), the model sees
only bars ≤ `T`, projects, and is graded on `(T, T+horizon]`. Metrics: direction
hit rate, trade hit rate, mean trade return, MAPE@horizon, **naive-drift baseline
for direction and MAPE**, `beats_baseline` flag, and **confidence↔accuracy
correlation**. The app shows the verdict verbatim, green or red. If a change makes
projections prettier but `beats_baseline` regresses, the change is bad.

---

## 7. Universe scan (`scanner.py`)

Threaded scan of ~45 default tickers (equities, crypto, indices) → best-scale
confidence per name → Top-10 table with buy price, sell target, timeframe, `H`.
Extend by editing `DEFAULT_UNIVERSE`.

---

## 8. Application (`app.py`)

Dash + Plotly, local only (`127.0.0.1:8050`). Tabs: **Visualizer** (search any
ticker → 3D + 2D + per-scale cards), **Top 10 Fractals** (scan), **Backtest**
(falsification verdict). Every surface carries the not-financial-advice framing.

Run: `pip install -r requirements.txt && python -m fractal_model.app`

---

## 9. Open problems / where to take it next (for Opus)

1. **Beat the baseline.** Current direction edge is negative at tested scales.
   Levers: (a) condition motif search on regime (only trade when `H` is
   decisively >0.5); (b) require minimum `Δα` before emitting a projection;
   (c) DTW instead of fixed-grid resampling for shape matching; (d) weight
   matches by recency and by Hurst-consistency more aggressively.
2. **Confidence calibration.** `conf↔accuracy ≈ +0.16` is promising but weak.
   Fit an isotonic/Platt calibrator on walk-forward output so displayed
   confidence equals empirical hit probability.
3. **Statistical significance.** Add block-bootstrap / stationary-bootstrap
   surrogate tests: shuffle-preserving autocorrelation, confirm detected motifs
   beat surrogates. Without this, apparent skill may be data-mining.
4. **Multiple-testing control.** The scanner ranks by max confidence over 5
   scales × 45 tickers → selection bias. Apply a false-discovery correction to
   the Top-10.
5. **Intraday / higher-frequency fractals** and **cross-asset motif transfer**
   (does BTC's fractal predict NFLX's?) are natural extensions of the owner's
   "hiding in plain sight" thesis.
6. **Wavelet leaders** for a cleaner multifractal spectrum than MF-DFA.

---

## 10. File map

```
fractal_model/
  data.py          # sources, cache, cleaning
  hurst.py         # DFA + R/S Hurst
  multifractal.py  # MF-DFA, Δα
  motif.py         # shape space + scale-invariant matcher + Hurst-consistency
  projection.py    # motif transport, ensemble path, buy/sell, confidence, scales
  backtest.py      # walk-forward + naive baseline (falsification)
  scanner.py       # universe scan → Top-10
  viz3d.py         # 3D + 2D Plotly figures
  app.py           # Dash desktop app
docs/PROJECT_SPEC.md  # this file
tests/test_smoke.py
```

## 11. References

- Mandelbrot, B. (1963). The variation of certain speculative prices.
- Hurst, H.E. (1951). Long-term storage capacity of reservoirs.
- Peng et al. (1994). DFA of DNA nucleotides.
- Kantelhardt et al. (2002). Multifractal DFA of nonstationary time series.
- Mandelbrot & Wallis (1969). Robustness of R/S.

*Nothing in this project is financial advice. Projections are pattern-implied
scenarios with measured error, presented with explicit confidence, and are not
forecasts.*
