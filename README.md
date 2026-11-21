# GTA VI Trophy Rarity Forecast

Entry for the [make.fun](https://make.fun) **gta6-trophy-rarity-forecast-01** bounty:
forecast the Exophase earn rate (percent of tracked players who earned each
trophy) for every PlayStation 5 trophy of Grand Theft Auto VI at
2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%).

## What it does

`node main.ts` reads every `*.in` case file from `$MAKEFUN_INPUT_DIR`
(default `/input/`) and writes one `<id>.out` per case to
`$MAKEFUN_OUTPUT_DIR` (default `/output/`). Each case is a JSON object with
the trophy list and the house's recorded reads of Exophase's trophy page
(players tracked + per-trophy earn rates, oldest first). Each output is:

```json
{"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}, ...]}
```

— one row per trophy, five finite, non-decreasing numbers in 0..100.
Deterministic (no randomness), standard library only, no network, ~0.2 s
per run (sandbox budget: 8 s).

## The model

Everything is done in **logit space** (`logit(p) = ln(p/(1-p))`, the scale
the pinball-loss score is computed on), per trophy, per case:

**Model A — logit-space trend.** Between Exophase refreshes the page is
frozen, so only reads where the rate *or* the players-tracked count changed
carry signal. A recency-weighted (24 h half-life) least-squares line is fit
to `logit(rate)` vs `ln(1 + hours since release)` over those refresh points
and extrapolated to the target with **horizon damping**
(`w = 1/(1 + Δln/2)`): adoption curves decelerate, so only a shrinking share
of the recent trend is carried over a multi-day horizon.

**Model B — earners/players separation.** Implied earners
`E = rate/100 × players_tracked` are extrapolated with a damped log-linear
growth (growth rate decaying as `1/(1 + Δt/48h)`), while `players_tracked`
is extrapolated with a damped linear fit over its step pattern. The median
rate is `Ê/P̂` — free of the denominator's jumps when Exophase adds players.
B is blended 50/50 with A in logit space, capped at 1.2 logit units away
from A so an exponential earner run-away cannot dominate the median.

**Median shrinkage.** The extrapolated median is shrunk back toward the
last-read anchor when the trend fit is noisy
(`shrink = 1/(1 + σ_fit/0.6)`).

**Spread.** `sqrt(σ_fit² + 0.22² + (0.0042·Δt)² + (0.25·|move|)²)` in logit
units, plus a model-disagreement term when A and B diverge, clamped to
[0.20, 0.85]. Quantiles are `sigmoid(median + z_τ · spread)` with the
standard normal quantiles, rounded to 4 decimals.

**Launch-night dilution prior.** When the history has never seen a
players-tracked refresh (the flat launch-night page), the median is shaded
down 0.3 logit units and the spread set to 0.55: an Exophase refresh before
the target dilutes every rate but the 100% one, and the pinball loss on the
logit scale makes rare trophies matter as much as common ones.

The trophy sitting at 100% (an auto-earn) is forecast to stay at 100.

## How to run

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with *.in case files> MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22.18+ runs the TypeScript directly; on Node 22.6–22.17 use
`node --experimental-strip-types main.ts`.

## Verification

- Practice cases `p00`/`p01`: valid output, 55 rows each, ids unique,
  quantiles finite, non-decreasing, in 0..100.
- Synthetic final case `f00`: valid output, story trophies extrapolated up
  with damped trends, wide honest 90% bands.
- Determinism: two runs into separate directories produce byte-identical
  outputs (`diff -r`).
- Runtime ~0.2 s per case (limit 8 s); output ~2.8 KB per case (limit
  65,536 bytes).

## License

MIT (see `LICENSE`).

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01) [![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lexlion11/gta6-rarity-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01) [![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lexlion11/gta6-rarity-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
