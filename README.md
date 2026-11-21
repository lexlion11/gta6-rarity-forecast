# GTA VI Trophy Earn-Rate Forecast — make.fun gta6-trophy-rarity-forecast-01 entry

Forecasts each GTA VI PS5 trophy's Exophase earn rate (the percent of tracked players who
have earned it) at a future target time, as five quantiles: 0.05, 0.25, 0.5, 0.75, 0.95.

## What it does

`main.ts` (Node 22, standard library only, no network) reads every `*.in` case file from
`$MAKEFUN_INPUT_DIR` (or `/input/`) and writes one `<id>.out` per case to
`$MAKEFUN_OUTPUT_DIR` (or `/output/`):

```json
{"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}]}
```

One row per trophy, each id once, five finite numbers in `0..100`, non-decreasing.

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Deterministic: no randomness at all, no clock, no network. Two runs write byte-identical
files (verified).

## The model

**1. Collapse the read history into page states.** Exophase refreshes on its own schedule,
so many reads of the same page repeat identical values. A new state starts whenever
`players_tracked`, `trophies_earned` or any rate changes. Trends are then fit over states,
so one unchanged page weighs one point rather than thirty.

**2. Find where the real data starts.** The launch-night block before Exophase's first
refresh carries no trend information — every read of it is the same page. The first state
that differs from the launch page is where the trend begins.

**3. Median: two trend views of logit(rate), averaged and shrunk toward persistence.**

- **A** — a least-squares line of `logit(rate)` against hours, extrapolated from the last
  read to the target. Its slope is damped by `span / (span + dt)`, because a penetration
  curve decelerates: the further out the target is, the smaller a fraction of the observed
  change carries.
- **B** — a least-squares line of `logit(rate)` against `ln(1 + hours since release)`,
  read at the target. This is the penetration-curve view, which decelerates in time by
  construction.

The two are averaged and shrunk toward pure persistence by how many refresh states were
observed (`w = clamp((n-1)/3, 0.4, 1)`): with one refresh there is little to extrapolate
from, with several the trend deserves the weight. The median is then held inside a sane
band around the last read.

**4. Spread.** The standard deviation combines three terms, all in logit space:

- a random-walk term, `0.26 * sqrt(horizon in days)` — narrower than the house baseline's
  `0.5`, because the median here is a trend rather than persistence;
- the trend fit's own residual, scaled by the extrapolation ratio `1 + dt/span`;
- a term for how far the two trend views disagree, `0.25 * |A - B|`.

It is clamped to `[0.22, 1.4]`. The quantiles are then `sigmoid(median + z * sigma)` at
`z = [-1.645, -0.674, 0, 0.674, 1.645]`.

**5. Launch-night prior.** When the entire history is one unchanged page, the tracked base
is still the launch-night cohort, and the rules document what the first refresh did: 638
tracked players became 3,868, and every rate but the 100% one fell. The model applies that
implied dilution to the median and widens the spread to `sigma = 0.65`.

**6. Already-complete trophies.** A trophy already at 100% is reported as 100% at every
quantile.

The scoring is mean pinball loss on the logit scale, where a rare trophy (1%) weighs as
much as a common one (50%): the model therefore works entirely in logit space, and its
spread is deliberately not narrow — a narrow spread pays for every miss on a rare trophy.

## Layout

```
main.ts        entrypoint: reads the case files, forecasts each, writes <id>.out
LICENSE        MIT
```

## make.fun

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lexlion11/gta6-rarity-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lexlion11/gta6-rarity-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
