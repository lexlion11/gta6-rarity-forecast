# GTA VI Trophy Rarity Forecast

Entry for the [make.fun](https://make.fun) **gta6-trophy-rarity-forecast-01** bounty:
forecast every Grand Theft Auto VI (PS5) trophy's Exophase earn rate at
2026-11-26T00:00:00Z, as five quantiles (5 / 25 / 50 / 75 / 95 percent).

## What it does

`node main.ts` reads every `*.in` case file from `$MAKEFUN_INPUT_DIR` (or
`/input/`) and writes one `<id>.out` JSON file per case to
`MAKEFUN_OUTPUT_DIR` (or `/output/`):

```json
{"trophies":[{"id":4,"q":[0.1931,0.5403,1.1,2.2264,6.0088]}]}
```

One row per trophy, five finite, non-decreasing numbers in 0..100 — the
quantiles of that trophy's earn rate (percent of tracked players who earned
it) at the case's target time. Node 22 standard library only, no network, no
randomness: two runs of the same case are byte-identical, and a case finishes
in well under a second.

## The model

The house baseline freezes every rate at its last read and scores ~1.51 on the
practice because a launch-week rate is anything but flat. This entry forecasts
the two things that actually move:

**Segments, not reads.** Consecutive reads with an identical
`(players_tracked, rates)` state are one Exophase refresh state; the history
collapses to the segments where something changed.

**Implied earners.** For every segment, `E = rate/100 * players_tracked` — the
signal free of the denominator's jumps.

**Marginal-earner progression.** Per trophy, the fraction of each new batch of
tracked players that has earned it, `m = dE/dP`, is estimated over the recent
segment steps (weighted by step size), with a damped extrapolation of its own
trend. The step from the launch-night segment to the first refresh is a
structural one-off dilution, not progression, and is excluded from those
statistics. The tracked player base is extrapolated with damped linear growth.
Then:

```
E_fc = E_last + m_fc * (P_fc - P_last)
rate_fc = E_fc / P_fc * 100
```

The model is self-damping by construction: `m -> 1` saturates a trophy toward
high rates (everyone earns it eventually), `m -> 0` lets the growing
denominator dilute it down — exactly the launch-week dynamics Exophase shows.

**Calibrated spread.** Quantiles are `sigmoid(median_logit + z * sigma)` with
`sigma = 0.4 * sqrt(horizon_days)` (narrower than the house's 0.5, which is
wide enough to pay at every trophy), bumped for rare trophies, thin data, and
recent logit volatility.

**Launch-night dilution prior.** When the entire history shows a single
`players_tracked` (no refresh ever observed — the practice cases), the median
shifts down in logit space and the band widens, pricing the dilution a refresh
of a freshly launched game's counts brings.

The 100% trophy is held at 100: every refresh observed kept it there.

## Run it

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## License

MIT — see [LICENSE](LICENSE).

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01) [![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lexlion11/gta6-rarity-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01) [![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lexlion11/gta6-rarity-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
