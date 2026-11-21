// gta6-trophy-rarity-forecast-01 entry
//
// Forecast each trophy's Exophase earn rate (percent of tracked players who earned
// it) at the case's target time, as five quantiles (5/25/50/75/95).
//
// Model (logit space, no randomness, deterministic):
//   1. Collapse the read history into "segments" — consecutive reads with an
//      identical (players_tracked, rates) state are one Exophase refresh state.
//   2. Per trophy, implied earners E = rate/100 * players_tracked.
//   3. Marginal-earner model: the fraction of each new batch of tracked players
//      that has earned the trophy, m = dE/dP, estimated over recent segment
//      pairs (weighted by step size), with a damped extrapolation of its own
//      trend. Player base P is extrapolated with damped linear growth.
//          E_fc = E_last + m_fc * (P_fc - P_last)
//          rate_fc = E_fc / P_fc * 100
//      This is self-damping: m -> 1 saturates the rate toward high values,
//      m -> 0 lets the growing denominator dilute the rate down.
//   4. Quantiles: sigmoid(median_logit + z * sigma) with sigma calibrated to
//      the horizon (random-walk-like widening, narrower than the house's
//      0.5/sqrt(day)) plus bumps for rare trophies and thin data.
//   5. Launch-night dilution prior: when the whole history shows a single
//      players_tracked (no refresh ever observed — the practice cases), shift
//      the median down in logit space and widen, pricing the dilution a
//      refresh of a freshly launched game's counts brings.
//
// Node 22 standard library only. No network, no randomness.

import * as fs from "node:fs";
import * as path from "node:path";

// ---------- types ----------

interface TrophyRow {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface HistoryRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: (number | null)[];
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: TrophyRow[];
  history: HistoryRead[];
}

interface Segment {
  key: string;
  firstMs: number;
  lastMs: number;
  P: number;
  rates: (number | null)[];
}

// ---------- math helpers ----------

const Z_LEVELS = [-1.6448536269514729, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514729];
const CLAMP_LOW = 0.01;
const CLAMP_HIGH = 99.99;

function clampRate(x: number): number {
  return Math.min(CLAMP_HIGH, Math.max(CLAMP_LOW, x));
}

function logit(ratePct: number): number {
  const p = clampRate(ratePct) / 100;
  return Math.log(p / (1 - p));
}

function sigmoid(z: number): number {
  return 100 / (1 + Math.exp(-z));
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return Math.sqrt(s / xs.length);
}

// ---------- history segmentation ----------

function buildSegments(history: HistoryRead[]): Segment[] {
  const segs: Segment[] = [];
  for (const read of history) {
    const key =
      read.players_tracked +
      "|" +
      read.rates.map((r) => (r === null ? "n" : r.toFixed(2))).join(",");
    const last = segs[segs.length - 1];
    if (last !== undefined && last.key === key) {
      last.lastMs = Date.parse(read.at);
    } else {
      segs.push({
        key,
        firstMs: Date.parse(read.at),
        lastMs: Date.parse(read.at),
        P: read.players_tracked,
        rates: read.rates.slice(),
      });
    }
  }
  return segs;
}

// ---------- forecast branches ----------

interface QuantileResult {
  medianLogit: number;
  sigma: number;
}

/**
 * Launch-night dilution prior: the whole history carries one players_tracked
 * (Exophase has never refreshed while we watched). A refresh of a freshly
 * launched game's counts dilutes every rate but the 100% one, rare trophies
 * most in logit terms. Shift the median down, widen the band.
 */
function dilutionBranch(lastRate: number, dDays: number): QuantileResult {
  let delta: number;
  if (lastRate <= 0.1) {
    delta = 0.7;
  } else {
    delta = 0.15 + 0.55 * (1 - Math.min(1, lastRate / 5));
  }
  const sigma = lastRate <= 0.1 ? 1.25 : 1.05;
  return { medianLogit: logit(lastRate) - delta, sigma };
}

/**
 * Time-trend fallback: players_tracked is flat but rates move (Exophase
 * refreshed counts without growing the tracked base). Damped linear trend of
 * logit(rate) per hour.
 */
function timeTrendBranch(
  ratePoints: { tMs: number; rate: number }[],
  hHours: number,
  dDays: number
): QuantileResult {
  const last = ratePoints[ratePoints.length - 1];
  const slopes: number[] = [];
  for (let j = Math.max(0, ratePoints.length - 4); j < ratePoints.length - 1; j++) {
    const dt = (ratePoints[j + 1].tMs - ratePoints[j].tMs) / 3.6e6;
    if (dt > 0) slopes.push((logit(ratePoints[j + 1].rate) - logit(ratePoints[j].rate)) / dt);
  }
  const slope = mean(slopes);
  const damp = 1 / (1 + hHours / 36);
  const medianLogit = logit(last.rate) + slope * hHours * damp;
  const sigma = Math.min(1.1, 0.45 * Math.sqrt(dDays) + 0.15);
  return { medianLogit, sigma };
}

/**
 * Marginal-earner model: m = dE/dP over recent segment steps, with a damped
 * extrapolation of m's own trend; player base extrapolated with damped linear
 * growth; rate = E_fc / P_fc.
 */
function marginalEarnerBranch(
  points: { tMs: number; P: number; E: number; rate: number }[],
  hHours: number,
  dDays: number
): QuantileResult {
  const last = points[points.length - 1];

  // consecutive steps with a growing tracked base
  const steps: { dP: number; dtH: number; m: number; dLogit: number }[] = [];
  for (let j = 0; j < points.length - 1; j++) {
    const dP = points[j + 1].P - points[j].P;
    const dtH = (points[j + 1].tMs - points[j].tMs) / 3.6e6;
    if (dP > 0) {
      const m = (points[j + 1].E - points[j].E) / dP;
      steps.push({
        dP,
        dtH: Math.max(dtH, 1e-6),
        m: Math.min(1, Math.max(0, m)),
        dLogit: logit(points[j + 1].rate) - logit(points[j].rate),
      });
    }
  }

  if (steps.length === 0) {
    return timeTrendBranch(
      points.map((p) => ({ tMs: p.tMs, rate: p.rate })),
      hHours,
      dDays
    );
  }

  // The step from the launch-night segment to the first refresh is a structural
  // one-off dilution (the denominator jumps while earners catch up), not
  // progression; with enough steps, leave it out of the marginal-earner
  // statistics so it does not drag the older baseline down and inflate the
  // trend term.
  const usable = steps.length >= 4 ? steps.slice(1) : steps;
  const recent = usable.slice(-3);
  const older = usable.slice(0, -3);

  const wMean = (ss: { dP: number; m: number }[]) => {
    let num = 0;
    let den = 0;
    for (const s of ss) {
      num += s.m * s.dP;
      den += s.dP;
    }
    return den > 0 ? num / den : 0;
  };

  const mNew = wMean(recent);
  const mOld = older.length > 0 ? wMean(older) : mNew;
  // damped trend of the marginal-earner fraction, bounded to [0, 1]; near
  // saturation (mNew > 0.8) the trend is capped hard since m cannot exceed 1
  const mTrend = mNew > 0.8 ? Math.min(0.1, 0.5 * (mNew - mOld)) : 0.5 * (mNew - mOld);
  const mFc = Math.min(1, Math.max(0, mNew + mTrend));

  // player-base growth: players/hour over recent steps, damped over the horizon
  let dPSum = 0;
  let dtSum = 0;
  for (const s of recent) {
    dPSum += s.dP;
    dtSum += s.dtH;
  }
  const vP = Math.max(0, dtSum > 0 ? dPSum / dtSum : 0);
  const dampP = 1 / (1 + hHours / 72);
  const PFc = Math.max(last.P, Math.min(last.P + vP * hHours * dampP, 12 * last.P));

  // earner extrapolation
  const EFc = Math.min(Math.max(0, last.E + mFc * (PFc - last.P)), PFc);
  const rateFc = (EFc / PFc) * 100;

  let medianLogit: number;
  if (rateFc <= 0.02) {
    // never/nearly never earned: keep a tiny median, the upper quantiles carry
    // the chance the trophy starts appearing before the target
    medianLogit = logit(0.03);
  } else {
    medianLogit = logit(rateFc);
  }

  // calibrated spread: random-walk-like widening with the horizon, bumps for
  // rare trophies and thin data
  let sigma = 0.4 * Math.sqrt(dDays);
  const medianRate = sigmoid(medianLogit);
  if (medianRate < 0.5) sigma += 0.2;
  if (points.length < 4) sigma += 0.1;
  // recent logit volatility adds width: the spread of the recent moves (noise
  // around the trend), not the trend itself
  const vol = Math.min(0.6, std(recent.map((s) => s.dLogit)) * 0.6);
  sigma = Math.min(1.1, sigma + vol * 0.3);

  return { medianLogit, sigma };
}

// ---------- per-case driver ----------

function quantileRow(medianLogit: number, sigma: number): number[] {
  const qs = Z_LEVELS.map((z) => sigmoid(medianLogit + z * sigma));
  // clamp + enforce non-decreasing
  const out: number[] = [];
  let prev = CLAMP_LOW;
  for (const q of qs) {
    let v = round4(Math.min(CLAMP_HIGH, Math.max(prev, q)));
    if (v < prev) v = prev;
    out.push(v);
    prev = v;
  }
  return out;
}

function lastNonNullRate(segs: Segment[], idx: number): number | null {
  for (let s = segs.length - 1; s >= 0; s--) {
    const r = segs[s].rates[idx];
    if (r !== null && r !== undefined) return r;
  }
  return null;
}

function forecastCase(c: CaseFile): { id: number; q: number[] }[] {
  const segs = buildSegments(c.history);
  const hHours = c.horizon_minutes / 60;
  const dDays = hHours / 24;

  const distinctP = new Set(segs.map((s) => s.P));
  const singleP = distinctP.size <= 1;

  const rows: { id: number; q: number[] }[] = [];

  for (let i = 0; i < c.trophies.length; i++) {
    const trophy = c.trophies[i];
    const lastRate = lastNonNullRate(segs, i);

    // safety fallback: never-seen trophy
    if (lastRate === null) {
      rows.push({ id: trophy.id, q: quantileRow(logit(0.03), 1.0) });
      continue;
    }

    // the 100% trophy: every refresh so far kept it at 100 — hold it there
    if (lastRate >= 99.99) {
      rows.push({ id: trophy.id, q: [99.97, 99.99, 100, 100, 100] });
      continue;
    }

    let res: QuantileResult;

    if (singleP) {
      // no refresh ever observed: launch-night dilution prior
      res = dilutionBranch(lastRate, dDays);
    } else {
      // per-trophy points over segments where it was read
      const points: { tMs: number; P: number; E: number; rate: number }[] = [];
      for (const s of segs) {
        const r = s.rates[i];
        if (r !== null && r !== undefined) {
          points.push({ tMs: s.firstMs, P: s.P, E: (r / 100) * s.P, rate: r });
        }
      }
      if (points.length === 0) {
        res = dilutionBranch(lastRate, dDays);
      } else if (points.length === 1) {
        res = { medianLogit: logit(points[0].rate), sigma: Math.min(1.1, 0.4 * Math.sqrt(dDays) + 0.25) };
      } else {
        res = marginalEarnerBranch(points, hHours, dDays);
      }
    }

    rows.push({ id: trophy.id, q: quantileRow(res.medianLogit, res.sigma) });
  }

  return rows;
}

// ---------- main ----------

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output/";

  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();

  fs.mkdirSync(outputDir, { recursive: true });

  for (const file of files) {
    const raw = fs.readFileSync(path.join(inputDir, file), "utf8");
    const c = JSON.parse(raw) as CaseFile;

    let rows: { id: number; q: number[] }[];
    try {
      rows = forecastCase(c);
    } catch (_err) {
      // hard safety net: last-read flat forecast with a moderate spread so the
      // case stays valid instead of crashing out
      const segs = buildSegments(c.history);
      rows = c.trophies.map((t, i) => {
        const r = lastNonNullRate(segs, i) ?? 0.03;
        if (r >= 99.99) return { id: t.id, q: [99.97, 99.99, 100, 100, 100] };
        return { id: t.id, q: quantileRow(logit(r), 0.6) };
      });
    }

    const out = JSON.stringify({ trophies: rows });
    if (Buffer.byteLength(out, "utf8") > 65536) {
      throw new Error(`output too large for case ${c.id}`);
    }
    fs.writeFileSync(path.join(outputDir, c.id + ".out"), out);
  }
}

main();
