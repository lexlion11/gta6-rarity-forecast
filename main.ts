// GTA VI PS5 trophy earn-rate forecast — make.fun gta6-trophy-rarity-forecast-01 entry.
//
// Deterministic (no randomness, no network), Node 22 standard library only. Reads every
// *.in case file from $MAKEFUN_INPUT_DIR (or /input/) and writes one <id>.out to
// $MAKEFUN_OUTPUT_DIR (or /output/) per case.
//
// Forecast model, per case:
//   1. Page states. Consecutive reads with the same players_tracked, trophies_earned and
//      rates are the same unchanged Exophase page; each new state is one refresh. Trends
//      are fit over the states after the first change (the launch-night flat block carries
//      no trend information — every read of it is the same page, duplicated).
//   2. Median. Two trend extrapolations of logit(rate), averaged and shrunk toward
//      persistence by how many refresh states were observed:
//      A) a line of logit(rate) against hours, its slope damped by 1/(1 + 0.4 dt/span)
//         because progression slows as the launch burst passes;
//      B) a line of logit(rate) against ln(1 + hours since release), read at the target —
//         the penetration-curve view, which decelerates in time by construction.
//      Both are fit on refresh states (not raw reads), so a long unchanged page weighs
//      one point, not thirty. The median is clamped to a sane band around the last rate.
//   3. Spread. sqrt of a random-walk term (0.26 logit units per sqrt day — narrower than
//      the house baseline's 0.5, because the median is a trend, not persistence), the
//      fit's residual scaled by the extrapolation ratio, and a term for how much the two
//      trend views disagree.
//   4. Launch-night prior. When the whole history is one unchanged page, the tracked base
//      is still the launch-night cohort; Exophase refreshes on its own schedule and grows
//      it (the rules' launch-night refresh took 638 tracked players to 3,868, and every
//      rate but the 100% one lower). The median shifts down by the implied dilution and
//      the spread is wide.
//
// Output: {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]} — one row per trophy, each id
// once, five finite non-decreasing numbers in 0..100.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface RawRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  target: string;
  horizon_minutes: number;
  levels: number[];
  game: { name: string; release_date: string };
  trophies: Trophy[];
  history: RawRead[];
}

/** One unchanged page state: the values Exophase last refreshed to, and when it was first seen. */
interface State {
  t: number;
  P: number;
  earned: number;
  rates: Array<number | null>;
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const logitOf = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99);
  return Math.log(p / (100 - p));
};
const sigmoidPct = (x: number): number => 100 / (1 + Math.exp(-x));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Least squares y = a + b x. Returns [intercept, slope, residual std] (residual std 0 when n < 3). */
function lsq(xs: number[], ys: number[]): [number, number, number] {
  const n = xs.length;
  if (n === 0) return [Number.NaN, Number.NaN, 0];
  if (n === 1) return [ys[0], 0, 0];
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxy += dx * (ys[i] - my);
    sxx += dx * dx;
  }
  if (sxx === 0) return [my, 0, 0];
  const b = sxy / sxx;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    ss += r * r;
  }
  return [a, b, Math.sqrt(ss / Math.max(1, n - 2))];
}

const sameRates = (a: Array<number | null>, b: Array<number | null>): boolean => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if ((a[i] ?? null) !== (b[i] ?? null)) return false;
  }
  return true;
};

function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  const tTarget = Date.parse(c.target);
  const tRelease = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const dDays = c.horizon_minutes / 1440;

  // Collapse the read history to page states: a new state starts whenever the page
  // changed (players tracked, trophies earned, or any rate).
  const states: State[] = [];
  for (const r of c.history) {
    const t = Date.parse(r.slot);
    const last = states[states.length - 1];
    if (
      last === undefined ||
      last.P !== r.players_tracked ||
      last.earned !== r.trophies_earned ||
      !sameRates(last.rates, r.rates)
    ) {
      states.push({ t, P: r.players_tracked, earned: r.trophies_earned, rates: r.rates });
    }
  }

  const lastState = states[states.length - 1];
  if (lastState === undefined) throw new Error(`${c.id} has no history`);

  // The first state that differs from the launch-night page; -1 when the page never
  // changed (the whole history is one unchanged page).
  let firstChange = -1;
  for (let k = 1; k < states.length; k++) {
    const s = states[k];
    if (
      s.P !== states[0].P ||
      s.earned !== states[0].earned ||
      !sameRates(s.rates, states[0].rates)
    ) {
      firstChange = k;
      break;
    }
  }

  const flat = firstChange === -1;
  const active = flat ? states : states.slice(firstChange);
  const tLast = lastState.t;
  const dtH = (tTarget - tLast) / 3_600_000; // hours from the last read to the target
  const spanH = Math.max(
    1,
    (tLast - (flat ? states[0].t : states[firstChange].t)) / 3_600_000,
  );
  const HTarget = Math.max(0, (tTarget - tRelease) / 3_600_000);

  // Launch-night prior (flat history): the rules' documented launch-night refresh took
  // the tracked base from 638 to 3,868 players; earners grow, but by less.
  const FLAT_G_P = 3868 / 638;
  const FLAT_G_E = 2.2;

  const rows = c.trophies.map((tr, i): { id: number; q: number[] } => {
    const rL = lastState.rates[i] ?? 0;
    // A trophy already at 100% stays at 100%.
    if (rL >= 99.99) return { id: tr.id, q: [99.99, 99.99, 99.99, 99.99, 99.99] };

    let med: number;
    let sigma: number;

    if (flat) {
      const rf = clamp((rL * FLAT_G_E) / FLAT_G_P, 0.01, 99.99);
      med = logitOf(rf);
      sigma = 0.65;
    } else {
      // Per-trophy points over the post-refresh states that list this trophy.
      const pts = active
        .filter((s) => s.rates[i] != null)
        .map((s) => ({ t: s.t, r: s.rates[i] as number }));
      const n = pts.length;
      const logitL = logitOf(rL);
      // Damping shared by both trends: a penetration curve decelerates, so the change
      // projected past the last read is a fraction of the change observed over the
      // span — the further the target, the smaller the fraction (span/(span+dt)).
      const g = spanH / (spanH + dtH);
      const HLast = Math.max(0, (tLast - tRelease) / 3_600_000);

      // A) logit(rate) against hours: the fitted slope, damped and anchored at the last read.
      let cA = logitL;
      if (n >= 2) {
        const xs = pts.map((p) => (p.t - tLast) / 3_600_000);
        const ys = pts.map((p) => logitOf(p.r));
        const [, bA] = lsq(xs, ys);
        if (Number.isFinite(bA)) cA = logitL + bA * dtH * g;
      }

      // B) logit(rate) against ln(1 + hours since release): the penetration-curve slope,
      // read from the last read to the target, damped the same way.
      let cB = logitL;
      let rho = 0.08;
      if (n >= 2) {
        const xs = pts.map((p) =>
          Math.log(1 + Math.max(0, (p.t - tRelease) / 3_600_000)),
        );
        const ys = pts.map((p) => logitOf(p.r));
        const [, bB, rB] = lsq(xs, ys);
        if (Number.isFinite(bB)) {
          cB = logitL + bB * (Math.log(1 + HTarget) - Math.log(1 + HLast)) * g;
          rho = Math.max(rB, 0.06);
        }
      }

      // Average the two views, shrunk toward persistence by the refresh count.
      const w = clamp((n - 1) / 3, 0.4, 1);
      const trend = (cA + cB) / 2;
      med = w * trend + (1 - w) * logitL;
      // Keep the median within a sane band of the last read.
      med = clamp(med, logitL - 2.5, logitL + 3.5);
      med = clamp(med, logitOf(0.01), logitOf(99.9));

      const disagree = Math.abs(cA - cB);
      sigma = Math.sqrt(
        Math.pow(0.26 * Math.sqrt(dDays), 2) +
          Math.pow(rho * (1 + dtH / spanH), 2) +
          Math.pow(0.25 * disagree, 2) +
          0.01,
      );
      sigma = clamp(sigma, 0.22, 1.4);
    }

    const q = Z.map((z) => round4(clamp(sigmoidPct(med + z * sigma), 0.01, 99.99)));
    // Non-decreasing by construction (z rising, sigma >= 0); enforce against float edge cases.
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: tr.id, q };
  });

  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
