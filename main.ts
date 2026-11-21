// GTA VI PS5 trophy earn-rate forecast — make.fun gta6-trophy-rarity-forecast-01
//
// Reads every *.in case file from $MAKEFUN_INPUT_DIR (default /input/) and
// writes one <id>.out per case to $MAKEFUN_OUTPUT_DIR (default /output/).
// Output per case: {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}.
//
// Forecast model (per trophy, per case), all in logit space
// (logit(p) = ln(p/(1-p)), p = rate/100, clamped 0.01..99.99):
//
//   A. Logit-space trend: weighted least squares of logit(rate) against
//      ln(1 + hours since release) over the reads where the trophy's rate or
//      the players-tracked count changed (the refresh points — between
//      refreshes Exophase's page is frozen, so only those reads carry
//      signal). Recency-weighted (half-life ~24h). The fitted slope is
//      extrapolated to the target with horizon damping: adoption curves
//      decelerate, so only a shrinking share of the recent trend is carried
//      over long horizons (w = 1/(1 + dln/2)).
//
//   B. Earners/players separation: implied earners E = rate/100 *
//      players_tracked are extrapolated with a damped log-linear growth
//      (growth rate decays as 1/(1 + dt/48h)); players_tracked is
//      extrapolated with a damped linear fit over its step pattern. The
//      median rate is E-hat / P-hat, which is free of the denominator's
//      jumps. B is blended 50/50 with A in logit space, with B capped at
//      1.2 logit units away from A so an exponential earner run-away
//      cannot dominate the median.
//
//   Spread: sqrt(fit-residual^2 + base^2 + (k*horizon)^2) in logit units,
//   plus a model-disagreement term when A and B diverge. When the history
//   has never seen a players-tracked refresh (launch-night flat page), a
//   dilution prior applies: the median is shaded down and the spread
//   widened, because an Exophase refresh before the target dilutes every
//   rate but the 100% one.
//
// Deterministic (no randomness at all), Node standard library only, no
// network. Runs in well under the 8-second sandbox budget.

import * as fs from 'node:fs';
import * as path from 'node:path';

// ---------- types ----------

interface TrophyRow {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface HistRead {
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
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: TrophyRow[];
  history: HistRead[];
  notes?: string;
}

// ---------- constants ----------

// Standard normal quantiles for the five forecast levels.
const Z_STD = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

const LOGIT_LO = Math.log(0.0001 / 0.9999); // logit(0.01%)  ~ -9.21
const LOGIT_HI = -LOGIT_LO; // logit(99.99%) ~ +9.21

// Model constants (see header comment).
const W_TAU_RATE = 24; // recency half-life (hours) for rate/earner fits
const W_TAU_PLAYERS = 36; // recency half-life (hours) for the players fit
const DAMP_A = 2.0; // logit-trend damping scale in ln(1+h) units
const DAMP_B_HOURS = 48; // earner-growth damping time constant (hours)
const DAMP_P = 0.75; // players growth damping factor
const P_MAX_MULT = 8; // cap on players growth vs last read
const BLEND_CAP = 1.2; // max |logitB - logitA| allowed in the blend
const SPREAD_BASE = 0.22; // base logit spread
const SPREAD_K = 0.0042; // logit spread per hour of horizon
const SPREAD_MOVE = 0.25; // logit spread per unit of extrapolated move
const SPREAD_MIN = 0.2;
const SPREAD_MAX = 0.85;
const SHRINK_SIGMA = 0.6; // trend-noise scale for shrinking the extrapolation
const FLAT_DILUTION_SHIFT = -0.3; // median shift (logit) when no refresh ever seen
const FLAT_SPREAD = 0.55; // spread when no refresh ever seen

// ---------- small helpers ----------

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function logitPct(p: number): number {
  const x = clamp(p, 0.01, 99.99) / 100;
  return Math.log(x / (1 - x));
}

function sigmoidPct(l: number): number {
  return 100 / (1 + Math.exp(-l));
}

// Weighted least squares of y on x. Returns intercept, slope and the
// weighted residual std (0 for a perfect/degenerate fit).
function wlsq(xs: number[], ys: number[], ws: number[]): { intercept: number; slope: number; resid: number } {
  const n = xs.length;
  if (n === 0) return { intercept: NaN, slope: NaN, resid: NaN };
  let sw = 0;
  let swx = 0;
  let swy = 0;
  for (let i = 0; i < n; i++) {
    sw += ws[i];
    swx += ws[i] * xs[i];
    swy += ws[i] * ys[i];
  }
  if (sw <= 0) return { intercept: NaN, slope: NaN, resid: NaN };
  const mx = swx / sw;
  const my = swy / sw;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxy += ws[i] * dx * (ys[i] - my);
    sxx += ws[i] * dx * dx;
  }
  if (sxx <= 1e-12) return { intercept: my, slope: 0, resid: 0 };
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let sr = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    sr += ws[i] * r * r;
  }
  return { intercept, slope, resid: Math.sqrt(sr / sw) };
}

// Inverse standard normal CDF (Acklam's approximation), for non-standard
// level vectors (the rules fix [0.05,0.25,0.5,0.75,0.95], but be robust).
function normInv(p: number): number {
  if (p <= 0) return -1e300;
  if (p >= 1) return 1e300;
  const a = [-39.969683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let z: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= phigh) {
    q = p - 0.5;
    const r = q * q;
    z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  return z;
}

function zForLevels(levels: number[]): number[] {
  if (
    levels.length === 5 &&
    Math.abs(levels[0] - 0.05) < 1e-9 &&
    Math.abs(levels[1] - 0.25) < 1e-9 &&
    Math.abs(levels[2] - 0.5) < 1e-9 &&
    Math.abs(levels[3] - 0.75) < 1e-9 &&
    Math.abs(levels[4] - 0.95) < 1e-9
  ) {
    return Z_STD;
  }
  return levels.map(normInv);
}

function round4(v: number): number {
  return Math.round(v * 10000) / 10000;
}

// ---------- per-case forecast ----------

function forecastCase(c: CaseFile): { trophies: { id: number; q: number[] }[] } {
  const releaseMs = Date.parse((c.game && c.game.release_date ? c.game.release_date : '2026-11-19') + 'T00:00:00Z');
  const reads = [...c.history].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  const z = zForLevels(c.levels);

  const tH = reads.map((r) => (Date.parse(r.at) - releaseMs) / 3600000); // hours since release
  const tLast = tH.length > 0 ? tH[tH.length - 1] : 0;
  const tTarget = (Date.parse(c.target) - releaseMs) / 3600000;
  const dt = Math.max(tTarget - tLast, 1 / 60); // hours from last read to target
  const dLn = Math.log(1 + Math.max(tTarget, 0)) - Math.log(1 + Math.max(tLast, 0));

  const P = reads.map((r) => r.players_tracked);
  const flatP = P.length > 0 && P.every((v) => v === P[0]);
  const PLast = P.length > 0 ? P[P.length - 1] : 1;

  // ---- players_tracked forecast (step pattern -> damped linear fit) ----
  let Phat = PLast;
  if (!flatP && P.length >= 2) {
    const idx: number[] = [];
    for (let j = 0; j < P.length; j++) {
      if (j === 0 || P[j] !== P[j - 1] || j === P.length - 1) idx.push(j);
    }
    const fit = wlsq(
      idx.map((j) => tH[j]),
      idx.map((j) => P[j]),
      idx.map((j) => Math.exp(-(tLast - tH[j]) / W_TAU_PLAYERS)),
    );
    if (isFinite(fit.slope)) {
      const growth = fit.slope * dt * DAMP_P;
      Phat = PLast + Math.max(growth, 0);
      Phat = Math.min(Phat, PLast * P_MAX_MULT);
    }
  }

  // ---- per-trophy forecast ----
  const rows = c.trophies.map((tr, i): { id: number; q: number[] } => {
    // Observations for this trophy: reads where its rate or the player base
    // changed (refresh points), plus the last read that lists it. Between
    // refreshes the page is frozen, so duplicated reads carry no signal.
    const obs: { t: number; rate: number; E: number }[] = [];
    let prevRate: number | null = null;
    let prevP: number | null = null;
    let lastNonNull = -1;
    for (let j = 0; j < reads.length; j++) {
      const r = reads[j].rates[i];
      if (r === null || r === undefined || !isFinite(r)) continue;
      lastNonNull = j;
      const Pj = P[j];
      if (obs.length === 0 || r !== prevRate || Pj !== prevP) {
        obs.push({ t: tH[j], rate: r, E: (r / 100) * Pj });
        prevRate = r;
        prevP = Pj;
      }
    }
    // Make sure the most recent observation is represented.
    if (lastNonNull >= 0) {
      const r = reads[lastNonNull].rates[i] as number;
      const last = obs[obs.length - 1];
      if (!last || last.t !== tH[lastNonNull]) {
        obs.push({ t: tH[lastNonNull], rate: r, E: (r / 100) * P[lastNonNull] });
      }
    }

    // Fallback when there is no data at all for this trophy.
    if (obs.length === 0) {
      return { id: tr.id, q: [0.01, 0.01, 0.01, 0.01, 0.01] };
    }

    const rLast = obs[obs.length - 1].rate;
    const ELast = obs[obs.length - 1].E;
    const anchor = logitPct(rLast);

    // A trophy sitting at 100% stays there (the launch-night refresh left
    // the 100% trophy untouched; it is an auto-earn trophy).
    if (rLast >= 99.99) {
      return { id: tr.id, q: [100, 100, 100, 100, 100] };
    }

    // ---- Model A: logit(rate) trend vs ln(1 + hours since release) ----
    let logitA = anchor;
    let sigmaA = 0;
    if (obs.length >= 2) {
      const fit = wlsq(
        obs.map((o) => Math.log(1 + Math.max(o.t, 0))),
        obs.map((o) => logitPct(o.rate)),
        obs.map((o) => Math.exp(-(tLast - o.t) / W_TAU_RATE)),
      );
      if (isFinite(fit.slope) && isFinite(fit.intercept)) {
        const wA = 1 / (1 + dLn / DAMP_A);
        logitA = anchor + fit.slope * dLn * wA;
        sigmaA = fit.resid;
      }
    }
    logitA = clamp(logitA, LOGIT_LO, LOGIT_HI);

    // ---- Model B: earners / players extrapolation ----
    let logitB = NaN;
    if (obs.length >= 2 && ELast > 0 && Phat > 0) {
      const fit = wlsq(
        obs.map((o) => o.t),
        obs.map((o) => Math.log(o.E)),
        obs.map((o) => Math.exp(-(tLast - o.t) / W_TAU_RATE)),
      );
      if (isFinite(fit.slope)) {
        const g = fit.slope * (1 / (1 + dt / DAMP_B_HOURS));
        let Ehat = ELast * Math.exp(g * dt);
        Ehat = Math.min(Ehat, Phat); // earners cannot exceed tracked players
        Ehat = Math.max(Ehat, ELast * 0.9); // earners do not vanish
        const rateB = (Ehat / Phat) * 100;
        if (isFinite(rateB) && rateB > 0) {
          logitB = logitPct(rateB);
        }
      }
    }

    // ---- Blend A and B (B capped near A) ----
    let logitMed = logitA;
    let disagree = 0;
    if (isFinite(logitB)) {
      const logitBc = clamp(logitB, logitA - BLEND_CAP, logitA + BLEND_CAP);
      disagree = Math.abs(logitA - logitBc);
      logitMed = 0.5 * (logitA + logitBc);
    }

    // ---- Launch-night dilution prior (no refresh ever observed) ----
    let spread: number;
    if (flatP) {
      logitMed += FLAT_DILUTION_SHIFT;
      spread = FLAT_SPREAD;
    } else {
      // Shrink the extrapolation toward the anchor when the trend fit is
      // noisy: an unreliable slope should not move the median fully.
      const shrink = 1 / (1 + sigmaA / SHRINK_SIGMA);
      logitMed = anchor + shrink * (logitMed - anchor);
      const move = Math.abs(logitA - anchor);
      spread = Math.sqrt(
        sigmaA * sigmaA +
          SPREAD_BASE * SPREAD_BASE +
          (SPREAD_K * dt) * (SPREAD_K * dt) +
          (SPREAD_MOVE * move) * (SPREAD_MOVE * move),
      );
      spread = Math.sqrt(spread * spread + (0.5 * disagree) * (0.5 * disagree));
      spread = clamp(spread, SPREAD_MIN, SPREAD_MAX);
    }
    logitMed = clamp(logitMed, LOGIT_LO, LOGIT_HI);

    // ---- Quantiles ----
    const q: number[] = [];
    for (let k = 0; k < z.length; k++) {
      let v = sigmoidPct(logitMed + z[k] * spread);
      v = clamp(round4(v), 0.01, 99.99);
      q.push(v);
    }
    // Enforce non-decreasing quantiles.
    for (let k = 1; k < q.length; k++) {
      if (q[k] < q[k - 1]) q[k] = q[k - 1];
    }
    return { id: tr.id, q };
  });

  return { trophies: rows };
}

// ---------- fallback (never expected, keeps the output shape valid) ----------

function fallbackCase(c: CaseFile): { trophies: { id: number; q: number[] }[] } {
  const reads = [...c.history].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  return {
    trophies: c.trophies.map((tr, i) => {
      let r: number | null = null;
      for (let j = reads.length - 1; j >= 0; j--) {
        const v = reads[j].rates[i];
        if (v !== null && v !== undefined && isFinite(v)) {
          r = v;
          break;
        }
      }
      const v = r === null ? 0.01 : clamp(round4(clamp(r, 0.01, 99.99)), 0.01, 99.99);
      return { id: tr.id, q: [v, v, v, v, v] };
    }),
  };
}

// ---------- main ----------

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || '/input/';
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || '/output/';
  const files = fs
    .readdirSync(inDir)
    .filter((f) => f.endsWith('.in'))
    .sort();
  fs.mkdirSync(outDir, { recursive: true });
  for (const f of files) {
    const raw = fs.readFileSync(path.join(inDir, f), 'utf8');
    let c: CaseFile;
    try {
      c = JSON.parse(raw) as CaseFile;
    } catch {
      // Unparseable case file: nothing sensible can be written.
      continue;
    }
    try {
      fs.writeFileSync(path.join(outDir, c.id + '.out'), JSON.stringify(forecastCase(c)));
    } catch {
      fs.writeFileSync(path.join(outDir, c.id + '.out'), JSON.stringify(fallbackCase(c)));
    }
  }
}

main();
