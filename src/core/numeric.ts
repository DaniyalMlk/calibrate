/**
 * Small numeric helpers shared across the engine.
 *
 * Everything here is deliberately dependency-free and total: functions either
 * return a finite number or throw, so callers never have to reason about `NaN`
 * leaking through a likelihood.
 */

/**
 * The normal-metric scaling constant. Multiplying a logistic exponent by 1.702
 * brings the two-parameter logistic curve within 0.01 of the normal-ogive curve
 * across the whole ability range, which is why item parameters calibrated under
 * a normal-ogive model can be reused in a logistic one.
 */
export const NORMAL_METRIC_SCALE = 1.702;

/** The logistic metric, i.e. no rescaling of the exponent. */
export const LOGISTIC_METRIC_SCALE = 1;

/**
 * Numerically stable standard logistic function, 1 / (1 + exp(-x)).
 *
 * The naive form overflows for large negative `x`; branching on the sign keeps
 * `exp` applied to a non-positive argument, so the result underflows to zero
 * instead of blowing up to infinity.
 */
export function logistic(x: number): number {
  if (!Number.isFinite(x)) {
    if (Number.isNaN(x)) throw new RangeError('logistic: argument is NaN');
    return x > 0 ? 1 : 0;
  }
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

/** Derivative of the standard logistic function: sigma(x) * (1 - sigma(x)). */
export function logisticDerivative(x: number): number {
  const p = logistic(x);
  return p * (1 - p);
}

/** Restrict `x` to the closed interval [lo, hi]. */
export function clamp(x: number, lo: number, hi: number): number {
  if (lo > hi) throw new RangeError(`clamp: empty interval [${lo}, ${hi}]`);
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

/** Sum of a numeric array, left to right. */
export function sum(values: readonly number[]): number {
  let total = 0;
  for (const v of values) total += v;
  return total;
}

/** Arithmetic mean. Throws on an empty array rather than returning NaN. */
export function mean(values: readonly number[]): number {
  if (values.length === 0) throw new RangeError('mean: empty array');
  return sum(values) / values.length;
}

/** Population variance (divides by n). Throws on an empty array. */
export function variance(values: readonly number[]): number {
  const m = mean(values);
  let acc = 0;
  for (const v of values) acc += (v - m) * (v - m);
  return acc / values.length;
}

/** Root mean square of the deviations of `values` from `target`. */
export function rootMeanSquareError(values: readonly number[], target: readonly number[]): number {
  if (values.length !== target.length) {
    throw new RangeError(
      `rootMeanSquareError: length mismatch (${values.length} vs ${target.length})`,
    );
  }
  if (values.length === 0) throw new RangeError('rootMeanSquareError: empty array');
  let acc = 0;
  for (let i = 0; i < values.length; i += 1) {
    const d = (values[i] as number) - (target[i] as number);
    acc += d * d;
  }
  return Math.sqrt(acc / values.length);
}

/** `count` evenly spaced values covering [start, end] inclusive. */
export function linspace(start: number, end: number, count: number): number[] {
  if (!Number.isInteger(count) || count < 2) {
    throw new RangeError(`linspace: count must be an integer >= 2, received ${count}`);
  }
  const step = (end - start) / (count - 1);
  const out: number[] = new Array<number>(count);
  for (let i = 0; i < count; i += 1) out[i] = start + step * i;
  out[count - 1] = end;
  return out;
}

/** Throw unless `value` is a finite number. `label` names the field in the message. */
export function requireFinite(value: number, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be a finite number, received ${String(value)}`);
  }
  return value;
}

/**
 * Pearson correlation between two equal-length series.
 *
 * Returns `NaN` when either series is constant. That is the honest answer —
 * a correlation is undefined without variation to correlate — and it is
 * reachable in practice: a recovery study run at a single ability point has no
 * spread in the generating values at all.
 */
export function correlation(xs: readonly number[], ys: readonly number[]): number {
  if (xs.length !== ys.length) {
    throw new RangeError(`correlation: length mismatch (${xs.length} vs ${ys.length})`);
  }
  if (xs.length === 0) throw new RangeError('correlation: empty array');
  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    const dx = (xs[i] as number) - mx;
    const dy = (ys[i] as number) - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  if (sxx === 0 || syy === 0) return Number.NaN;
  return sxy / Math.sqrt(sxx * syy);
}
