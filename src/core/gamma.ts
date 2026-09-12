/**
 * The gamma function, the regularized incomplete gamma function, and the tail
 * probabilities built on them.
 *
 * Every significance test in the engine ends at the same place: a statistic and
 * the probability of exceeding it under a null. For the chi-square family that
 * probability is the regularized upper incomplete gamma function `Q(k/2, x/2)`,
 * and for the normal it is a complementary error function, which is the same
 * object at a half-integer argument. Implementing one gives both.
 *
 * Written rather than taken from a dependency for the same reason as everything
 * else here: a test statistic whose p-value comes from an unexamined black box
 * is a test statistic nobody can defend. The two series below are the standard
 * pair — an ascending series that converges quickly below the mean and a
 * continued fraction that converges quickly above it — and the crossover between
 * them is the only real decision.
 */

import { requireFinite } from './numeric.js';

/**
 * Lanczos coefficients at g = 7, n = 9.
 *
 * Accurate to roughly 15 significant digits over the half-plane this uses,
 * which is more than the statistics downstream can make use of.
 */
const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
] as const;

/**
 * Natural log of the gamma function, for positive real arguments.
 *
 * The log rather than the function itself because the values wanted here run
 * far past the range of a double long before the argument gets interesting:
 * `gamma(172)` overflows, while `logGamma(172)` is about 711.
 */
export function logGamma(x: number): number {
  requireFinite(x, 'logGamma: x');
  if (x <= 0) throw new RangeError(`logGamma: x must be positive, received ${x}`);

  // The reflection formula, for the half-plane the Lanczos sum does not cover.
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }

  const z = x - 1;
  let series = LANCZOS[0] as number;
  for (let i = 1; i < LANCZOS.length; i += 1) {
    series += (LANCZOS[i] as number) / (z + i);
  }
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(series);
}

/** Iteration cap for both series. Neither has ever come close in practice. */
const MAX_ITERATIONS = 1000;
const EPSILON = 1e-15;

/**
 * The ascending series for `P(a, x)`, used where `x` is below `a + 1`.
 *
 * Converges in a number of terms that grows with `x`, which is why it is only
 * used on the near side of the mean.
 */
function lowerSeries(a: number, x: number): number {
  if (x === 0) return 0;
  let term = 1 / a;
  let total = term;
  for (let n = 1; n <= MAX_ITERATIONS; n += 1) {
    term *= x / (a + n);
    total += term;
    if (Math.abs(term) < Math.abs(total) * EPSILON) break;
  }
  return total * Math.exp(-x + a * Math.log(x) - logGamma(a));
}

/**
 * The continued fraction for `Q(a, x)`, used where `x` is at or above `a + 1`.
 *
 * Evaluated by the modified Lentz method, which runs the fraction forwards and
 * handles the zero denominators that would otherwise stop it by nudging them to
 * a tiny value rather than by special-casing.
 */
function upperFraction(a: number, x: number): number {
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;

  for (let i = 1; i <= MAX_ITERATIONS; i += 1) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < EPSILON) break;
  }

  return h * Math.exp(-x + a * Math.log(x) - logGamma(a));
}

/**
 * The regularized lower incomplete gamma function `P(a, x)`.
 *
 * Runs from 0 at `x = 0` to 1 as `x` grows. This is the chi-square cumulative
 * distribution at `a = k/2`, `x = statistic/2`.
 */
export function lowerGamma(a: number, x: number): number {
  requireFinite(a, 'lowerGamma: a');
  requireFinite(x, 'lowerGamma: x');
  if (a <= 0) throw new RangeError(`lowerGamma: a must be positive, received ${a}`);
  if (x < 0) throw new RangeError(`lowerGamma: x must not be negative, received ${x}`);
  return x < a + 1 ? lowerSeries(a, x) : 1 - upperFraction(a, x);
}

/**
 * The regularized upper incomplete gamma function `Q(a, x) = 1 - P(a, x)`.
 *
 * Computed from whichever series is accurate in its own right rather than by
 * subtracting the other from one. That distinction is the whole point in the far
 * tail: `1 - P` loses every significant digit once `P` rounds to 1, and the far
 * tail is exactly where a p-value is being read.
 */
export function upperGamma(a: number, x: number): number {
  requireFinite(a, 'upperGamma: a');
  requireFinite(x, 'upperGamma: x');
  if (a <= 0) throw new RangeError(`upperGamma: a must be positive, received ${a}`);
  if (x < 0) throw new RangeError(`upperGamma: x must not be negative, received ${x}`);
  return x < a + 1 ? 1 - lowerSeries(a, x) : upperFraction(a, x);
}

/**
 * The probability that a chi-square variate on `degreesOfFreedom` exceeds
 * `statistic`.
 *
 * A negative statistic is not tolerated. It cannot arise from a quadratic form,
 * so if one appears the caller has a sign error upstream and returning 1 would
 * hide it.
 */
export function chiSquareUpperTail(statistic: number, degreesOfFreedom: number): number {
  requireFinite(statistic, 'chiSquareUpperTail: statistic');
  if (statistic < 0) {
    throw new RangeError(`chiSquareUpperTail: statistic must not be negative, received ${statistic}`);
  }
  if (!Number.isInteger(degreesOfFreedom) || degreesOfFreedom < 1) {
    throw new RangeError(
      `chiSquareUpperTail: degrees of freedom must be a positive integer, received ${degreesOfFreedom}`,
    );
  }
  return upperGamma(degreesOfFreedom / 2, statistic / 2);
}

/**
 * The standard normal upper tail, `P(Z > z)`.
 *
 * Expressed through the incomplete gamma function at `a = 1/2`, which is what
 * the complementary error function is. Below zero it reflects rather than
 * evaluating, since the series are only defined for a non-negative argument.
 */
export function normalUpperTail(z: number): number {
  requireFinite(z, 'normalUpperTail: z');
  if (z < 0) return 1 - normalUpperTail(-z);
  return 0.5 * upperGamma(0.5, (z * z) / 2);
}

/** The two-sided standard normal tail, `P(|Z| > |z|)`. */
export function normalTwoSidedTail(z: number): number {
  return 2 * normalUpperTail(Math.abs(z));
}
