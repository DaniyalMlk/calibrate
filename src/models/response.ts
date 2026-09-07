import { logistic, requireFinite } from '../core/numeric.js';
import { metricScale, type Item, type ItemParameters } from './item.js';

/** A scored dichotomous response: 1 correct, 0 incorrect. */
export type Response = 0 | 1;

/** An item paired with the response a candidate gave to it. */
export interface ScoredResponse {
  readonly item: Item;
  readonly response: Response;
}

/**
 * The logistic exponent, `D * a * (theta - b)`.
 *
 * Separated out because the score function, the information function and the
 * response function all need it, and because it is the single place the metric
 * scaling is applied.
 */
export function exponent(parameters: ItemParameters, theta: number): number {
  requireFinite(theta, 'ability (theta)');
  return metricScale(parameters.metric) * parameters.a * (theta - parameters.b);
}

/**
 * Probability of a correct response under the four-parameter logistic model,
 * `P(theta) = c + (d - c) / (1 + exp(-D a (theta - b)))`.
 *
 * The 1PL, 2PL and 3PL are the special cases described in `item.ts`, so this one
 * function serves all of them.
 */
export function probabilityCorrect(parameters: ItemParameters, theta: number): number {
  const { c, d } = parameters;
  return c + (d - c) * logistic(exponent(parameters, theta));
}

/** Probability of an incorrect response, `1 - P(theta)`. */
export function probabilityIncorrect(parameters: ItemParameters, theta: number): number {
  return 1 - probabilityCorrect(parameters, theta);
}

/** Probability of the response actually observed — `P` if correct, `1 - P` if not. */
export function probabilityOfResponse(
  parameters: ItemParameters,
  theta: number,
  response: Response,
): number {
  const p = probabilityCorrect(parameters, theta);
  return response === 1 ? p : 1 - p;
}

/**
 * First derivative of the response function with respect to ability.
 *
 * `P'(theta) = (d - c) * D * a * psi * (1 - psi)`, where `psi` is the logistic
 * kernel. Written in terms of `psi` rather than `P` so it stays exact when the
 * asymptotes are nonzero.
 */
export function responseDerivative(parameters: ItemParameters, theta: number): number {
  const { a, c, d } = parameters;
  const psi = logistic(exponent(parameters, theta));
  return (d - c) * metricScale(parameters.metric) * a * psi * (1 - psi);
}

/**
 * Fisher information contributed by a single item at a given ability.
 *
 * `I(theta) = [P'(theta)]^2 / (P(theta) * (1 - P(theta)))`.
 *
 * This general form is used rather than the model-specific shortcuts because it
 * is correct for all four models and avoids a branch per model. The 2PL
 * shortcut `D^2 a^2 P Q` and the Birnbaum 3PL form
 * `D^2 a^2 (Q/P) [(P - c)/(1 - c)]^2` both fall out of it algebraically, and the
 * test suite checks that they agree.
 */
export function itemInformation(parameters: ItemParameters, theta: number): number {
  const p = probabilityCorrect(parameters, theta);
  const q = 1 - p;
  // Far out in the tails `p * q` underflows to zero while `P'` underflows
  // faster; the limit of the ratio is zero, so report that rather than NaN.
  const denominator = p * q;
  if (denominator <= Number.MIN_VALUE) return 0;
  const derivative = responseDerivative(parameters, theta);
  return (derivative * derivative) / denominator;
}

/**
 * The ability at which an item is most informative.
 *
 * For any model with `d = 1` this has the closed form
 * `b + (1 / (D a)) * ln[(1 + sqrt(1 + 8c)) / 2]`, which reduces to `b` when
 * `c = 0`. The 4PL has no comparable closed form, so it falls back to a
 * golden-section search over a window six logits wide on each side of `b` —
 * wide enough to bracket the peak for any admissible parameter set.
 */
export function informationPeak(parameters: ItemParameters): number {
  const { a, b, c, d } = parameters;
  const scale = metricScale(parameters.metric);
  if (d === 1) {
    return b + Math.log((1 + Math.sqrt(1 + 8 * c)) / 2) / (scale * a);
  }
  return goldenSectionMaximum(
    (theta) => itemInformation(parameters, theta),
    b - 6 / (scale * a),
    b + 6 / (scale * a),
  );
}

const INVERSE_PHI = (Math.sqrt(5) - 1) / 2;

/** Golden-section search for the maximiser of a unimodal function on [lo, hi]. */
function goldenSectionMaximum(
  f: (x: number) => number,
  lo: number,
  hi: number,
  tolerance = 1e-10,
): number {
  let a = lo;
  let b = hi;
  let c = b - INVERSE_PHI * (b - a);
  let d = a + INVERSE_PHI * (b - a);
  let fc = f(c);
  let fd = f(d);
  while (b - a > tolerance) {
    if (fc > fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - INVERSE_PHI * (b - a);
      fc = f(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + INVERSE_PHI * (b - a);
      fd = f(d);
    }
  }
  return (a + b) / 2;
}

/** Sum of item information over a set of items — the test information function. */
export function testInformation(items: readonly Item[], theta: number): number {
  let total = 0;
  for (const item of items) total += itemInformation(item.parameters, theta);
  return total;
}

/**
 * Standard error of measurement implied by a test information value.
 *
 * `SEM(theta) = 1 / sqrt(I(theta))`. With no information the error is infinite,
 * which is the honest answer: before any item is administered the estimate is
 * whatever the prior says and the likelihood contributes nothing.
 */
export function standardError(information: number): number {
  if (information < 0) {
    throw new RangeError(`information must be non-negative, received ${information}`);
  }
  if (information === 0) return Number.POSITIVE_INFINITY;
  return 1 / Math.sqrt(information);
}

/**
 * Expected number-correct score on a set of items — the "true score" in classical
 * test theory, and the bridge between the ability metric and a raw score.
 */
export function expectedScore(items: readonly Item[], theta: number): number {
  let total = 0;
  for (const item of items) total += probabilityCorrect(item.parameters, theta);
  return total;
}
