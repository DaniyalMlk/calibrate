import { logistic, requireFinite } from '../core/numeric.js';
import { metricScale, type Metric } from './item.js';

/**
 * The polytomous response models the engine supports.
 *
 * - `graded` — Samejima's graded response model. Built from *cumulative*
 *   boundary curves: each threshold answers "did the response reach at least
 *   this category?", and a category probability is the gap between two adjacent
 *   boundaries. The natural fit for rubrics whose levels are ordered bands of a
 *   single underlying quality ("no credit / partial / full").
 * - `partial-credit` — Masters' partial credit model. Built from *adjacent*
 *   step comparisons: each threshold governs the odds of clearing one step
 *   given that you reached the step below it. The natural fit for a task solved
 *   in sequence, where each step is its own hurdle.
 * - `generalized-partial-credit` — Muraki's generalisation of the above, with a
 *   free discrimination per item rather than one shared across the bank.
 *
 * The distinction between the graded family and the partial credit family is
 * not cosmetic: it changes what a threshold *means*, and therefore whether
 * thresholds must be ordered. See `validatePolytomousParameters`.
 */
export type PolytomousModel = 'graded' | 'partial-credit' | 'generalized-partial-credit';

/**
 * Parameters of a polytomous item.
 *
 * An item with `thresholds.length === m` scores into `m + 1` ordered categories
 * numbered `0..m`. Category 0 is always the lowest — no credit — and needs no
 * threshold of its own, which is why there is one fewer threshold than
 * category.
 */
export interface PolytomousParameters {
  /**
   * Discrimination. Fixed at 1 for `partial-credit`, free for the other two.
   */
  readonly a: number;
  /**
   * Category thresholds, `m` of them for `m + 1` categories.
   *
   * Under `graded` these are boundary locations: `thresholds[k - 1]` is the
   * ability at which reaching category `k` or higher becomes more likely than
   * not. Under the partial credit models they are step difficulties: the
   * ability at which categories `k - 1` and `k` are equally likely.
   */
  readonly thresholds: readonly number[];
  readonly model: PolytomousModel;
  /** The metric the parameters were calibrated on. */
  readonly metric: Metric;
}

/** The largest discrimination accepted, matching the dichotomous bound. */
const MAX_DISCRIMINATION = 20;
/** The largest absolute threshold accepted, matching the dichotomous bound. */
const MAX_THRESHOLD = 20;
/** The most categories a single item may score into. */
const MAX_CATEGORIES = 64;

/**
 * Validate a polytomous parameter set, returning a frozen copy.
 *
 * The ordering rule is the part worth reading. Under the graded model a
 * threshold is a cumulative boundary, so `P(X >= k)` must fall as `k` rises;
 * unordered thresholds would make some category probability negative, which is
 * not a modelling choice but an arithmetic impossibility. It is rejected.
 *
 * Under the partial credit models a threshold is an adjacent step, and
 * *reversals are meaningful*: a step difficulty below its predecessor says that
 * the category is never the single most likely outcome at any ability — a
 * narrow middle band of a rubric, or a category respondents skip past. The
 * probabilities stay positive and sum to one throughout. Rejecting reversals
 * here would throw away a real and diagnostically useful result, so they are
 * allowed.
 */
export function validatePolytomousParameters(
  parameters: PolytomousParameters,
): PolytomousParameters {
  const { a, thresholds, model, metric } = parameters;

  if (
    model !== 'graded' &&
    model !== 'partial-credit' &&
    model !== 'generalized-partial-credit'
  ) {
    throw new RangeError(
      `model must be "graded", "partial-credit" or "generalized-partial-credit", ` +
        `received ${String(model)}`,
    );
  }
  if (metric !== 'logistic' && metric !== 'normal') {
    throw new RangeError(`metric must be "logistic" or "normal", received ${String(metric)}`);
  }

  requireFinite(a, 'discrimination (a)');
  if (a <= 0) {
    throw new RangeError(`discrimination (a) must be positive, received ${a}`);
  }
  if (a > MAX_DISCRIMINATION) {
    throw new RangeError(
      `discrimination (a) must be at most ${MAX_DISCRIMINATION}, received ${a}; ` +
        'above this the category curves are numerically step functions',
    );
  }
  if (model === 'partial-credit' && a !== 1) {
    throw new RangeError(
      `the partial credit model fixes discrimination at 1, received ${a}; ` +
        'use "generalized-partial-credit" for a free discrimination',
    );
  }

  if (!Array.isArray(thresholds) && !ArrayBuffer.isView(thresholds)) {
    throw new RangeError('thresholds must be an array');
  }
  if (thresholds.length === 0) {
    throw new RangeError('at least one threshold is required (two or more categories)');
  }
  if (thresholds.length + 1 > MAX_CATEGORIES) {
    throw new RangeError(
      `an item may score into at most ${MAX_CATEGORIES} categories, ` +
        `received ${thresholds.length + 1}`,
    );
  }

  for (let k = 0; k < thresholds.length; k += 1) {
    const value = thresholds[k] as number;
    requireFinite(value, `threshold ${k + 1}`);
    if (Math.abs(value) > MAX_THRESHOLD) {
      throw new RangeError(
        `threshold ${k + 1} must lie within +/-${MAX_THRESHOLD}, received ${value}`,
      );
    }
  }

  if (model === 'graded') {
    for (let k = 1; k < thresholds.length; k += 1) {
      const previous = thresholds[k - 1] as number;
      const current = thresholds[k] as number;
      if (current <= previous) {
        throw new RangeError(
          `graded response thresholds must be strictly increasing, but threshold ` +
            `${k + 1} (${current}) is not above threshold ${k} (${previous}); ` +
            'unordered boundaries imply a negative category probability',
        );
      }
    }
  }

  return Object.freeze({
    a,
    thresholds: Object.freeze([...thresholds]),
    model,
    metric,
  });
}

/** A graded response item: ordered cumulative boundaries and a free slope. */
export function graded(
  a: number,
  thresholds: readonly number[],
  metric: Metric = 'logistic',
): PolytomousParameters {
  return validatePolytomousParameters({ a, thresholds, model: 'graded', metric });
}

/** A partial credit item: adjacent step difficulties, discrimination fixed at 1. */
export function partialCredit(
  thresholds: readonly number[],
  metric: Metric = 'logistic',
): PolytomousParameters {
  return validatePolytomousParameters({
    a: 1,
    thresholds,
    model: 'partial-credit',
    metric,
  });
}

/** A generalized partial credit item: adjacent steps with a free discrimination. */
export function generalizedPartialCredit(
  a: number,
  thresholds: readonly number[],
  metric: Metric = 'logistic',
): PolytomousParameters {
  return validatePolytomousParameters({
    a,
    thresholds,
    model: 'generalized-partial-credit',
    metric,
  });
}

/** Number of ordered categories the item scores into: one more than its thresholds. */
export function categoryCount(parameters: PolytomousParameters): number {
  return parameters.thresholds.length + 1;
}

/** The highest category score attainable — equivalently, the number of thresholds. */
export function maximumScore(parameters: PolytomousParameters): number {
  return parameters.thresholds.length;
}

/**
 * The scaled exponent for boundary `k`, one-based: `D * a * (theta - b_k)`.
 */
function boundaryExponent(
  parameters: PolytomousParameters,
  theta: number,
  k: number,
): number {
  const scale = metricScale(parameters.metric) * parameters.a;
  return scale * (theta - (parameters.thresholds[k - 1] as number));
}

/**
 * Cumulative boundary probability `P*_k(theta) = P(X >= k)` for the graded model.
 *
 * By definition `P*_0 = 1` (every response is at least category 0) and
 * `P*_{m + 1} = 0` (no response exceeds the top category). Those two conventions
 * are what make the adjacent differences below a proper distribution without a
 * separate normalising step.
 */
export function boundaryProbability(
  parameters: PolytomousParameters,
  theta: number,
  k: number,
): number {
  requireFinite(theta, 'ability (theta)');
  if (parameters.model !== 'graded') {
    throw new RangeError('boundary probabilities are defined for the graded response model only');
  }
  if (!Number.isInteger(k) || k < 0 || k > categoryCount(parameters)) {
    throw new RangeError(
      `boundary index must be an integer in [0, ${categoryCount(parameters)}], received ${k}`,
    );
  }
  if (k === 0) return 1;
  if (k === categoryCount(parameters)) return 0;
  return logistic(boundaryExponent(parameters, theta, k));
}

/**
 * Difference of two logistic values, `sigma(u) - sigma(v)` with `u >= v`,
 * without cancellation.
 *
 * The graded model wants adjacent boundary probabilities subtracted, and far
 * out in either tail both of them sit against the same asymptote: at
 * `theta = 15` a five-category item has every boundary at `1 - 1e-12`, and the
 * naive difference of two doubles that close together keeps almost none of the
 * significant digits it should. The identity
 *
 * ```
 * sigma(u) - sigma(v) = sigma(u) * sigma(-v) * (1 - exp(v - u))
 * ```
 *
 * moves the subtraction inside `expm1`, where the argument is non-positive and
 * the result is accurate to full precision. Each remaining factor is evaluated
 * on the side of its own asymptote where it is small, so the product underflows
 * gracefully to zero instead of rounding to it prematurely.
 */
function logisticDifference(u: number, v: number): number {
  if (u <= v) return 0;
  // -expm1(v - u) === 1 - exp(v - u), evaluated where it is well conditioned.
  return logistic(u) * logistic(-v) * -Math.expm1(v - u);
}

/**
 * Category probabilities under the graded response model.
 *
 * `P_k = P*_k - P*_{k + 1}`, with the boundary conventions above.
 */
function gradedProbabilities(parameters: PolytomousParameters, theta: number): number[] {
  const m = parameters.thresholds.length;
  const exponents: number[] = new Array<number>(m);
  for (let k = 1; k <= m; k += 1) {
    exponents[k - 1] = boundaryExponent(parameters, theta, k);
  }

  const probabilities: number[] = new Array<number>(m + 1);
  // P_0 = 1 - P*_1 = sigma(-x_1), exact with no subtraction at all.
  probabilities[0] = logistic(-(exponents[0] as number));
  for (let k = 1; k < m; k += 1) {
    probabilities[k] = logisticDifference(exponents[k - 1] as number, exponents[k] as number);
  }
  // P_m = P*_m - 0, likewise exact.
  probabilities[m] = logistic(exponents[m - 1] as number);
  return probabilities;
}

/**
 * Category probabilities under the (generalized) partial credit model.
 *
 * `P_k = exp(z_k) / sum_c exp(z_c)`, where `z_k = D a (k theta - sum_{v<=k} b_v)`
 * accumulates the step exponents up to category `k` and `z_0 = 0`.
 *
 * The exponents are shifted by their maximum before exponentiating. This is the
 * standard softmax stabilisation and it is load-bearing here rather than
 * decorative: `z_k` grows linearly in both `k` and `theta`, so a six-category
 * item evaluated at `theta = 12` reaches `exp(70)` — still finite, but a
 * twelve-category item at the same ability overflows to `Infinity` and turns
 * every probability into `NaN`. Subtracting the maximum makes the largest term
 * exactly 1 and leaves the ratios untouched.
 */
function partialCreditProbabilities(
  parameters: PolytomousParameters,
  theta: number,
): number[] {
  const m = parameters.thresholds.length;
  const scale = metricScale(parameters.metric) * parameters.a;

  const exponents: number[] = new Array<number>(m + 1);
  exponents[0] = 0;
  let running = 0;
  let largest = 0;
  for (let k = 1; k <= m; k += 1) {
    running += scale * (theta - (parameters.thresholds[k - 1] as number));
    exponents[k] = running;
    if (running > largest) largest = running;
  }

  let total = 0;
  const probabilities: number[] = new Array<number>(m + 1);
  for (let k = 0; k <= m; k += 1) {
    const value = Math.exp((exponents[k] as number) - largest);
    probabilities[k] = value;
    total += value;
  }
  for (let k = 0; k <= m; k += 1) {
    probabilities[k] = (probabilities[k] as number) / total;
  }
  return probabilities;
}

/**
 * The full distribution over categories at a given ability.
 *
 * Returns an array of length `categoryCount(parameters)` that sums to one.
 */
export function categoryProbabilities(
  parameters: PolytomousParameters,
  theta: number,
): number[] {
  requireFinite(theta, 'ability (theta)');
  return parameters.model === 'graded'
    ? gradedProbabilities(parameters, theta)
    : partialCreditProbabilities(parameters, theta);
}

/** Probability of scoring in exactly category `k`. */
export function categoryProbability(
  parameters: PolytomousParameters,
  theta: number,
  k: number,
): number {
  const count = categoryCount(parameters);
  if (!Number.isInteger(k) || k < 0 || k >= count) {
    throw new RangeError(
      `category must be an integer in [0, ${count - 1}], received ${k}`,
    );
  }
  return categoryProbabilities(parameters, theta)[k] as number;
}

/**
 * First derivatives of every category probability with respect to ability.
 *
 * Both families have clean analytic forms and neither needs finite differences:
 *
 * - graded: `dP_k/dtheta = D a [P*_k (1 - P*_k) - P*_{k+1} (1 - P*_{k+1})]`
 * - partial credit: `dP_k/dtheta = D a P_k (k - E[X])`
 *
 * The partial credit form is the score identity of an exponential family — each
 * category is pushed in proportion to how far its own score sits from the
 * item's expected score — and it is why the information function below collapses
 * to a variance.
 */
export function categoryDerivatives(
  parameters: PolytomousParameters,
  theta: number,
): number[] {
  requireFinite(theta, 'ability (theta)');
  const m = parameters.thresholds.length;
  const scale = metricScale(parameters.metric) * parameters.a;

  if (parameters.model === 'graded') {
    const slopes: number[] = new Array<number>(m + 2);
    // The fixed boundaries P*_0 = 1 and P*_{m+1} = 0 are constants in theta.
    slopes[0] = 0;
    slopes[m + 1] = 0;
    for (let k = 1; k <= m; k += 1) {
      const star = logistic(boundaryExponent(parameters, theta, k));
      slopes[k] = scale * star * (1 - star);
    }
    const derivatives: number[] = new Array<number>(m + 1);
    for (let k = 0; k <= m; k += 1) {
      derivatives[k] = (slopes[k] as number) - (slopes[k + 1] as number);
    }
    return derivatives;
  }

  const probabilities = partialCreditProbabilities(parameters, theta);
  let expected = 0;
  for (let k = 0; k <= m; k += 1) expected += k * (probabilities[k] as number);

  const derivatives: number[] = new Array<number>(m + 1);
  for (let k = 0; k <= m; k += 1) {
    derivatives[k] = scale * (probabilities[k] as number) * (k - expected);
  }
  return derivatives;
}

/**
 * Second derivatives of every category probability with respect to ability.
 *
 * Needed by the observed information of a response pattern, which depends on the
 * categories actually scored rather than on their expectation.
 *
 * - graded: `d2P_k = d2P*_k - d2P*_{k+1}` with
 *   `d2P*_k = (D a)^2 P*_k (1 - P*_k) (1 - 2 P*_k)`
 * - partial credit: `d2P_k = (D a)^2 P_k [(k - E[X])^2 - Var(X)]`
 */
export function categorySecondDerivatives(
  parameters: PolytomousParameters,
  theta: number,
): number[] {
  requireFinite(theta, 'ability (theta)');
  const m = parameters.thresholds.length;
  const scale = metricScale(parameters.metric) * parameters.a;
  const scaleSquared = scale * scale;

  if (parameters.model === 'graded') {
    const curvatures: number[] = new Array<number>(m + 2);
    curvatures[0] = 0;
    curvatures[m + 1] = 0;
    for (let k = 1; k <= m; k += 1) {
      const star = logistic(boundaryExponent(parameters, theta, k));
      curvatures[k] = scaleSquared * star * (1 - star) * (1 - 2 * star);
    }
    const second: number[] = new Array<number>(m + 1);
    for (let k = 0; k <= m; k += 1) {
      second[k] = (curvatures[k] as number) - (curvatures[k + 1] as number);
    }
    return second;
  }

  const probabilities = partialCreditProbabilities(parameters, theta);
  let expected = 0;
  for (let k = 0; k <= m; k += 1) expected += k * (probabilities[k] as number);
  let spread = 0;
  for (let k = 0; k <= m; k += 1) {
    const deviation = k - expected;
    spread += (probabilities[k] as number) * deviation * deviation;
  }

  const second: number[] = new Array<number>(m + 1);
  for (let k = 0; k <= m; k += 1) {
    const deviation = k - expected;
    second[k] = scaleSquared * (probabilities[k] as number) * (deviation * deviation - spread);
  }
  return second;
}

/**
 * Expected category score at a given ability, `E[X | theta] = sum_k k P_k`.
 *
 * The polytomous analogue of the dichotomous response function: it rises
 * monotonically from 0 to the maximum score and is the term a test
 * characteristic curve sums over when a form mixes formats.
 */
export function expectedCategoryScore(
  parameters: PolytomousParameters,
  theta: number,
): number {
  const probabilities = categoryProbabilities(parameters, theta);
  let total = 0;
  for (let k = 0; k < probabilities.length; k += 1) {
    total += k * (probabilities[k] as number);
  }
  return total;
}

/**
 * Variance of the category score at a given ability, `Var[X | theta]`.
 *
 * Reported in its own right because it *is* the partial credit information
 * function up to the squared slope, and because a category variance near zero
 * is the signature of an item that has stopped discriminating.
 */
export function categoryScoreVariance(
  parameters: PolytomousParameters,
  theta: number,
): number {
  const probabilities = categoryProbabilities(parameters, theta);
  let expected = 0;
  for (let k = 0; k < probabilities.length; k += 1) {
    expected += k * (probabilities[k] as number);
  }
  let spread = 0;
  for (let k = 0; k < probabilities.length; k += 1) {
    const deviation = k - expected;
    spread += (probabilities[k] as number) * deviation * deviation;
  }
  return spread;
}

/**
 * Fisher information contributed by a polytomous item at a given ability.
 *
 * `I(theta) = sum_k [P'_k(theta)]^2 / P_k(theta)`, the general form for any
 * categorical response model — the dichotomous `(P')^2 / (P Q)` is the
 * two-category case of exactly this sum.
 *
 * The partial credit models admit the closed form `D^2 a^2 Var(X | theta)`,
 * which `partialCreditInformation` evaluates directly. The general sum is used
 * here for both families so that the closed form has something independent to
 * be checked against, and the test suite does check it.
 *
 * Categories whose probability has underflowed are skipped rather than divided
 * by: their squared derivative underflows quadratically faster, so the limit of
 * each dropped term is zero.
 */
export function polytomousInformation(
  parameters: PolytomousParameters,
  theta: number,
): number {
  const probabilities = categoryProbabilities(parameters, theta);
  const derivatives = categoryDerivatives(parameters, theta);
  let total = 0;
  for (let k = 0; k < probabilities.length; k += 1) {
    const p = probabilities[k] as number;
    if (p <= Number.MIN_VALUE) continue;
    const d = derivatives[k] as number;
    total += (d * d) / p;
  }
  return total;
}

/**
 * Information of a partial credit item from its closed form,
 * `I(theta) = D^2 a^2 Var(X | theta)`.
 *
 * Substituting the derivative identity `P'_k = D a P_k (k - E[X])` into the
 * general sum leaves `D^2 a^2 sum_k P_k (k - E[X])^2`, which is the variance.
 * Throws for the graded model, which has no comparable collapse.
 */
export function partialCreditInformation(
  parameters: PolytomousParameters,
  theta: number,
): number {
  if (parameters.model === 'graded') {
    throw new RangeError(
      'the graded response model has no variance form for information; ' +
        'use polytomousInformation',
    );
  }
  const scale = metricScale(parameters.metric) * parameters.a;
  return scale * scale * categoryScoreVariance(parameters, theta);
}

/**
 * The ability at which a polytomous item is most informative.
 *
 * No closed form exists for either family once an item has more than two
 * categories, so this is a golden-section search over a window that brackets
 * every admissible peak: the information of these models is concentrated around
 * the thresholds, and padding the threshold range by six logits of slope on each
 * side leaves the maximum strictly inside.
 *
 * Information can be genuinely bimodal when thresholds are far apart — a rubric
 * with a wide dead band in the middle informs at each end and barely between —
 * so this reports *a* peak rather than pretending the surface is unimodal. The
 * search is still well defined and returns the peak of whichever mode it
 * brackets; `polytomousInformation` over a grid is the honest tool when the
 * whole shape matters.
 */
export function polytomousInformationPeak(parameters: PolytomousParameters): number {
  const scale = metricScale(parameters.metric) * parameters.a;
  const thresholds = parameters.thresholds;
  let lowest = thresholds[0] as number;
  let highest = thresholds[0] as number;
  for (const value of thresholds) {
    if (value < lowest) lowest = value;
    if (value > highest) highest = value;
  }
  return goldenSectionMaximum(
    (theta) => polytomousInformation(parameters, theta),
    lowest - 6 / scale,
    highest + 6 / scale,
  );
}

const INVERSE_PHI = (Math.sqrt(5) - 1) / 2;

/** Golden-section search for the maximiser of a function on [lo, hi]. */
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

/**
 * Name the model in the compact notation used in reports: `GRM`, `PCM` or
 * `GPCM`, suffixed with the number of categories.
 */
export function describePolytomousModel(parameters: PolytomousParameters): string {
  const label =
    parameters.model === 'graded'
      ? 'GRM'
      : parameters.model === 'partial-credit'
        ? 'PCM'
        : 'GPCM';
  return `${label}(${categoryCount(parameters)})`;
}
