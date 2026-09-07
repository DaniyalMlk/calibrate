import { LOGISTIC_METRIC_SCALE, NORMAL_METRIC_SCALE, requireFinite } from '../core/numeric.js';

/**
 * The scale on which an item's exponent is expressed.
 *
 * Item parameters are only meaningful together with the metric they were
 * calibrated on: the same `a` means a steeper curve under `normal` than under
 * `logistic`. Carrying the metric on the item makes a mismatched bank a type
 * error rather than a silently wrong score.
 */
export type Metric = 'logistic' | 'normal';

export function metricScale(metric: Metric): number {
  return metric === 'normal' ? NORMAL_METRIC_SCALE : LOGISTIC_METRIC_SCALE;
}

/**
 * Item parameters in the four-parameter logistic parameterisation.
 *
 * Every supported dichotomous model is a special case of the 4PL, so the engine
 * carries one shape internally and lets the constructors below pin the
 * parameters a given model holds fixed:
 *
 * - Rasch / 1PL: `a` fixed across the bank, `c = 0`, `d = 1`
 * - 2PL: free `a`, `c = 0`, `d = 1`
 * - 3PL: free `a`, free `c`, `d = 1`
 * - 4PL: all free
 */
export interface ItemParameters {
  /** Discrimination: the slope of the curve at its inflection point. */
  readonly a: number;
  /** Difficulty: the ability at which the curve reaches its midpoint. */
  readonly b: number;
  /** Lower asymptote — the probability a candidate of minimal ability answers correctly. */
  readonly c: number;
  /** Upper asymptote — one minus the probability a maximal-ability candidate slips. */
  readonly d: number;
  /** The metric `a` and `b` were calibrated on. */
  readonly metric: Metric;
}

/** An item in a bank: identity and content metadata alongside its parameters. */
export interface Item {
  readonly id: string;
  readonly parameters: ItemParameters;
  /** Optional blueprint category, used by content-balanced selection. */
  readonly domain?: string;
  /** Free-form labels, e.g. skill tags. */
  readonly tags?: readonly string[];
}

/** The largest discrimination the engine will accept. */
const MAX_DISCRIMINATION = 20;
/** The largest absolute difficulty the engine will accept. */
const MAX_DIFFICULTY = 20;

/**
 * Validate a full parameter set, returning a frozen copy.
 *
 * The bounds are not arbitrary: a discrimination above ~20 makes the response
 * function a step function to within float precision, at which point the
 * likelihood is flat almost everywhere and every estimator degenerates. Failing
 * loudly at construction is much cheaper to debug than a session that silently
 * stops converging.
 */
export function validateItemParameters(parameters: ItemParameters): ItemParameters {
  const { a, b, c, d, metric } = parameters;

  requireFinite(a, 'discrimination (a)');
  requireFinite(b, 'difficulty (b)');
  requireFinite(c, 'lower asymptote (c)');
  requireFinite(d, 'upper asymptote (d)');

  if (metric !== 'logistic' && metric !== 'normal') {
    throw new RangeError(`metric must be "logistic" or "normal", received ${String(metric)}`);
  }
  if (a <= 0) {
    throw new RangeError(`discrimination (a) must be positive, received ${a}`);
  }
  if (a > MAX_DISCRIMINATION) {
    throw new RangeError(
      `discrimination (a) must be at most ${MAX_DISCRIMINATION}, received ${a}; ` +
        'above this the response function is numerically a step function',
    );
  }
  if (Math.abs(b) > MAX_DIFFICULTY) {
    throw new RangeError(
      `difficulty (b) must lie within +/-${MAX_DIFFICULTY}, received ${b}`,
    );
  }
  if (c < 0 || c >= 1) {
    throw new RangeError(`lower asymptote (c) must lie in [0, 1), received ${c}`);
  }
  if (d <= 0 || d > 1) {
    throw new RangeError(`upper asymptote (d) must lie in (0, 1], received ${d}`);
  }
  if (c >= d) {
    throw new RangeError(
      `lower asymptote (c = ${c}) must be strictly below upper asymptote (d = ${d})`,
    );
  }

  return Object.freeze({ a, b, c, d, metric });
}

/** A Rasch item: fixed unit discrimination, no guessing, no slipping. */
export function rasch(b: number, metric: Metric = 'logistic'): ItemParameters {
  return validateItemParameters({ a: 1, b, c: 0, d: 1, metric });
}

/** A one-parameter item with a discrimination shared across the bank. */
export function onePL(b: number, sharedA = 1, metric: Metric = 'logistic'): ItemParameters {
  return validateItemParameters({ a: sharedA, b, c: 0, d: 1, metric });
}

/** A two-parameter logistic item. */
export function twoPL(a: number, b: number, metric: Metric = 'logistic'): ItemParameters {
  return validateItemParameters({ a, b, c: 0, d: 1, metric });
}

/** A three-parameter logistic item, with a lower asymptote for guessing. */
export function threePL(
  a: number,
  b: number,
  c: number,
  metric: Metric = 'logistic',
): ItemParameters {
  return validateItemParameters({ a, b, c, d: 1, metric });
}

/** A four-parameter logistic item, with both guessing and slipping. */
export function fourPL(
  a: number,
  b: number,
  c: number,
  d: number,
  metric: Metric = 'logistic',
): ItemParameters {
  return validateItemParameters({ a, b, c, d, metric });
}

/** Build a validated item record. */
export function makeItem(
  id: string,
  parameters: ItemParameters,
  meta: { domain?: string; tags?: readonly string[] } = {},
): Item {
  if (typeof id !== 'string' || id.length === 0) {
    throw new RangeError('item id must be a non-empty string');
  }
  const item: Item = {
    id,
    parameters: validateItemParameters(parameters),
    ...(meta.domain === undefined ? {} : { domain: meta.domain }),
    ...(meta.tags === undefined ? {} : { tags: Object.freeze([...meta.tags]) }),
  };
  return Object.freeze(item);
}

/**
 * Name the most constrained model consistent with a parameter set.
 *
 * Useful for reporting: a bank described as "3PL" that never has `c > 0` is
 * really a 2PL bank, and knowing that changes which estimator is appropriate.
 */
export function describeModel(parameters: ItemParameters): '1PL' | '2PL' | '3PL' | '4PL' {
  const { a, c, d } = parameters;
  if (d < 1) return '4PL';
  if (c > 0) return '3PL';
  if (a === 1) return '1PL';
  return '2PL';
}
