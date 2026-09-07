import { bracketSignChange, safeguardedRoot } from '../core/root.js';
import {
  probabilityCorrect,
  responseDerivative,
  responseSecondDerivative,
  standardError as standardErrorOf,
  type ScoredResponse,
} from '../models/response.js';
import { expectedInformation, scoreFunction } from './likelihood.js';
import type { AbilityEstimate } from './mle.js';

/**
 * Warm's bias-correction term, `J(theta) = sum_i P'_i P''_i / (P_i Q_i)`.
 *
 * This is the quantity that measures how asymmetric the likelihood is at a given
 * ability. Maximum likelihood is biased outward — estimates in the tails are too
 * extreme — because the likelihood of an extreme pattern is skewed, and `J`
 * quantifies exactly that skew.
 */
export function warmCorrection(responses: readonly ScoredResponse[], theta: number): number {
  let total = 0;
  for (const { item } of responses) {
    const p = probabilityCorrect(item.parameters, theta);
    const q = 1 - p;
    const denominator = p * q;
    if (denominator <= Number.MIN_VALUE) continue;
    total +=
      (responseDerivative(item.parameters, theta) *
        responseSecondDerivative(item.parameters, theta)) /
      denominator;
  }
  return total;
}

/**
 * The weighted score: the likelihood's score function plus `J / (2 I)`.
 *
 * Its root is Warm's weighted likelihood estimate.
 */
export function weightedScore(responses: readonly ScoredResponse[], theta: number): number {
  const information = expectedInformation(responses, theta);
  if (information <= Number.MIN_VALUE) return scoreFunction(responses, theta);
  return scoreFunction(responses, theta) + warmCorrection(responses, theta) / (2 * information);
}

export interface WleOptions {
  /** Lower edge of the search window. Default -6. */
  readonly min?: number;
  /** Upper edge of the search window. Default 6. */
  readonly max?: number;
  /** Absolute tolerance on the estimate. Default 1e-8. */
  readonly tolerance?: number;
  /** Iteration budget for the root finder. Default 100. */
  readonly maxIterations?: number;
  /** Grid resolution for the bracketing scan. Default 96. */
  readonly scanSteps?: number;
}

/**
 * Warm's weighted likelihood estimate (WLE) of ability.
 *
 * Multiplying the likelihood by a weight function whose derivative is
 * `J / (2 I)` removes the first-order bias of maximum likelihood. Two properties
 * follow, and both matter for adaptive testing:
 *
 * 1. Estimates in the tails are pulled in towards where the evidence actually
 *    supports, without borrowing the shrinkage of a prior. WLE is a frequentist
 *    estimator; it does not assume the candidate came from a normal population.
 * 2. It is finite for all-correct and all-incorrect patterns, where maximum
 *    likelihood has no answer at all. For a single Rasch item answered
 *    correctly, the weighted score reduces to `1.5 - 2 P(theta)`, whose root is
 *    `b + ln 3` — a real estimate rather than an infinity.
 *
 * That makes WLE the natural default for reporting a score at the end of a test,
 * where a prior-based shrinkage towards the population mean is hard to defend to
 * the candidate whose score it lowered.
 *
 * The derivative of the weighted score has no compact closed form, so the solver
 * is given a central finite difference of it. The step is bracketed, so an
 * imprecise derivative costs a few extra iterations rather than correctness.
 */
export function estimateWle(
  responses: readonly ScoredResponse[],
  options: WleOptions = {},
): AbilityEstimate {
  if (responses.length === 0) {
    throw new RangeError('estimateWle: at least one response is required');
  }
  const min = options.min ?? -6;
  const max = options.max ?? 6;
  const tolerance = options.tolerance ?? 1e-8;
  const maxIterations = options.maxIterations ?? 100;
  const scanSteps = options.scanSteps ?? 96;
  if (min >= max) {
    throw new RangeError(`estimateWle: empty search window [${min}, ${max}]`);
  }

  const f = (theta: number): number => weightedScore(responses, theta);
  const h = 1e-5;
  const df = (theta: number): number => (f(theta + h) - f(theta - h)) / (2 * h);

  const bracket = bracketSignChange(f, min, max, scanSteps);
  if (bracket === null) {
    const preferHigh = f((min + max) / 2) > 0;
    const theta = preferHigh ? max : min;
    return {
      theta,
      standardError: standardErrorOf(expectedInformation(responses, theta)),
      method: 'wle',
      converged: false,
      boundary: preferHigh ? 'upper' : 'lower',
      iterations: 0,
    };
  }

  const [lo, hi] = bracket;
  const result =
    lo === hi
      ? { root: lo, converged: true, iterations: 0 }
      : safeguardedRoot(f, df, lo, hi, { tolerance, maxIterations });

  return {
    theta: result.root,
    standardError: standardErrorOf(expectedInformation(responses, result.root)),
    method: 'wle',
    converged: result.converged,
    boundary: 'none',
    iterations: result.iterations,
  };
}
