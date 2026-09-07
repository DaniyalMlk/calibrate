import { bracketSignChange, safeguardedRoot } from '../core/root.js';
import { standardError as standardErrorOf } from '../models/response.js';
import type { ScoredResponse } from '../models/response.js';
import {
  expectedInformation,
  logLikelihood,
  observedInformation,
  patternBoundedness,
  scoreFunction,
} from './likelihood.js';

export type EstimatorName = 'mle' | 'eap' | 'map' | 'wle';

/**
 * Where an ability estimate sits relative to the search window.
 *
 * `none` means an interior maximum was located. `upper` and `lower` mean the
 * likelihood has no finite maximiser in that direction and the reported ability
 * is the edge of the window, not an estimate.
 */
export type BoundaryFlag = 'none' | 'upper' | 'lower';

export interface AbilityEstimate {
  /** The ability estimate on the logit scale. */
  readonly theta: number;
  /** Standard error at the estimate, from the Fisher information of the items administered. */
  readonly standardError: number;
  /** Which estimator produced this. */
  readonly method: EstimatorName;
  /** True only when an interior optimum was located to tolerance. */
  readonly converged: boolean;
  /** Whether the estimate is pinned to the edge of the window. */
  readonly boundary: BoundaryFlag;
  /** Iterations performed by the underlying solver. */
  readonly iterations: number;
}

export interface MleOptions {
  /** Lower edge of the search window. Default -5. */
  readonly min?: number;
  /** Upper edge of the search window. Default 5. */
  readonly max?: number;
  /** Absolute tolerance on the ability estimate. Default 1e-8. */
  readonly tolerance?: number;
  /** Iteration budget for the root finder. Default 100. */
  readonly maxIterations?: number;
  /**
   * Grid resolution for the initial bracketing scan. Default 64.
   *
   * The 3PL log-likelihood is not guaranteed to be unimodal, so the estimator
   * scans for a sign change in the score function rather than assuming one.
   */
  readonly scanSteps?: number;
}

/**
 * Maximum likelihood estimate of ability from a scored response pattern.
 *
 * The estimate is the interior stationary point of the log-likelihood, located
 * as a root of the score function. Newton's method uses the exact derivative of
 * the score — the negative observed information — but is confined to a bracket
 * found by a grid scan, so a locally negative curvature under the 3PL cannot
 * throw it out of the window.
 *
 * All-correct and all-incorrect patterns have no finite maximiser. They are
 * reported with `boundary` set and `converged: false`; the `theta` field carries
 * the edge of the window so callers who want a usable number have one, but the
 * flag is what should be trusted.
 */
export function estimateMle(
  responses: readonly ScoredResponse[],
  options: MleOptions = {},
): AbilityEstimate {
  const min = options.min ?? -5;
  const max = options.max ?? 5;
  const tolerance = options.tolerance ?? 1e-8;
  const maxIterations = options.maxIterations ?? 100;
  const scanSteps = options.scanSteps ?? 64;

  if (min >= max) {
    throw new RangeError(`estimateMle: empty search window [${min}, ${max}]`);
  }

  const boundedness = patternBoundedness(responses);
  if (boundedness === 'empty') {
    throw new RangeError('estimateMle: at least one response is required');
  }
  if (boundedness === 'unbounded-high') {
    return atBoundary(responses, max, 'upper');
  }
  if (boundedness === 'unbounded-low') {
    return atBoundary(responses, min, 'lower');
  }

  const score = (theta: number): number => scoreFunction(responses, theta);
  // d/dtheta of the score function is the second derivative of the
  // log-likelihood, which is the negative of the observed information.
  const scoreDerivative = (theta: number): number => -observedInformation(responses, theta);

  const bracket = bracketSignChange(score, min, max, scanSteps);
  if (bracket === null) {
    // The likelihood is monotone across the whole window: the maximiser lies
    // outside it. Report the better endpoint, flagged.
    const preferHigh = logLikelihood(responses, max) >= logLikelihood(responses, min);
    return atBoundary(responses, preferHigh ? max : min, preferHigh ? 'upper' : 'lower');
  }

  const [lo, hi] = bracket;
  if (lo === hi) {
    return {
      theta: lo,
      standardError: standardErrorOf(expectedInformation(responses, lo)),
      method: 'mle',
      converged: true,
      boundary: 'none',
      iterations: 0,
    };
  }

  const result = safeguardedRoot(score, scoreDerivative, lo, hi, { tolerance, maxIterations });
  return {
    theta: result.root,
    standardError: standardErrorOf(expectedInformation(responses, result.root)),
    method: 'mle',
    converged: result.converged,
    boundary: 'none',
    iterations: result.iterations,
  };
}

function atBoundary(
  responses: readonly ScoredResponse[],
  theta: number,
  boundary: Exclude<BoundaryFlag, 'none'>,
): AbilityEstimate {
  return {
    theta,
    standardError: standardErrorOf(expectedInformation(responses, theta)),
    method: 'mle',
    converged: false,
    boundary,
    iterations: 0,
  };
}
