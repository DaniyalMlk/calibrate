import {
  normalGaussHermiteRule,
  normalGridRule,
  type QuadratureRule,
} from '../core/quadrature.js';
import { bracketSignChange, safeguardedRoot } from '../core/root.js';
import type { ScoredResponse } from '../models/response.js';
import { expectedInformation, logLikelihood, observedInformation, scoreFunction } from './likelihood.js';
import type { AbilityEstimate } from './mle.js';
import {
  normalPrior,
  priorInformation,
  priorScore,
  STANDARD_NORMAL_PRIOR,
  type NormalPrior,
} from './prior.js';

/** Which quadrature family to build the posterior on. */
export type QuadratureKind = 'gauss-hermite' | 'grid';

export interface BayesOptions {
  /** Prior over ability. Defaults to standard normal. */
  readonly prior?: NormalPrior;
  /** Quadrature family. Defaults to Gauss–Hermite. */
  readonly quadrature?: QuadratureKind;
  /** Number of quadrature points. Defaults to 41. */
  readonly points?: number;
}

/** An ability estimate together with the posterior it was computed from. */
export interface PosteriorEstimate extends AbilityEstimate {
  /** Posterior standard deviation — the Bayesian counterpart of the standard error. */
  readonly posteriorSd: number;
}

function ruleFor(prior: NormalPrior, options: BayesOptions): QuadratureRule {
  const points = options.points ?? 41;
  return options.quadrature === 'grid'
    ? normalGridRule(prior.mean, prior.sd, points)
    : normalGaussHermiteRule(prior.mean, prior.sd, points);
}

/**
 * Expected a posteriori (EAP) ability estimate — the mean of the posterior.
 *
 * The posterior is evaluated on a quadrature rule that already represents the
 * prior, so the weighted sum needs only the likelihood at each node:
 *
 * ```
 * theta_hat = sum_k theta_k L(theta_k) W_k / sum_k L(theta_k) W_k
 * ```
 *
 * The likelihood is computed in logs and the maximum is subtracted before
 * exponentiating. Without that shift a 40-item pattern produces likelihoods
 * around `1e-20` at every node, and after weighting the numerator and
 * denominator both underflow to zero — a bug that only appears once the test
 * gets long enough to matter, which is to say in production.
 *
 * Unlike maximum likelihood, EAP is defined for every pattern including the
 * empty one, where it returns the prior mean. That is what makes it the natural
 * estimator for the opening items of an adaptive test.
 */
export function estimateEap(
  responses: readonly ScoredResponse[],
  options: BayesOptions = {},
): PosteriorEstimate {
  const prior = options.prior ?? STANDARD_NORMAL_PRIOR;
  normalPrior(prior.mean, prior.sd);
  const rule = ruleFor(prior, options);

  const logLikelihoods = rule.nodes.map((theta) => logLikelihood(responses, theta));
  const maxLogLikelihood = Math.max(...logLikelihoods);

  let mass = 0;
  let firstMoment = 0;
  let secondMoment = 0;
  for (let k = 0; k < rule.nodes.length; k += 1) {
    const theta = rule.nodes[k] as number;
    const weight =
      (rule.weights[k] as number) * Math.exp((logLikelihoods[k] as number) - maxLogLikelihood);
    mass += weight;
    firstMoment += weight * theta;
    secondMoment += weight * theta * theta;
  }

  if (mass === 0 || !Number.isFinite(mass)) {
    throw new Error(
      'estimateEap: the posterior has no mass on the quadrature grid; ' +
        'the prior is probably far from the region the responses support',
    );
  }

  const theta = firstMoment / mass;
  const varianceEstimate = Math.max(secondMoment / mass - theta * theta, 0);
  const posteriorSd = Math.sqrt(varianceEstimate);

  return {
    theta,
    standardError: posteriorSd,
    posteriorSd,
    method: 'eap',
    converged: true,
    boundary: 'none',
    iterations: rule.nodes.length,
  };
}

export interface MapOptions extends BayesOptions {
  /** Absolute tolerance on the estimate. Default 1e-8. */
  readonly tolerance?: number;
  /** Iteration budget for the root finder. Default 100. */
  readonly maxIterations?: number;
  /** How many prior standard deviations to search on each side. Default 8. */
  readonly searchWidth?: number;
}

/**
 * Maximum a posteriori (MAP) ability estimate — the mode of the posterior.
 *
 * Solves `S(theta) + d/dtheta log prior(theta) = 0`, reusing the same bracketed
 * solver as maximum likelihood. A proper normal prior guarantees the posterior
 * score is positive at the bottom of a wide enough window and negative at the
 * top, so a bracket always exists — which is why MAP, unlike MLE, has a finite
 * answer for all-correct and all-incorrect patterns.
 *
 * The reported standard error is `1 / sqrt(I(theta) + 1/sd^2)`: the information
 * in the responses plus the information in the prior.
 */
export function estimateMap(
  responses: readonly ScoredResponse[],
  options: MapOptions = {},
): PosteriorEstimate {
  const prior = options.prior ?? STANDARD_NORMAL_PRIOR;
  normalPrior(prior.mean, prior.sd);
  const tolerance = options.tolerance ?? 1e-8;
  const maxIterations = options.maxIterations ?? 100;
  const searchWidth = options.searchWidth ?? 8;
  if (searchWidth <= 0) {
    throw new RangeError(`estimateMap: searchWidth must be positive, received ${searchWidth}`);
  }

  const lo = prior.mean - searchWidth * prior.sd;
  const hi = prior.mean + searchWidth * prior.sd;

  const posteriorScore = (theta: number): number =>
    scoreFunction(responses, theta) + priorScore(prior, theta);
  const posteriorScoreDerivative = (theta: number): number =>
    -observedInformation(responses, theta) - priorInformation(prior);

  const bracket = bracketSignChange(posteriorScore, lo, hi, 64);
  if (bracket === null) {
    throw new Error(
      `estimateMap: no stationary point of the posterior in [${lo}, ${hi}]; ` +
        'widen searchWidth or check the item parameters',
    );
  }

  const [bracketLo, bracketHi] = bracket;
  const result =
    bracketLo === bracketHi
      ? { root: bracketLo, converged: true, iterations: 0 }
      : safeguardedRoot(posteriorScore, posteriorScoreDerivative, bracketLo, bracketHi, {
          tolerance,
          maxIterations,
        });

  const totalInformation = expectedInformation(responses, result.root) + priorInformation(prior);
  const posteriorSd = 1 / Math.sqrt(totalInformation);

  return {
    theta: result.root,
    standardError: posteriorSd,
    posteriorSd,
    method: 'map',
    converged: result.converged,
    boundary: 'none',
    iterations: result.iterations,
  };
}
