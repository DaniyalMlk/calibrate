import { requireFinite } from '../core/numeric.js';

/**
 * A normal prior over ability.
 *
 * The ability scale is only defined up to a linear transformation, and the
 * convention throughout this engine — as in most operational testing programmes
 * — is that the calibration population is standard normal. A prior is therefore
 * not a statement of belief about an individual so much as a statement about the
 * population the item parameters were calibrated on.
 */
export interface NormalPrior {
  readonly mean: number;
  readonly sd: number;
}

/** The standard normal prior, the default throughout. */
export const STANDARD_NORMAL_PRIOR: NormalPrior = Object.freeze({ mean: 0, sd: 1 });

/** Build and validate a normal prior. */
export function normalPrior(mean = 0, sd = 1): NormalPrior {
  requireFinite(mean, 'prior mean');
  requireFinite(sd, 'prior standard deviation');
  if (sd <= 0) {
    throw new RangeError(`prior standard deviation must be positive, received ${sd}`);
  }
  return Object.freeze({ mean, sd });
}

/** Log density of the prior at `theta`, including the normalising constant. */
export function priorLogDensity(prior: NormalPrior, theta: number): number {
  const z = (theta - prior.mean) / prior.sd;
  return -0.5 * z * z - Math.log(prior.sd) - 0.5 * Math.log(2 * Math.PI);
}

/**
 * Derivative of the prior log density: `-(theta - mean) / sd^2`.
 *
 * Added to the likelihood's score function to get the score of the posterior,
 * which is what MAP estimation solves for.
 */
export function priorScore(prior: NormalPrior, theta: number): number {
  return -(theta - prior.mean) / (prior.sd * prior.sd);
}

/**
 * Curvature contributed by the prior: `1 / sd^2`.
 *
 * Constant for a normal prior, and the reason a posterior standard deviation is
 * always finite even when the likelihood alone carries no information.
 */
export function priorInformation(prior: NormalPrior): number {
  return 1 / (prior.sd * prior.sd);
}
