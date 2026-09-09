import { requireFinite } from '../core/numeric.js';
import type { ScoredResponse } from '../models/response.js';
import { logLikelihood } from './likelihood.js';
import {
  normalPrior,
  priorLogDensity,
  STANDARD_NORMAL_PRIOR,
  type NormalPrior,
} from './prior.js';

export interface PosteriorOptions {
  /** Prior over ability. Defaults to standard normal. */
  readonly prior?: NormalPrior;
  /** Lower edge of the grid. Defaults to `prior.mean - halfWidth * prior.sd`. */
  readonly lower?: number;
  /** Upper edge of the grid. Defaults to `prior.mean + halfWidth * prior.sd`. */
  readonly upper?: number;
  /** How many prior standard deviations to cover on each side. Defaults to 6. */
  readonly halfWidth?: number;
  /** Number of grid points, at least 3. Defaults to 241. */
  readonly points?: number;
}

/**
 * A posterior over ability, evaluated on an evenly spaced grid.
 *
 * `density` is normalised so that the trapezoid rule over `grid` integrates to
 * one, which means the summary moments and the cumulative distribution are all
 * computed on the same footing as the curve a caller draws.
 */
export interface Posterior {
  /** Ability values, evenly spaced and strictly increasing. */
  readonly grid: readonly number[];
  /** Posterior density at each grid point, normalised over the grid. */
  readonly density: readonly number[];
  /** Spacing between adjacent grid points. */
  readonly step: number;
  /** Posterior mean — the EAP estimate, computed on this grid. */
  readonly mean: number;
  /** Posterior standard deviation. */
  readonly sd: number;
  /** Posterior mode — the MAP estimate, interpolated between grid points. */
  readonly mode: number;
  /**
   * The larger of the two edge densities, divided by the peak density.
   *
   * A diagnostic on the grid rather than on the posterior: everything reported
   * here is conditioned on the posterior lying inside `[lower, upper]`, and
   * this is how that assumption is checked. Near zero means the density has
   * died away well before the bounds and the truncation costs nothing;
   * appreciably above zero means real mass is being cut off and the moments are
   * biased inward. It is a ratio of two densities, so it is exact — no estimate
   * of the neglected tail is involved, which would need the normalising
   * constant over the whole line that truncation is precisely what denies you.
   */
  readonly edgeRatio: number;
}

/** An equal-tailed credible interval over a posterior. */
export interface CredibleInterval {
  readonly lower: number;
  readonly upper: number;
  /** The central mass the interval covers, as requested. */
  readonly mass: number;
  /** Width of the interval, the quantity that shrinks as a test proceeds. */
  readonly width: number;
}

/**
 * Trapezoid integral of `values` sampled at even spacing `step`.
 *
 * The trapezoid rule rather than Simpson's, deliberately: the caller draws this
 * same density as a polyline between the same grid points, so the area the
 * trapezoid rule computes is exactly the area the reader sees. Simpson's rule
 * would integrate a quadratic through each triple and report a mass that the
 * picture does not have.
 */
function trapezoid(values: readonly number[], step: number): number {
  const last = values.length - 1;
  let total = ((values[0] as number) + (values[last] as number)) / 2;
  for (let i = 1; i < last; i += 1) {
    total += values[i] as number;
  }
  return total * step;
}

/**
 * Refine the location of a maximum by fitting a parabola through the highest
 * sample and its two neighbours.
 *
 * Reporting the grid point with the largest density would make the mode a
 * multiple of the grid spacing — visibly quantised, and jumping between two
 * values as responses accumulate even when the true mode moves smoothly. A
 * three-point parabola recovers the vertex to well inside one spacing, which is
 * enough that the grid resolution stops being observable in the output.
 */
function refinePeak(grid: readonly number[], density: readonly number[]): number {
  let peak = 0;
  for (let i = 1; i < density.length; i += 1) {
    if ((density[i] as number) > (density[peak] as number)) {
      peak = i;
    }
  }
  if (peak === 0 || peak === density.length - 1) {
    return grid[peak] as number;
  }

  const left = density[peak - 1] as number;
  const centre = density[peak] as number;
  const right = density[peak + 1] as number;
  const curvature = left - 2 * centre + right;
  if (curvature >= 0) {
    return grid[peak] as number;
  }

  const offset = (0.5 * (left - right)) / curvature;
  const step = (grid[1] as number) - (grid[0] as number);
  return (grid[peak] as number) + offset * step;
}

/**
 * Evaluate the posterior over ability on an evenly spaced grid.
 *
 * The posterior is proportional to the likelihood times the prior, so the log
 * posterior is the sum of the two log densities. The maximum is subtracted
 * before exponentiating, for the same reason `estimateEap` does it: on a
 * forty-item pattern every unshifted likelihood is around `1e-20`, and
 * normalising a vector of those underflows to zero over the whole grid.
 *
 * Even spacing is the point. `estimateEap` uses Gauss-Hermite nodes, which
 * cluster near the prior mean and are the right choice for computing a mean —
 * but a band drawn between them has vertices that bunch in the middle and
 * stretch at the tails, and the eye reads that as structure in the posterior
 * rather than structure in the quadrature.
 */
export function posteriorDensity(
  responses: readonly ScoredResponse[],
  options: PosteriorOptions = {},
): Posterior {
  const prior = options.prior ?? STANDARD_NORMAL_PRIOR;
  normalPrior(prior.mean, prior.sd);

  const halfWidth = options.halfWidth ?? 6;
  requireFinite(halfWidth, 'posteriorDensity halfWidth');
  if (halfWidth <= 0) {
    throw new RangeError(`posteriorDensity: halfWidth must be positive, received ${halfWidth}`);
  }

  const lower = options.lower ?? prior.mean - halfWidth * prior.sd;
  const upper = options.upper ?? prior.mean + halfWidth * prior.sd;
  requireFinite(lower, 'posteriorDensity lower bound');
  requireFinite(upper, 'posteriorDensity upper bound');
  if (upper <= lower) {
    throw new RangeError(
      `posteriorDensity: upper bound must exceed lower bound, received [${lower}, ${upper}]`,
    );
  }

  const points = options.points ?? 241;
  if (!Number.isInteger(points) || points < 3) {
    throw new RangeError(`posteriorDensity: points must be an integer of at least 3, received ${points}`);
  }

  const step = (upper - lower) / (points - 1);
  const grid = new Array<number>(points);
  const logPosterior = new Array<number>(points);
  for (let i = 0; i < points; i += 1) {
    const theta = i === points - 1 ? upper : lower + i * step;
    grid[i] = theta;
    logPosterior[i] = logLikelihood(responses, theta) + priorLogDensity(prior, theta);
  }

  const peak = Math.max(...logPosterior);
  if (!Number.isFinite(peak)) {
    throw new Error(
      'posteriorDensity: the log posterior is not finite anywhere on the grid; ' +
        'check the item parameters and the grid bounds',
    );
  }

  const unnormalised = logPosterior.map((value) => Math.exp(value - peak));
  const mass = trapezoid(unnormalised, step);
  if (mass <= 0 || !Number.isFinite(mass)) {
    throw new Error(
      'posteriorDensity: the posterior has no mass on the grid; ' +
        'widen the bounds or check that the prior overlaps the responses',
    );
  }

  const density = unnormalised.map((value) => value / mass);

  const peakDensity = Math.max(...density);
  const edgeRatio =
    peakDensity > 0
      ? Math.max(density[0] as number, density[points - 1] as number) / peakDensity
      : 0;

  const weighted = grid.map((theta, i) => theta * (density[i] as number));
  const mean = trapezoid(weighted, step);

  const centred = grid.map((theta, i) => (theta - mean) * (theta - mean) * (density[i] as number));
  const varianceEstimate = Math.max(trapezoid(centred, step), 0);

  return {
    grid,
    density,
    step,
    mean,
    sd: Math.sqrt(varianceEstimate),
    mode: refinePeak(grid, density),
    edgeRatio,
  };
}

/**
 * Cumulative posterior mass at each grid point.
 *
 * Built with the same trapezoid rule the density was normalised with, so the
 * last entry is one to floating-point tolerance rather than to within whatever
 * a different rule would have said, and rescaled so that it is exactly one.
 * `credibleInterval` inverts this, and an inversion against a cumulative
 * distribution that does not reach its own maximum silently clips the upper
 * tail.
 */
export function posteriorCdf(posterior: Posterior): number[] {
  const { density, step } = posterior;
  const cumulative = new Array<number>(density.length);
  cumulative[0] = 0;
  for (let i = 1; i < density.length; i += 1) {
    cumulative[i] =
      (cumulative[i - 1] as number) +
      (((density[i - 1] as number) + (density[i] as number)) / 2) * step;
  }

  const total = cumulative[density.length - 1] as number;
  if (total <= 0) {
    throw new Error('posteriorCdf: the posterior has no mass');
  }
  return cumulative.map((value) => value / total);
}

/**
 * Invert a cumulative distribution at `probability` by linear interpolation
 * between the two grid points that bracket it.
 */
function quantile(grid: readonly number[], cdf: readonly number[], probability: number): number {
  if (probability <= 0) {
    return grid[0] as number;
  }
  if (probability >= 1) {
    return grid[grid.length - 1] as number;
  }

  let hi = 1;
  while (hi < cdf.length - 1 && (cdf[hi] as number) < probability) {
    hi += 1;
  }
  const lo = hi - 1;

  const cdfLo = cdf[lo] as number;
  const cdfHi = cdf[hi] as number;
  const span = cdfHi - cdfLo;
  if (span <= 0) {
    return grid[lo] as number;
  }

  const weight = (probability - cdfLo) / span;
  return (grid[lo] as number) + weight * ((grid[hi] as number) - (grid[lo] as number));
}

/**
 * Equal-tailed credible interval over a posterior.
 *
 * Equal-tailed rather than highest-density, for a reportability reason rather
 * than a numerical one: an equal-tailed interval is the pair of posterior
 * quantiles, so "2.5% of the posterior lies below this candidate's interval" is
 * a sentence that survives being quoted in a score report. A highest-density
 * interval is narrower on a skewed posterior, but its endpoints are not
 * quantiles of anything and it can be disjoint on a bimodal posterior — which a
 * mixed response pattern on high-guessing items really can produce.
 *
 * The interval is drawn from the same grid the density is drawn from, so the
 * band a reader sees and the numbers underneath it cannot disagree.
 */
export function credibleInterval(posterior: Posterior, mass = 0.95): CredibleInterval {
  requireFinite(mass, 'credibleInterval mass');
  if (mass <= 0 || mass >= 1) {
    throw new RangeError(`credibleInterval: mass must lie in (0, 1), received ${mass}`);
  }

  const cdf = posteriorCdf(posterior);
  const tail = (1 - mass) / 2;
  const lower = quantile(posterior.grid, cdf, tail);
  const upper = quantile(posterior.grid, cdf, 1 - tail);

  return { lower, upper, mass, width: upper - lower };
}

/**
 * The posterior mass falling inside `[lower, upper]`.
 *
 * The complement of `credibleInterval`: that one asks which interval holds a
 * given mass, this one asks what mass a given interval holds. It is what
 * answers "how confident are we that this candidate is above the cut score",
 * which is the question a testing programme actually applies a decision to.
 */
export function posteriorMassBetween(
  posterior: Posterior,
  lower: number,
  upper: number,
): number {
  requireFinite(lower, 'posteriorMassBetween lower');
  requireFinite(upper, 'posteriorMassBetween upper');
  if (upper < lower) {
    throw new RangeError(
      `posteriorMassBetween: upper must not be below lower, received [${lower}, ${upper}]`,
    );
  }

  const cdf = posteriorCdf(posterior);
  const { grid } = posterior;
  const at = (theta: number): number => {
    if (theta <= (grid[0] as number)) {
      return 0;
    }
    if (theta >= (grid[grid.length - 1] as number)) {
      return 1;
    }
    let hi = 1;
    while (hi < grid.length - 1 && (grid[hi] as number) < theta) {
      hi += 1;
    }
    const lo = hi - 1;
    const span = (grid[hi] as number) - (grid[lo] as number);
    const weight = span <= 0 ? 0 : (theta - (grid[lo] as number)) / span;
    return (cdf[lo] as number) + weight * ((cdf[hi] as number) - (cdf[lo] as number));
  };

  return Math.min(1, Math.max(0, at(upper) - at(lower)));
}
