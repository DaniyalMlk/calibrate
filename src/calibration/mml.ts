import { clamp } from '../core/numeric.js';
import { normalGaussHermiteRule, type QuadratureRule } from '../core/quadrature.js';
import { MISSING, type ResponseMatrix } from './matrix.js';
import { screenExtremeItems, type ScreenResult } from './screen.js';
import {
  expectationStep,
  posteriorOverNodes,
  pointProbability,
  type ItemPoint,
} from './expected.js';
import { countStandardErrors, maximiseItem, type MaximiseOptions } from './maximise.js';
import { toItems, type CalibratedItem, type CalibrationModel } from './jmle.js';
import type { Item } from '../models/item.js';

/** How the standard errors of the item parameters are computed. */
export type ErrorMethod = 'cross-product' | 'expected-counts';

/** How the population distribution is treated during estimation. */
export type LatentDistribution = 'normal' | 'empirical';

export interface MarginalOptions {
  /** Model to fit. Default `rasch`. */
  readonly model?: CalibrationModel;
  /**
   * Quadrature rule standing in for the population. Defaults to a 41-point
   * Gauss-Hermite rule on the standard normal.
   */
  readonly rule?: QuadratureRule;
  /**
   * Whether the population weights are held at the rule's own or re-estimated
   * from the data on every cycle. Default `normal`.
   */
  readonly latent?: LatentDistribution;
  /** Relative change in the marginal log-likelihood accepted as converged. Default 1e-7. */
  readonly tolerance?: number;
  /** Ceiling on EM cycles. Default 500. */
  readonly maxIterations?: number;
  /** Bounds on the estimated discrimination. Default 0.05 to 4. */
  readonly discriminationBounds?: readonly [number, number];
  /** Bounds on the estimated difficulties. Default -6 to 6. */
  readonly difficultyBounds?: readonly [number, number];
  /** Which standard errors to report. Default `cross-product`. */
  readonly errors?: ErrorMethod;
}

export interface MarginalItem extends CalibratedItem {
  /** Whether this item's maximisation step reached a stationary point. */
  readonly stationary: boolean;
  /** Whether either of its parameters finished pinned to a bound. */
  readonly atBound: boolean;
}

export interface MarginalResult {
  readonly model: CalibrationModel;
  readonly items: readonly MarginalItem[];
  /** The marginal log-likelihood at the returned parameters. */
  readonly logLikelihood: number;
  /**
   * The marginal log-likelihood at the starting values, followed by its value
   * after each EM cycle. Non-decreasing when the population is held fixed.
   */
  readonly history: readonly number[];
  readonly iterations: number;
  readonly converged: boolean;
  /** The population weights actually used, which differ from the rule's if estimated. */
  readonly population: QuadratureRule;
  /**
   * Shape of the fitted population, as skewness and excess kurtosis.
   *
   * Its mean and variance say nothing — they are fixed at zero and one, because
   * that is what defines the metric the item parameters are reported on. What is
   * left free, and what a fitted population can therefore tell you, is its
   * shape: whether the respondents piled up at one end, and whether the tails
   * are heavier or lighter than a normal's. Both are near zero when the
   * population is held fixed, and both are worth looking at before trusting a
   * calibration that assumed normality.
   */
  readonly populationSkewness: number;
  readonly populationExcessKurtosis: number;
  /** Which items, if any, were removed before estimation. */
  readonly screening: ScreenResult;
  readonly errorMethod: ErrorMethod;
}

/** Turn a marginal calibration into items the rest of the engine can use. */
export function toMarginalItems(result: MarginalResult): Item[] {
  return toItems({
    model: result.model,
    items: result.items,
    abilities: [],
    personIds: [],
    iterations: result.iterations,
    converged: result.converged,
    maxChange: 0,
    screening: result.screening,
  });
}

/**
 * Marginal maximum likelihood calibration by expectation-maximisation.
 *
 * The joint estimator in `jmle.ts` treats each respondent's ability as a
 * parameter to be estimated. That is the source of its one structural defect:
 * the number of parameters grows with the sample, every ability is estimated
 * from only as many responses as there are items, and the noise in those
 * estimates does not average out as more people are tested. It propagates into
 * the item parameters instead, and the difficulties come out inflated away from
 * zero by roughly `L / (L - 1)` on a test of `L` items. Collecting ten times the
 * data does not help, because ten times the data brings ten times the ability
 * parameters with it.
 *
 * Marginal estimation removes the person parameters rather than estimating them.
 * Ability is treated as a draw from a population distribution and integrated
 * out, leaving a likelihood in the item parameters alone. The parameter count is
 * then fixed — two per item, whatever the sample — and the usual large-sample
 * argument applies again.
 *
 * The integral has no closed form, so it is done on a quadrature rule, and the
 * maximisation is done by EM:
 *
 * - **Expectation.** Given the current item parameters, compute each
 *   respondent's posterior over the ability nodes and accumulate the
 *   posterior-weighted counts: how many people of each ability sat each item,
 *   and how many passed it.
 * - **Maximisation.** Given those counts, re-fit each item independently. The
 *   items decouple completely here, which is the practical reason the method
 *   scales: a thousand-item bank is a thousand two-parameter problems, not one
 *   two-thousand-parameter problem.
 *
 * Each cycle is guaranteed not to decrease the marginal log-likelihood, which is
 * the property the test suite checks rather than assumes — a violation would
 * mean the expectation and maximisation steps disagree about what is being
 * maximised, which is the way this algorithm usually goes wrong.
 *
 * **The scale still has to be pinned.** Shifting every difficulty and the
 * population mean together leaves the likelihood unchanged, exactly as under
 * joint estimation. Holding the population at the standard normal is what fixes
 * it here: the metric is defined as the one in which the population has mean
 * zero and unit variance, and the item parameters are whatever they have to be
 * in that metric. When the population is estimated instead, the estimated
 * weights are re-standardised on every cycle for the same reason.
 */
export function marginalCalibrate(
  matrix: ResponseMatrix,
  options: MarginalOptions = {},
): MarginalResult {
  const model = options.model ?? 'rasch';
  const latent = options.latent ?? 'normal';
  const tolerance = options.tolerance ?? 1e-7;
  const maxIterations = options.maxIterations ?? 500;
  const errorMethod = options.errors ?? 'cross-product';
  const discriminationBounds = options.discriminationBounds ?? ([0.05, 4] as const);
  const difficultyBounds = options.difficultyBounds ?? ([-6, 6] as const);
  const baseRule = options.rule ?? normalGaussHermiteRule(0, 1, 41);

  if (!(tolerance > 0)) {
    throw new RangeError(`marginalCalibrate: tolerance must be positive, received ${tolerance}`);
  }
  if (!Number.isInteger(maxIterations) || maxIterations < 1) {
    throw new RangeError(
      `marginalCalibrate: maxIterations must be a positive integer, received ${maxIterations}`,
    );
  }
  if (!(discriminationBounds[0] > 0 && discriminationBounds[0] < discriminationBounds[1])) {
    throw new RangeError('marginalCalibrate: discrimination bounds must satisfy 0 < min < max');
  }
  if (!(difficultyBounds[0] < difficultyBounds[1])) {
    throw new RangeError('marginalCalibrate: difficulty bounds must satisfy min < max');
  }
  if (baseRule.nodes.length !== baseRule.weights.length) {
    throw new RangeError('marginalCalibrate: quadrature rule has mismatched nodes and weights');
  }

  const screening = screenExtremeItems(matrix);
  const screened = screening.matrix;
  const itemCount = screened.itemCount;

  const maximiseOptions: MaximiseOptions = {
    model,
    discriminationBounds,
    difficultyBounds,
  };

  let items: ItemPoint[] = startingValues(screened, difficultyBounds);
  let rule: QuadratureRule = { nodes: [...baseRule.nodes], weights: [...baseRule.weights] };

  let counts = expectationStep(screened, items, rule);
  const history: number[] = [counts.logLikelihood];
  let logLikelihood = counts.logLikelihood;
  let converged = false;
  let iterations = 0;

  for (; iterations < maxIterations; iterations += 1) {
    const next: ItemPoint[] = [];
    for (let item = 0; item < itemCount; item += 1) {
      const step = maximiseItem(
        items[item] as ItemPoint,
        rule,
        counts.answered[item] as Float64Array,
        counts.correct[item] as Float64Array,
        maximiseOptions,
      );
      next.push(step.item);
    }
    items = next;

    if (latent === 'empirical') {
      // The population's own maximisation step: the posterior mass that landed
      // at each node, averaged over respondents, is the maximiser of the same
      // expected complete-data log-likelihood with respect to the weights.
      const fitted: QuadratureRule = {
        nodes: rule.nodes,
        weights: Array.from(counts.density, (mass) => mass / screened.personCount),
      };
      const moments = weightedMoments(fitted);
      rule = standardise(fitted, moments);
      items = rescaleToPopulation(
        items,
        moments,
        discriminationBounds,
        difficultyBounds,
      );
    }

    counts = expectationStep(screened, items, rule);
    history.push(counts.logLikelihood);

    // Relative change, so the test means the same thing on a hundred
    // respondents as on a hundred thousand, where the log-likelihood itself is
    // a thousand times larger and an absolute threshold would never fire.
    const change = counts.logLikelihood - logLikelihood;
    const scale = Math.max(1, Math.abs(counts.logLikelihood));
    logLikelihood = counts.logLikelihood;
    if (Math.abs(change) / scale < tolerance) {
      converged = true;
      iterations += 1;
      break;
    }
  }

  const errors =
    errorMethod === 'cross-product'
      ? crossProductErrors(screened, items, rule, model)
      : items.map((item, index) =>
          countStandardErrors(
            item,
            rule,
            counts.answered[index] as Float64Array,
            counts.correct[index] as Float64Array,
            model,
          ),
        );

  const shape = populationShape(rule);
  const reported: MarginalItem[] = [];
  for (let index = 0; index < itemCount; index += 1) {
    const item = items[index] as ItemPoint;
    const { correct, answered } = screened.itemScore(index);
    const final = maximiseItem(
      item,
      rule,
      counts.answered[index] as Float64Array,
      counts.correct[index] as Float64Array,
      maximiseOptions,
    );
    const error = errors[index] as { difficulty: number; discrimination: number };
    reported.push({
      id: screened.itemIds[index] as string,
      discrimination: item.discrimination,
      difficulty: item.difficulty,
      difficultyStandardError: error.difficulty,
      discriminationStandardError: error.discrimination,
      answered,
      proportionCorrect: correct / answered,
      stationary: final.stationary,
      atBound: final.atBound,
    });
  }

  return {
    model,
    items: reported,
    logLikelihood,
    history,
    iterations,
    converged,
    population: rule,
    populationSkewness: shape.skewness,
    populationExcessKurtosis: shape.excessKurtosis,
    screening,
    errorMethod,
  };
}

/**
 * Starting values: the logit of each item's failure rate.
 *
 * Exact under the Rasch model when the population is centred at zero, which it
 * is by construction here, so the first maximisation step usually has very
 * little left to do. Starting from zeros instead costs a handful of cycles and,
 * under the 2PL, occasionally sends a discrimination towards a bound on the
 * first pass, from which the damped step recovers only slowly.
 */
function startingValues(
  matrix: ResponseMatrix,
  bounds: readonly [number, number],
): ItemPoint[] {
  const items: ItemPoint[] = [];
  for (let item = 0; item < matrix.itemCount; item += 1) {
    const { correct, answered } = matrix.itemScore(item);
    const p = clamp(correct / answered, 0.01, 0.99);
    items.push({
      discrimination: 1,
      difficulty: clamp(Math.log((1 - p) / p), bounds[0], bounds[1]),
    });
  }
  return items;
}

/**
 * Skewness and excess kurtosis of a distribution given as nodes and weights.
 *
 * Both standardised by the distribution's own spread, so they describe shape
 * alone and are unaffected by the metric the population happens to sit on. A
 * normal population gives zero for both, to quadrature accuracy.
 */
function populationShape(rule: QuadratureRule): { skewness: number; excessKurtosis: number } {
  const { mean, sd } = weightedMoments(rule);
  if (!(sd > 1e-8)) return { skewness: 0, excessKurtosis: 0 };

  let total = 0;
  let third = 0;
  let fourth = 0;
  for (const [k, weight] of rule.weights.entries()) {
    const z = ((rule.nodes[k] as number) - mean) / sd;
    total += weight;
    third += weight * z * z * z;
    fourth += weight * z * z * z * z;
  }
  if (!(total > 0)) return { skewness: 0, excessKurtosis: 0 };
  return { skewness: third / total, excessKurtosis: fourth / total - 3 };
}

/** Mean and standard deviation of a distribution given as nodes and weights. */
function weightedMoments(rule: QuadratureRule): { mean: number; sd: number } {
  let total = 0;
  let mean = 0;
  for (const [k, weight] of rule.weights.entries()) {
    total += weight;
    mean += weight * (rule.nodes[k] as number);
  }
  if (!(total > 0)) return { mean: 0, sd: 1 };
  mean /= total;

  let variance = 0;
  for (const [k, weight] of rule.weights.entries()) {
    const centred = (rule.nodes[k] as number) - mean;
    variance += weight * centred * centred;
  }
  return { mean, sd: Math.sqrt(variance / total) };
}

/**
 * Re-express a fitted population on the standard metric, keeping the nodes.
 *
 * The nodes have to stay where they are: the counts, the response probabilities
 * and the previous cycle's rule are all indexed by node, and a rule whose nodes
 * move between cycles cannot be compared with any of them. So the distribution
 * is shifted and stretched by moving its *mass* — each node's weight is
 * deposited at the position that node maps to, split linearly between the two
 * nodes bracketing it.
 *
 * That interpolation is the one approximation in the whole procedure. In the
 * continuum, standardising the population and transforming the item parameters
 * to match leaves every response probability and therefore the likelihood
 * exactly unchanged; on a fixed node set the re-deposited mass is close to but
 * not identical to the shifted distribution. The consequence is small and worth
 * stating plainly: with the population held fixed the marginal log-likelihood is
 * non-decreasing on every cycle and the test suite asserts it exactly, while
 * with the population estimated it can dip by a fraction of the interpolation
 * error, and the suite asserts only that it does not dip meaningfully and that
 * the run as a whole improves the fit.
 */
function standardise(
  rule: QuadratureRule,
  moments: { mean: number; sd: number },
): QuadratureRule {
  const { nodes, weights } = rule;
  if (!(moments.sd > 1e-8)) return rule;

  const moved = new Array<number>(nodes.length).fill(0);
  for (const [k, weight] of weights.entries()) {
    const target = ((nodes[k] as number) - moments.mean) / moments.sd;
    spread(moved, nodes, target, weight);
  }

  let total = 0;
  for (const mass of moved) total += mass;
  return { nodes, weights: total > 0 ? moved.map((mass) => mass / total) : weights };
}

/** Deposit `mass` at position `target` onto the two nearest nodes, linearly. */
function spread(
  into: number[],
  nodes: readonly number[],
  target: number,
  mass: number,
): void {
  const last = nodes.length - 1;
  if (target <= (nodes[0] as number)) {
    into[0] = (into[0] as number) + mass;
    return;
  }
  if (target >= (nodes[last] as number)) {
    into[last] = (into[last] as number) + mass;
    return;
  }
  let hi = 1;
  while (hi < last && (nodes[hi] as number) < target) hi += 1;
  const lo = hi - 1;
  const left = nodes[lo] as number;
  const right = nodes[hi] as number;
  const share = right > left ? (target - left) / (right - left) : 0;
  into[lo] = (into[lo] as number) + mass * (1 - share);
  into[hi] = (into[hi] as number) + mass * share;
}

/**
 * Put the item parameters back on the metric of a re-standardised population.
 *
 * When the population weights are re-estimated they drift off mean zero and unit
 * variance, and standardising them changes the metric. The item parameters have
 * to follow, by exactly the transformation that leaves every response
 * probability unchanged: difficulties shift and stretch with the scale,
 * discriminations move the other way.
 */
function rescaleToPopulation(
  items: readonly ItemPoint[],
  moments: { mean: number; sd: number },
  discriminationBounds: readonly [number, number],
  difficultyBounds: readonly [number, number],
): ItemPoint[] {
  const { mean, sd } = moments;
  if (!(sd > 1e-8)) return [...items];

  // Writing theta = mean + sd * theta', the exponent a(theta - b) becomes
  // a*sd (theta' - (b - mean)/sd), so the transformation below leaves every
  // response probability exactly where it was.
  return items.map((item) => ({
    discrimination: clamp(
      item.discrimination * sd,
      discriminationBounds[0],
      discriminationBounds[1],
    ),
    difficulty: clamp((item.difficulty - mean) / sd, difficultyBounds[0], difficultyBounds[1]),
  }));
}

/**
 * Standard errors from the outer product of the respondents' score contributions.
 *
 * The curvature-based errors in `maximise.ts` are computed from the expected
 * counts as though those counts had been observed. They were not: they were
 * reconstructed from responses whose abilities are unknown, and treating a
 * reconstruction as data throws away the uncertainty in the reconstruction. The
 * resulting intervals are too narrow.
 *
 * This estimator avoids that by working on the marginal likelihood itself. Its
 * gradient for one respondent is not something extra to derive — by Fisher's
 * identity it is the posterior expectation of the complete-data gradient, which
 * is the same quantity the expectation step already averages, only accumulated
 * per respondent instead of pooled across them. Summing the outer products of
 * those per-respondent gradients estimates the information in the marginal
 * likelihood directly, uncertainty in the ability posteriors included.
 *
 * It is the cheaper of the two correct options — the alternative differentiates
 * the marginal likelihood twice — and it needs only one extra pass over the
 * data. What it asks in return is respondents: the outer product of `n` vectors
 * is a noisy estimate of a covariance when `n` is small, so on a few dozen
 * respondents it is the less stable choice, and the curvature form is the one
 * to fall back to.
 */
export function crossProductErrors(
  matrix: ResponseMatrix,
  items: readonly ItemPoint[],
  rule: QuadratureRule,
  model: CalibrationModel,
): { difficulty: number; discrimination: number }[] {
  const nodeCount = rule.nodes.length;
  const itemCount = items.length;
  const posterior = new Float64Array(nodeCount);

  // Per item, the accumulated cross-products of the two score components.
  const sumAA = new Float64Array(itemCount);
  const sumBB = new Float64Array(itemCount);
  const sumAB = new Float64Array(itemCount);

  // The response probabilities do not depend on the respondent, so they are
  // computed once per item per node rather than once per respondent per item.
  const probability = items.map((item) => {
    const row = new Float64Array(nodeCount);
    for (let k = 0; k < nodeCount; k += 1) row[k] = pointProbability(item, rule.nodes[k] as number);
    return row;
  });

  for (let person = 0; person < matrix.personCount; person += 1) {
    const row = matrix.row(person);
    posteriorOverNodes(row, items, rule, posterior);

    for (const [item, cell] of row.entries()) {
      if (cell === MISSING) continue;
      const point = items[item] as ItemPoint;
      const probabilities = probability[item] as Float64Array;
      let scoreA = 0;
      let scoreB = 0;
      for (let k = 0; k < nodeCount; k += 1) {
        const mass = posterior[k] as number;
        if (mass === 0) continue;
        const residual = cell - (probabilities[k] as number);
        scoreA += mass * residual * ((rule.nodes[k] as number) - point.difficulty);
        scoreB -= mass * point.discrimination * residual;
      }
      sumAA[item] = (sumAA[item] as number) + scoreA * scoreA;
      sumBB[item] = (sumBB[item] as number) + scoreB * scoreB;
      sumAB[item] = (sumAB[item] as number) + scoreA * scoreB;
    }
  }

  const errors: { difficulty: number; discrimination: number }[] = [];
  for (let item = 0; item < itemCount; item += 1) {
    const iaa = sumAA[item] as number;
    const ibb = sumBB[item] as number;
    const iab = sumAB[item] as number;

    if (model === 'rasch') {
      errors.push({
        difficulty: ibb > 0 ? 1 / Math.sqrt(ibb) : Number.POSITIVE_INFINITY,
        discrimination: 0,
      });
      continue;
    }

    const determinant = iaa * ibb - iab * iab;
    if (!(determinant > 1e-14) || iaa <= 0 || ibb <= 0) {
      errors.push({
        difficulty: Number.POSITIVE_INFINITY,
        discrimination: Number.POSITIVE_INFINITY,
      });
      continue;
    }
    errors.push({
      difficulty: Math.sqrt(iaa / determinant),
      discrimination: Math.sqrt(ibb / determinant),
    });
  }
  return errors;
}
