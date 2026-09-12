/**
 * Differential item functioning by logistic regression.
 *
 * The matched-group methods pool a comparison across strata, and pooling is what
 * costs them the non-uniform case: an item whose response curves cross gives
 * each group an advantage on its own side of the crossing point, and an average
 * over strata puts those advantages together and reports the remainder. How much
 * remainder there is depends on where the two groups happen to sit relative to
 * the crossing, which is not a property of the item.
 *
 * Regression does not pool. The response is modelled on the matching score, on
 * group membership, and on their product; the group term is the uniform
 * component and the product term is the interaction the pooled methods cannot
 * express. Nested likelihood-ratio tests then separate the two, and the price of
 * that separation is a model that has to be fitted and can fail to fit —
 * which is why convergence is reported rather than assumed.
 */

import { chiSquareUpperTail } from '../core/gamma.js';
import {
  invertMatrix,
  solveLinear,
  transposeProduct,
  weightedCrossProduct,
} from '../core/linear.js';
import { logistic } from '../core/numeric.js';
import type { DifCategory } from './mantel.js';
import type { DifObservation } from './strata.js';

export interface LogisticFit {
  /** One coefficient per column of the design matrix. */
  readonly coefficients: readonly number[];
  /** Square roots of the diagonal of the inverse information. */
  readonly standardErrors: readonly number[];
  readonly logLikelihood: number;
  readonly iterations: number;
  readonly converged: boolean;
}

export interface IrlsOptions {
  /** Newton steps before giving up. Default 50. */
  readonly maxIterations?: number;
  /** Convergence tolerance on the log-likelihood. Default 1e-10. */
  readonly tolerance?: number;
}

/**
 * Fit a binary logistic regression by iteratively reweighted least squares.
 *
 * Newton-Raphson on the log-likelihood, which for this model is the same thing
 * as a weighted least squares fit whose weights are recomputed each step. The
 * likelihood is globally concave whenever the design has full rank and the data
 * are not separable, so there is no starting-value problem and no local optimum
 * to fall into — every failure to converge here is separation or collinearity,
 * and both deserve to be reported rather than smoothed over.
 *
 * Weights are floored away from zero. As a fitted probability approaches 0 or 1
 * its weight `p(1 - p)` approaches zero with it, and a design matrix whose rows
 * have all been multiplied by nothing is singular — the fit would fail with a
 * linear algebra error rather than the statistical one it actually is.
 */
export function fitLogistic(
  design: readonly (readonly number[])[],
  outcomes: readonly number[],
  options: IrlsOptions = {},
): LogisticFit {
  const rows = design.length;
  if (rows === 0) throw new RangeError('fitLogistic: no observations');
  if (outcomes.length !== rows) {
    throw new RangeError(`fitLogistic: ${outcomes.length} outcomes for ${rows} rows`);
  }
  const width = (design[0] as readonly number[]).length;
  if (width === 0) throw new RangeError('fitLogistic: the design matrix has no columns');
  if (rows <= width) {
    throw new RangeError(`fitLogistic: ${rows} observations cannot determine ${width} coefficients`);
  }
  for (const [index, outcome] of outcomes.entries()) {
    if (outcome !== 0 && outcome !== 1) {
      throw new RangeError(`fitLogistic: outcome ${index} is ${outcome}, expected 0 or 1`);
    }
  }

  const maxIterations = options.maxIterations ?? 50;
  const tolerance = options.tolerance ?? 1e-10;
  const MINIMUM_WEIGHT = 1e-10;

  let coefficients = new Array<number>(width).fill(0);
  let logLikelihood = Number.NEGATIVE_INFINITY;
  let converged = false;
  let iterations = 0;

  for (let step = 1; step <= maxIterations; step += 1) {
    iterations = step;
    const weights = new Array<number>(rows);
    const residuals = new Array<number>(rows);
    let current = 0;

    for (let row = 0; row < rows; row += 1) {
      const values = design[row] as readonly number[];
      let linear = 0;
      for (let i = 0; i < width; i += 1) linear += (coefficients[i] as number) * (values[i] as number);
      const probability = logistic(linear);
      const outcome = outcomes[row] as number;
      // The log-likelihood written through the linear predictor rather than
      // through log(p): at a large negative predictor `p` underflows to zero and
      // `log(0)` is negative infinity, while this form stays finite.
      current += outcome * linear - softplus(linear);
      weights[row] = Math.max(probability * (1 - probability), MINIMUM_WEIGHT);
      residuals[row] = outcome - probability;
    }

    if (Number.isFinite(logLikelihood) && Math.abs(current - logLikelihood) < tolerance) {
      logLikelihood = current;
      converged = true;
      break;
    }
    logLikelihood = current;

    const information = weightedCrossProduct(design, weights);
    const gradient = transposeProduct(design, residuals);
    let stepDirection: number[];
    try {
      stepDirection = solveLinear(information, gradient);
    } catch {
      // A singular information matrix means the data cannot determine these
      // coefficients. Stop at the last good estimate rather than stepping into
      // nonsense, and let the caller see that it did not converge.
      break;
    }
    coefficients = coefficients.map((value, index) => value + (stepDirection[index] as number));
  }

  const finalWeights = design.map((values) => {
    let linear = 0;
    for (let i = 0; i < width; i += 1) linear += (coefficients[i] as number) * (values[i] as number);
    const probability = logistic(linear);
    return Math.max(probability * (1 - probability), MINIMUM_WEIGHT);
  });

  let standardErrors: number[];
  try {
    const covariance = invertMatrix(weightedCrossProduct(design, finalWeights));
    standardErrors = covariance.map((row, index) => Math.sqrt(Math.max(0, row[index] as number)));
  } catch {
    standardErrors = new Array<number>(width).fill(Number.POSITIVE_INFINITY);
    converged = false;
  }

  return { coefficients, standardErrors, logLikelihood, iterations, converged };
}

/** `log(1 + exp(x))`, without overflowing for a large positive `x`. */
function softplus(x: number): number {
  return x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x));
}

export interface LikelihoodRatioTest {
  readonly chiSquare: number;
  readonly degreesOfFreedom: number;
  readonly pValue: number;
}

function ratioTest(restricted: number, full: number, degreesOfFreedom: number): LikelihoodRatioTest {
  // Floored at zero: the full model cannot fit worse than the model nested
  // inside it, so a negative value is rounding rather than evidence.
  const chiSquare = Math.max(0, 2 * (full - restricted));
  return { chiSquare, degreesOfFreedom, pValue: chiSquareUpperTail(chiSquare, degreesOfFreedom) };
}

/**
 * Nagelkerke's pseudo R-squared, relative to an intercept-only model.
 *
 * Cox and Snell's version is the natural likelihood-ratio analogue but cannot
 * reach 1 for a binary outcome — its ceiling depends on the base rate, so two
 * items with different difficulties are not comparable on it. Nagelkerke's
 * rescales by that ceiling, which is the only reason it is the one reported.
 */
export function nagelkerke(nullLogLikelihood: number, modelLogLikelihood: number, n: number): number {
  if (!Number.isInteger(n) || n < 1) {
    throw new RangeError(`nagelkerke: n must be a positive integer, received ${n}`);
  }
  const coxSnell = 1 - Math.exp((2 / n) * (nullLogLikelihood - modelLogLikelihood));
  const ceiling = 1 - Math.exp((2 / n) * nullLogLikelihood);
  return ceiling <= 0 ? 0 : coxSnell / ceiling;
}

/** A pair of boundaries on the change in pseudo R-squared. */
export interface EffectRule {
  readonly negligible: number;
  readonly large: number;
}

/**
 * The two published cut-point sets, rather than one baked in.
 *
 * They disagree by a factor of nearly four, and the disagreement is real rather
 * than an error in one of them: the wider pair was proposed from substantive
 * judgement about what size of effect ought to matter, the narrower pair from
 * simulation studies asking what the test can actually detect.
 *
 * The narrower pair is the default, on evidence rather than preference. A
 * pseudo-R-squared is not a proportion of variance and does not reach the values
 * the intuition behind the wider pair expects: on an item made a full logit
 * harder for the focal group, with a combined p-value of 1.7e-34, the change in
 * Nagelkerke R-squared is 0.076 — comfortably large under the narrower pair and
 * negligible under the wider one. A classifier that calls that item negligible
 * is not being conservative, it is being wrong, so the wider pair is offered and
 * not chosen.
 */
export const EFFECT_RULES = {
  /** Jodoin and Gierl: calibrated against what the test detects in simulation. */
  jodoinGierl: { negligible: 0.035, large: 0.07 },
  /** Zumbo and Thomas: the original, much wider pair. Flags very little. */
  zumboThomas: { negligible: 0.13, large: 0.26 },
} as const satisfies Readonly<Record<string, EffectRule>>;

export interface LogisticDifOptions extends IrlsOptions {
  /** Effect-size boundaries. Defaults to the Jodoin-Gierl pair. */
  readonly rule?: EffectRule;
  /**
   * Standardise the matching score before fitting. Default true.
   *
   * The interaction column is the product of the matching score and the group
   * indicator, and on a raw total score that column runs from zero to the form
   * length while the indicator runs from zero to one. Centring puts the terms
   * on comparable scales, which matters for the conditioning of the information
   * matrix rather than for the fit: the likelihood-ratio tests are invariant to
   * it, and the coefficients are reported on the standardised scale.
   */
  readonly standardize?: boolean;
}

export interface LogisticDifResult {
  /** Does the item differ between groups at all? Two degrees of freedom. */
  readonly combined: LikelihoodRatioTest;
  /** Is it uniformly harder for one group? One degree of freedom. */
  readonly uniform: LikelihoodRatioTest;
  /** Does the gap change across the ability range? One degree of freedom. */
  readonly nonUniform: LikelihoodRatioTest;
  /** Nagelkerke R-squared gained by adding both group terms. */
  readonly deltaRSquared: number;
  /** The part of that gained by the group term alone. */
  readonly uniformRSquared: number;
  /** Intercept, matching score, group, and the interaction. */
  readonly coefficients: {
    readonly intercept: number;
    readonly match: number;
    readonly group: number;
    readonly interaction: number;
  };
  readonly standardErrors: {
    readonly intercept: number;
    readonly match: number;
    readonly group: number;
    readonly interaction: number;
  };
  readonly classification: DifCategory;
  /** True only if every one of the four nested fits converged. */
  readonly converged: boolean;
  readonly people: number;
  /** Whether the matching score was centred and scaled before fitting. */
  readonly standardized: boolean;
}

/**
 * Test an item for uniform and non-uniform DIF by nested logistic regressions.
 *
 * Four models are fitted: an intercept alone, the matching score, the score
 * plus group, and the score plus group plus their interaction. The differences
 * between successive log-likelihoods are the tests, and the differences between
 * the pseudo R-squareds are the effect sizes.
 *
 * The order matters. The uniform test compares the group model against the
 * score-only model, and the non-uniform test compares the full model against
 * the group model — so the interaction is tested *after* the uniform component
 * has been accounted for. Testing it first would attribute a plain difficulty
 * shift to the interaction whenever the groups' score distributions differ,
 * which they always do.
 */
export function logisticDif(
  observations: readonly DifObservation[],
  options: LogisticDifOptions = {},
): LogisticDifResult {
  if (observations.length === 0) throw new RangeError('logisticDif: no observations');
  const rule = options.rule ?? EFFECT_RULES.jodoinGierl;
  const standardize = options.standardize ?? true;

  let reference = 0;
  let focal = 0;
  for (const observation of observations) {
    if (observation.score !== 0 && observation.score !== 1) {
      throw new RangeError(
        `logisticDif: expects a dichotomous item, received a score of ${observation.score}`,
      );
    }
    if (observation.group === 'focal') focal += 1;
    else reference += 1;
  }
  if (reference === 0 || focal === 0) {
    throw new RangeError('logisticDif: both groups must be present');
  }

  const matches = observations.map((observation) => observation.match);
  const centre = standardize ? matches.reduce((a, b) => a + b, 0) / matches.length : 0;
  const spread = standardize ? standardDeviation(matches, centre) : 1;
  const scaled = matches.map((match) => (match - centre) / spread);

  const outcomes = observations.map((observation) => observation.score);
  const groups = observations.map((observation) => (observation.group === 'focal' ? 1 : 0));

  const intercepts = observations.map(() => [1]);
  const withMatch = scaled.map((match) => [1, match]);
  const withGroup = scaled.map((match, index) => [1, match, groups[index] as number]);
  const withInteraction = scaled.map((match, index) => {
    const group = groups[index] as number;
    return [1, match, group, match * group];
  });

  const nullFit = fitLogistic(intercepts, outcomes, options);
  const matchFit = fitLogistic(withMatch, outcomes, options);
  const groupFit = fitLogistic(withGroup, outcomes, options);
  const fullFit = fitLogistic(withInteraction, outcomes, options);

  const people = observations.length;
  const coefficients = fullFit.coefficients;
  const errors = fullFit.standardErrors;

  const deltaRSquared =
    nagelkerke(nullFit.logLikelihood, fullFit.logLikelihood, people) -
    nagelkerke(nullFit.logLikelihood, matchFit.logLikelihood, people);
  const uniformRSquared =
    nagelkerke(nullFit.logLikelihood, groupFit.logLikelihood, people) -
    nagelkerke(nullFit.logLikelihood, matchFit.logLikelihood, people);

  const combined = ratioTest(matchFit.logLikelihood, fullFit.logLikelihood, 2);

  return {
    combined,
    uniform: ratioTest(matchFit.logLikelihood, groupFit.logLikelihood, 1),
    nonUniform: ratioTest(groupFit.logLikelihood, fullFit.logLikelihood, 1),
    deltaRSquared,
    uniformRSquared,
    coefficients: {
      intercept: coefficients[0] as number,
      match: coefficients[1] as number,
      group: coefficients[2] as number,
      interaction: coefficients[3] as number,
    },
    standardErrors: {
      intercept: errors[0] as number,
      match: errors[1] as number,
      group: errors[2] as number,
      interaction: errors[3] as number,
    },
    classification: classifyLogistic(combined.pValue, deltaRSquared, rule),
    converged:
      nullFit.converged && matchFit.converged && groupFit.converged && fullFit.converged,
    people,
    standardized: standardize,
  };
}

function standardDeviation(values: readonly number[], centre: number): number {
  let total = 0;
  for (const value of values) total += (value - centre) ** 2;
  const sd = Math.sqrt(total / values.length);
  // Every candidate on the same matching score. Scaling by zero would produce a
  // column of NaN; leaving the scale at one produces a column of zeros, which
  // the rank check in the fit will catch and report honestly.
  return sd === 0 ? 1 : sd;
}

/**
 * Classify by significance first, then by effect size.
 *
 * Same two-part shape as the ETS rule, for the same reason: a chi-square grows
 * with the sample whether or not the effect does, so on a large administration
 * significance alone would flag most of the bank. The effect size is what says
 * whether a real difference is a difference worth acting on.
 */
export function classifyLogistic(
  pValue: number,
  deltaRSquared: number,
  rule: EffectRule = EFFECT_RULES.jodoinGierl,
): DifCategory {
  if (pValue >= 0.05 || deltaRSquared < rule.negligible) return 'A';
  return deltaRSquared >= rule.large ? 'C' : 'B';
}
