/**
 * Person fit: whether a response pattern looks like it came from the model.
 *
 * Item fit asks whether an item behaves for the population the way the model
 * says it should. This asks the same question of a candidate. A pattern that
 * misses easy items and answers hard ones is improbable under any single
 * ability, and the estimate the engine reports for it is a summary of
 * something that is not there — a candidate who guessed, who was interrupted,
 * who had advance knowledge of part of the form, or who was measured on a trait
 * the form does not test. The number alone cannot distinguish those, which is
 * why the statistic flags a pattern for a human rather than correcting it.
 */

import { metricScale, type Item, type ItemParameters } from '../models/item.js';
import {
  categoryProbabilitiesOf,
  expectedScoreOf,
  isPolytomous,
  responseCategory,
  responseItem,
  type AnyItem,
  type AnyResponse,
} from '../models/mixed.js';
import { standardiseMeanSquare } from '../calibration/fit.js';

export interface PersonFit {
  /**
   * The standardised log-likelihood of the pattern.
   *
   * Standard normal when the ability is known, and negative for a pattern less
   * likely than the model expects — so the test of interest is one-sided and
   * on the low side.
   */
  readonly lz: number;
  /**
   * The same statistic corrected for the ability having been estimated from
   * these very responses.
   *
   * `null` for a pattern containing a polytomous item: the correction is
   * written for the weights of the dichotomous score equation, and applying it
   * where those weights do not hold would produce a number that looks right.
   */
  readonly lzStar: number | null;
  /** Information-weighted mean square residual. Near 1 when the pattern fits. */
  readonly infit: number;
  /** Unweighted mean square residual, which the extreme items dominate. */
  readonly outfit: number;
  /** Infit on the Wilson-Hilferty scale, where it is roughly standard normal. */
  readonly standardisedInfit: number;
  /** Outfit on the same scale. */
  readonly standardisedOutfit: number;
  /** The log-likelihood of the pattern at the ability given. */
  readonly logLikelihood: number;
  /** What the model expects that log-likelihood to be. */
  readonly expectedLogLikelihood: number;
  /** How many responses the statistics were computed from. */
  readonly responses: number;
}

/** The smallest probability the log is taken of, so a certainty cannot produce -Infinity. */
const FLOOR = 1e-12;

/**
 * The log-likelihood of one response, what the model expects it to be, and its
 * variance — all three from the same category probabilities.
 *
 * Written over categories rather than over right and wrong so that a rubric
 * item needs no second code path. For a dichotomous item the sums below have
 * two terms and reduce to the familiar `P ln P + Q ln Q` and `P Q ln(P/Q)^2`
 * exactly, which the suite asserts rather than assumes.
 */
function logLikelihoodMoments(
  item: AnyItem,
  category: number,
  theta: number,
): { readonly observed: number; readonly expected: number; readonly variance: number } {
  const probabilities = categoryProbabilitiesOf(item, theta);
  const chosen = probabilities[category];
  if (chosen === undefined) {
    throw new RangeError(
      `personFit: item "${item.id}" has ${probabilities.length} categories, ` +
        `so category ${category} is not one of them`,
    );
  }

  let expected = 0;
  let secondMoment = 0;
  for (const probability of probabilities) {
    const logged = Math.log(Math.max(probability, FLOOR));
    expected += probability * logged;
    secondMoment += probability * logged * logged;
  }
  return {
    observed: Math.log(Math.max(chosen, FLOOR)),
    expected,
    // Var[ln P_X] = E[(ln P_X)^2] - (E[ln P_X])^2, which cannot be negative
    // in exact arithmetic and can be a small negative here.
    variance: Math.max(secondMoment - expected * expected, 0),
  };
}

/** The variance of the score of one response. */
function scoreVariance(item: AnyItem, theta: number): number {
  const probabilities = categoryProbabilitiesOf(item, theta);
  let expected = 0;
  for (const [category, probability] of probabilities.entries()) expected += category * probability;
  let variance = 0;
  for (const [category, probability] of probabilities.entries()) {
    variance += probability * (category - expected) * (category - expected);
  }
  return variance;
}

/**
 * The correction for an ability estimated from the same responses.
 *
 * `lz` is standard normal when the ability is *known*. Substituting an estimate
 * made from the same pattern does not leave it alone: the estimator has already
 * moved the ability towards whatever those responses said, which absorbs part
 * of the very discrepancy the statistic is trying to measure. The result is a
 * variance below one and a test that rejects less often than it claims —
 * conservative, which is the direction that quietly does nothing.
 *
 * Writing `D(theta) = sum_i (x_i - P_i) r_i` with `r_i = ln(P_i / Q_i)`, and
 * taking the estimator to solve `sum_i w_i (x_i - P_i) = 0` with `w_i = D a_i`
 * — which the maximum likelihood estimator does — a first-order expansion in
 * `(thetaHat - theta)` gives
 *
 * ```
 * D(thetaHat) ~ sum_i (x_i - P_i) (r_i - c w_i),   c = sum PQ r w / sum PQ w^2
 * ```
 *
 * so the variance to divide by is `sum_i P_i Q_i (r_i - c w_i)^2` rather than
 * `sum_i P_i Q_i r_i^2`. The correction subtracts the part of the statistic
 * the estimator has already used, and is a projection: the corrected variance
 * is always the smaller of the two, so `lzStar` is never nearer zero than `lz`.
 */
function correctedVariance(items: readonly Item[], theta: number): number {
  let sumPQrr = 0;
  let sumPQrw = 0;
  let sumPQww = 0;
  const terms: { readonly pq: number; readonly r: number; readonly w: number }[] = [];

  for (const item of items) {
    const parameters = item.parameters as ItemParameters;
    const probabilities = categoryProbabilitiesOf(item, theta);
    const p = probabilities[1] as number;
    const q = probabilities[0] as number;
    const r = Math.log(Math.max(p, FLOOR) / Math.max(q, FLOOR));
    const w = metricScale(parameters.metric) * parameters.a;
    const pq = p * q;
    sumPQrr += pq * r * r;
    sumPQrw += pq * r * w;
    sumPQww += pq * w * w;
    terms.push({ pq, r, w });
  }

  if (!(sumPQww > 0)) return sumPQrr;
  const c = sumPQrw / sumPQww;
  let corrected = 0;
  for (const { pq, r, w } of terms) corrected += pq * (r - c * w) * (r - c * w);
  return corrected;
}

/**
 * How well a response pattern fits the model at a given ability.
 *
 * `theta` is normally the estimate the engine produced for this pattern, which
 * is what `lzStar` exists for. Pass a known ability — in a simulation, or from
 * an external measurement — and `lz` is the statistic to read.
 */
export function personFit(responses: readonly AnyResponse[], theta: number): PersonFit {
  if (responses.length === 0) {
    throw new RangeError('personFit: at least one response is required');
  }
  if (!Number.isFinite(theta)) {
    throw new RangeError(`personFit: ability must be finite, received ${theta}`);
  }

  let observed = 0;
  let expected = 0;
  let variance = 0;
  let squaredResiduals = 0;
  let weightTotal = 0;
  let outfitTotal = 0;
  let outfitVariance = 0;
  let infitVariance = 0;
  let polytomous = false;

  for (const response of responses) {
    const item = responseItem(response);
    const category = responseCategory(response);
    if (isPolytomous(item)) polytomous = true;

    const moments = logLikelihoodMoments(item, category, theta);
    observed += moments.observed;
    expected += moments.expected;
    variance += moments.variance;

    const residual = category - expectedScoreOf(item, theta);
    const weight = scoreVariance(item, theta);
    squaredResiduals += residual * residual;
    weightTotal += weight;

    // The kurtosis terms the Wilson-Hilferty standardisation needs. Computed
    // over the categories for the same reason the moments above are.
    const probabilities = categoryProbabilitiesOf(item, theta);
    const expectedScore = expectedScoreOf(item, theta);
    let fourth = 0;
    for (const [value, probability] of probabilities.entries()) {
      const deviation = value - expectedScore;
      fourth += probability * deviation ** 4;
    }
    if (weight > 0) {
      outfitTotal += (residual * residual) / weight;
      outfitVariance += fourth / (weight * weight) - 1;
      infitVariance += fourth - weight * weight;
    }
  }

  const count = responses.length;
  const infit = weightTotal > 0 ? squaredResiduals / weightTotal : Number.NaN;
  const outfit = outfitTotal / count;

  return {
    lz: variance > 0 ? (observed - expected) / Math.sqrt(variance) : Number.NaN,
    lzStar: polytomous
      ? null
      : lzStarOf(responses as readonly { item: Item; response: 0 | 1 }[], theta, observed, expected),
    infit,
    outfit,
    standardisedInfit: standardiseMeanSquare(
      infit,
      weightTotal > 0 ? infitVariance / (weightTotal * weightTotal) : 0,
    ),
    standardisedOutfit: standardiseMeanSquare(outfit, outfitVariance / (count * count)),
    logLikelihood: observed,
    expectedLogLikelihood: expected,
    responses: count,
  };
}

/** `lz` with the estimated-ability correction applied. */
function lzStarOf(
  responses: readonly { readonly item: Item }[],
  theta: number,
  observed: number,
  expected: number,
): number {
  const variance = correctedVariance(
    responses.map((response) => response.item),
    theta,
  );
  return variance > 0 ? (observed - expected) / Math.sqrt(variance) : Number.NaN;
}

/** A pattern flagged as misfitting, with the statistic that flagged it. */
export interface FlaggedPerson {
  readonly id: string;
  readonly fit: PersonFit;
  /** The statistic compared against the threshold. */
  readonly statistic: number;
}

export interface PersonScanOptions {
  /**
   * How far below zero the statistic must fall to be flagged. The default is
   * the one-sided 5% point of the standard normal.
   */
  readonly threshold?: number;
  /** Use the uncorrected `lz`. Only for an ability that was not estimated from the pattern. */
  readonly corrected?: boolean;
}

/**
 * Score every candidate in a sitting and return those whose pattern misfits.
 *
 * Returned worst-first. A scan over a whole sitting is a multiple-comparison
 * problem — at a nominal 5% a thousand well-behaved candidates produce about
 * fifty flags — so the count is a starting point for review and not a verdict
 * on anyone.
 */
export function scanPersons(
  sitting: readonly { readonly id: string; readonly responses: readonly AnyResponse[]; readonly theta: number }[],
  options: PersonScanOptions = {},
): FlaggedPerson[] {
  const threshold = options.threshold ?? -1.6448536269514722;
  const corrected = options.corrected ?? true;
  if (threshold >= 0) {
    throw new RangeError(
      `scanPersons: threshold must be negative, received ${threshold}; ` +
        'a misfitting pattern is one less likely than the model expects',
    );
  }

  const flagged: FlaggedPerson[] = [];
  for (const candidate of sitting) {
    const fit = personFit(candidate.responses, candidate.theta);
    const statistic = corrected && fit.lzStar !== null ? fit.lzStar : fit.lz;
    if (Number.isFinite(statistic) && statistic < threshold) {
      flagged.push({ id: candidate.id, fit, statistic });
    }
  }
  return flagged.sort((left, right) => left.statistic - right.statistic);
}
