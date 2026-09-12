/**
 * Differential item functioning for rubric-scored items.
 *
 * A two-by-two table generalises to a two-by-many one as soon as the studied
 * item has more than two outcomes, and the odds ratio stops being the natural
 * summary: there is no single pair of cells to take the odds of. What survives
 * is the score. Within a stratum, under the null, the focal group's total on the
 * item is a draw without replacement from the scores present at that level, and
 * its mean and variance follow from the hypergeometric distribution with no
 * further assumptions.
 *
 * That is Mantel's statistic, and it specialises exactly — not approximately —
 * to the uncorrected Mantel-Haenszel chi-square when the item happens to be
 * dichotomous. The test below asserts that equality, which is the best available
 * check that the general form is right: it has to reproduce a statistic computed
 * by an entirely different route.
 */

import { chiSquareUpperTail } from '../core/gamma.js';
import { requireFinite } from '../core/numeric.js';
import type { DifCategory, Favours } from './mantel.js';
import {
  informativeStrata,
  summariseStrata,
  type DifStratum,
  type StratumSummary,
} from './strata.js';

export interface GradedOptions {
  /**
   * The value of each category. Defaults to 0, 1, ... up to the top category.
   *
   * Worth exposing because a rubric's levels are not always worth their index.
   * A four-level rubric scored 0, 1, 3, 4 is testing something different from
   * one scored 0, 1, 2, 3, and the statistic should be told which it is rather
   * than assuming the levels are evenly spaced.
   */
  readonly categoryScores?: readonly number[];
}

function scoresFor(strata: readonly DifStratum[], options: GradedOptions, caller: string): number[] {
  const width = strata[0]?.reference.length ?? 0;
  if (width < 2) throw new RangeError(`${caller}: an item needs at least two categories`);
  for (const stratum of strata) {
    if (stratum.reference.length !== width) {
      throw new RangeError(`${caller}: strata disagree about how many categories the item has`);
    }
  }
  if (options.categoryScores === undefined) {
    return Array.from({ length: width }, (_, index) => index);
  }
  if (options.categoryScores.length !== width) {
    throw new RangeError(
      `${caller}: ${options.categoryScores.length} category scores for ${width} categories`,
    );
  }
  return options.categoryScores.map((value, index) =>
    requireFinite(value, `${caller}: category score ${index}`),
  );
}

export interface GeneralizedMantelResult {
  /** Mantel's chi-square, on one degree of freedom. No continuity correction. */
  readonly chiSquare: number;
  readonly pValue: number;
  /** The focal group's observed total on the item, summed over strata. */
  readonly observed: number;
  /** What that total would be under the null, summed over strata. */
  readonly expected: number;
  /** The hypergeometric variance of the observed total, summed over strata. */
  readonly variance: number;
  readonly favours: Favours;
  readonly strata: StratumSummary;
}

/**
 * Mantel's statistic for an item with ordered categories.
 *
 * The comparison is made on the focal group's side rather than the reference
 * group's, which is arbitrary — the two deviations are equal and opposite, so
 * squaring gives the same number either way — but it makes the sign of the
 * deviation directly readable as the direction of the effect.
 *
 * There is deliberately no continuity correction. Mantel's statistic is
 * conventionally reported without one, and applying one here would break the
 * exact identity with the uncorrected two-by-two case that makes the
 * implementation checkable.
 */
export function generalizedMantel(
  strata: readonly DifStratum[],
  options: GradedOptions = {},
): GeneralizedMantelResult {
  const usable = informativeStrata(strata);
  if (usable.length === 0) {
    throw new RangeError(
      'generalizedMantel: no stratum holds both groups with a mix of outcomes',
    );
  }
  const scores = scoresFor(usable, options, 'generalizedMantel');

  let observed = 0;
  let expected = 0;
  let variance = 0;

  for (const stratum of usable) {
    const total = stratum.total;
    let focalTotal = 0;
    let scoreSum = 0;
    let squareSum = 0;
    for (const [category, value] of scores.entries()) {
      const focalCount = stratum.focal[category] as number;
      const count = focalCount + (stratum.reference[category] as number);
      focalTotal += value * focalCount;
      scoreSum += value * count;
      squareSum += value * value * count;
    }

    observed += focalTotal;
    expected += (stratum.focalTotal * scoreSum) / total;
    if (total > 1) {
      // The population variance of the scores present at this level, times the
      // finite-population correction for a draw of the focal group's size.
      const spread = squareSum / total - (scoreSum / total) ** 2;
      variance += ((stratum.focalTotal * stratum.referenceTotal) / (total - 1)) * spread;
    }
  }

  const deviation = observed - expected;
  const chiSquare = variance > 0 ? (deviation * deviation) / variance : 0;

  return {
    chiSquare,
    pValue: chiSquareUpperTail(chiSquare, 1),
    observed,
    expected,
    variance,
    favours: deviation === 0 ? 'neither' : deviation > 0 ? 'focal' : 'reference',
    strata: summariseStrata(strata),
  };
}

export interface StandardizedMeanDifferenceResult {
  /** Focal mean minus reference mean, in score points. Negative is against focal. */
  readonly value: number;
  readonly focalMean: number;
  readonly referenceMean: number;
  /** Standard deviation of the item score over both groups together. */
  readonly scoreDeviation: number;
  /** The difference in standard deviation units. */
  readonly standardized: number;
  readonly classification: DifCategory;
  readonly favours: Favours;
  readonly strata: StratumSummary;
}

/** The polytomous boundary between negligible and moderate, in SD units. */
export const NEGLIGIBLE_STANDARDIZED = 0.17;
/** The polytomous boundary between moderate and large, in SD units. */
export const LARGE_STANDARDIZED = 0.25;

/**
 * The polytomous analogue of the ETS A/B/C rule.
 *
 * Same shape as the dichotomous rule and the same reasoning behind it: a
 * significance test decides whether there is an effect at all, and an effect
 * size decides whether it is worth acting on. Only the boundaries differ,
 * because they are expressed in standard deviations of the item score rather
 * than in delta points.
 */
export function classifyGraded(pValue: number, standardized: number): DifCategory {
  const magnitude = Math.abs(standardized);
  if (pValue >= 0.05 || magnitude <= NEGLIGIBLE_STANDARDIZED) return 'A';
  return magnitude >= LARGE_STANDARDIZED ? 'C' : 'B';
}

/**
 * The standardized mean difference on a rubric-scored item.
 *
 * Each stratum contributes the gap between the two groups' mean scores at that
 * level, weighted by the focal group's presence there; the result is divided by
 * the item's own score standard deviation so that rubrics of different lengths
 * can be compared.
 *
 * Dividing by the spread is what makes the number portable, and it is also the
 * step that can mislead. An item whose scores barely vary will show a large
 * standardized difference from a small absolute one, so the unstandardized
 * value and the deviation it was divided by are both reported rather than
 * folded away.
 */
export function standardizedMeanDifference(
  strata: readonly DifStratum[],
  options: GradedOptions = {},
): StandardizedMeanDifferenceResult {
  const usable = informativeStrata(strata);
  if (usable.length === 0) {
    throw new RangeError('standardizedMeanDifference: no stratum holds both groups');
  }
  const scores = scoresFor(usable, options, 'standardizedMeanDifference');

  let weight = 0;
  let focalSum = 0;
  let referenceSum = 0;
  let people = 0;
  let scoreSum = 0;
  let squareSum = 0;

  for (const stratum of usable) {
    let focalScore = 0;
    let referenceScore = 0;
    for (const [category, value] of scores.entries()) {
      const focalCount = stratum.focal[category] as number;
      const referenceCount = stratum.reference[category] as number;
      focalScore += value * focalCount;
      referenceScore += value * referenceCount;
      const count = focalCount + referenceCount;
      scoreSum += value * count;
      squareSum += value * value * count;
    }
    people += stratum.total;

    const w = stratum.focalTotal;
    weight += w;
    focalSum += w * (focalScore / stratum.focalTotal);
    referenceSum += w * (referenceScore / stratum.referenceTotal);
  }

  const focalMean = focalSum / weight;
  const referenceMean = referenceSum / weight;
  const value = focalMean - referenceMean;
  const scoreDeviation = Math.sqrt(Math.max(0, squareSum / people - (scoreSum / people) ** 2));
  const standardized = scoreDeviation > 0 ? value / scoreDeviation : 0;
  const magnitude = Math.abs(standardized);

  return {
    value,
    focalMean,
    referenceMean,
    scoreDeviation,
    standardized,
    classification:
      magnitude <= NEGLIGIBLE_STANDARDIZED ? 'A' : magnitude < LARGE_STANDARDIZED ? 'B' : 'C',
    favours: value === 0 ? 'neither' : value > 0 ? 'focal' : 'reference',
    strata: summariseStrata(strata),
  };
}
