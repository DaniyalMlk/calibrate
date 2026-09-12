/**
 * The Mantel-Haenszel family of differential item functioning statistics.
 *
 * Within one stratum of the matching criterion the studied item gives a
 * two-by-two table: group against right or wrong. Under the null that the item
 * is fair, the odds of a correct answer are the same in both groups at every
 * level, and the Mantel-Haenszel estimator is the constant odds ratio that best
 * describes all the tables at once.
 *
 * The reason this is the operational standard rather than something more
 * flexible is that it asks for almost nothing. No response model is fitted, no
 * ability is estimated, and nothing has to converge — the statistic is a ratio of
 * two sums over the tables. An item flagged here is flagged by arithmetic on the
 * observed counts, which is a much easier thing to defend to a testing
 * programme's reviewers than a flag that depends on a model having fitted.
 */

import { chiSquareUpperTail } from '../core/gamma.js';
import {
  informativeStrata,
  summariseStrata,
  type DifStratum,
  type StratumSummary,
} from './strata.js';

/**
 * The constant converting a log odds ratio to the ETS delta metric.
 *
 * Delta is an old item-difficulty scale with a mean of 13 and a standard
 * deviation of 4, and this factor is what puts a log odds ratio onto it. It
 * survives because two decades of flagging rules were written in its units, not
 * because the scale itself is still needed.
 */
export const DELTA_SCALE = 2.35;

/** ETS flagging categories, from negligible through to large. */
export type DifCategory = 'A' | 'B' | 'C';

/** Which group an item favours, or neither if the effect is nil. */
export type Favours = 'reference' | 'focal' | 'neither';

export interface MantelHaenszelResult {
  /** The common odds ratio. Above 1 favours the reference group. */
  readonly oddsRatio: number;
  readonly logOddsRatio: number;
  /** Robins-Breslow-Greenland standard error of the log odds ratio. */
  readonly logOddsStandardError: number;
  /** The ETS delta effect size. Negative means harder for the focal group. */
  readonly delta: number;
  readonly deltaStandardError: number;
  /** The continuity-corrected chi-square, on one degree of freedom. */
  readonly chiSquare: number;
  readonly pValue: number;
  readonly classification: DifCategory;
  readonly favours: Favours;
  /**
   * False when one of the two pooled products vanished, leaving the odds ratio
   * unbounded. The chi-square is still valid; the effect size is not finite.
   */
  readonly estimable: boolean;
  readonly strata: StratumSummary;
}

/** One-sided critical value at the five percent level, for the delta test. */
const NORMAL_ONE_SIDED = 1.6448536269514722;
/** The ETS boundary between negligible and moderate, in delta units. */
export const NEGLIGIBLE_DELTA = 1;
/** The ETS boundary between moderate and large, in delta units. */
export const LARGE_DELTA = 1.5;

function requireDichotomous(strata: readonly DifStratum[], caller: string): void {
  for (const stratum of strata) {
    if (stratum.reference.length !== 2) {
      throw new RangeError(
        `${caller}: expects a dichotomous item, received ${stratum.reference.length} categories; ` +
          'use generalizedMantel for a rubric-scored item',
      );
    }
  }
}

/**
 * The Mantel-Haenszel statistic over a stratified sample.
 *
 * Uninformative strata are dropped first. They contribute nothing to either sum,
 * so this changes no result — it only makes the reported stratum counts mean
 * what they say.
 */
export function mantelHaenszel(strata: readonly DifStratum[]): MantelHaenszelResult {
  requireDichotomous(strata, 'mantelHaenszel');
  const summary = summariseStrata(strata);
  const usable = informativeStrata(strata);
  if (usable.length === 0) {
    throw new RangeError(
      'mantelHaenszel: no stratum holds both groups with a mix of outcomes, ' +
        'so no comparison is possible at equal matching score',
    );
  }

  // The pooled products: numerator counts reference-right against focal-wrong,
  // the denominator the opposite pairing, each weighted by the stratum size.
  let numerator = 0;
  let denominator = 0;
  let observed = 0;
  let expected = 0;
  let variance = 0;
  // Robins-Breslow-Greenland accumulators.
  let pr = 0;
  let psQr = 0;
  let qs = 0;

  for (const stratum of usable) {
    const right = stratum.reference[1] as number;
    const wrong = stratum.reference[0] as number;
    const focalRight = stratum.focal[1] as number;
    const focalWrong = stratum.focal[0] as number;
    const total = stratum.total;

    const r = (right * focalWrong) / total;
    const s = (wrong * focalRight) / total;
    numerator += r;
    denominator += s;

    const rightTotal = right + focalRight;
    const wrongTotal = wrong + focalWrong;
    observed += right;
    expected += (stratum.referenceTotal * rightTotal) / total;
    if (total > 1) {
      variance +=
        (stratum.referenceTotal * stratum.focalTotal * rightTotal * wrongTotal) /
        (total * total * (total - 1));
    }

    const p = (right + focalWrong) / total;
    const q = (wrong + focalRight) / total;
    pr += p * r;
    psQr += p * s + q * r;
    qs += q * s;
  }

  // Holland and Thayer's continuity correction, floored at zero: with a
  // deviation under half a count the correction would otherwise overshoot and
  // square back into a positive statistic.
  const deviation = Math.max(0, Math.abs(observed - expected) - 0.5);
  const chiSquare = variance > 0 ? (deviation * deviation) / variance : 0;
  const pValue = chiSquareUpperTail(chiSquare, 1);

  const estimable = numerator > 0 && denominator > 0;
  const oddsRatio = denominator === 0 ? Number.POSITIVE_INFINITY : numerator / denominator;
  const logOddsRatio = Math.log(oddsRatio);
  const logOddsStandardError = estimable
    ? Math.sqrt(
        pr / (2 * numerator * numerator) +
          psQr / (2 * numerator * denominator) +
          qs / (2 * denominator * denominator),
      )
    : Number.POSITIVE_INFINITY;

  const delta = -DELTA_SCALE * logOddsRatio;
  const deltaStandardError = DELTA_SCALE * logOddsStandardError;

  return {
    oddsRatio,
    logOddsRatio,
    logOddsStandardError,
    delta,
    deltaStandardError,
    chiSquare,
    pValue,
    classification: classifyDelta(delta, deltaStandardError, pValue, estimable),
    favours: favoursOf(delta),
    estimable,
    strata: summary,
  };
}

/**
 * The ETS A/B/C rule.
 *
 * Read in order. An item is negligible if the null of no DIF survives at all, or
 * if the effect is under a delta point however precisely it is measured. It is
 * large only if the effect clears 1.5 delta points *and* the data can rule out
 * its being as small as 1 — the second half is what stops a large flag resting
 * on a wide confidence interval that happens to have a distant point estimate.
 * Everything in between is moderate.
 *
 * When the odds ratio is unbounded the effect size cannot be compared to a
 * boundary, so the rule falls back to the chi-square alone: every matched
 * comparison in the sample pointed the same way, which is as strong as the data
 * can express.
 */
export function classifyDelta(
  delta: number,
  deltaStandardError: number,
  pValue: number,
  estimable = true,
): DifCategory {
  if (!estimable) return pValue < 0.05 ? 'C' : 'A';
  const magnitude = Math.abs(delta);
  if (pValue >= 0.05 || magnitude < NEGLIGIBLE_DELTA) return 'A';
  const exceedsBoundary = magnitude - NORMAL_ONE_SIDED * deltaStandardError > NEGLIGIBLE_DELTA;
  return magnitude >= LARGE_DELTA && exceedsBoundary ? 'C' : 'B';
}

function favoursOf(delta: number): Favours {
  if (delta === 0 || Number.isNaN(delta)) return 'neither';
  return delta > 0 ? 'focal' : 'reference';
}

export interface StandardizedDifferenceResult {
  /** The standardized proportion difference, focal minus reference. */
  readonly value: number;
  /** Weighted mean proportion correct in the focal group. */
  readonly focalProportion: number;
  /** The same weights applied to the reference proportions. */
  readonly referenceProportion: number;
  readonly classification: DifCategory;
  readonly favours: Favours;
  readonly strata: StratumSummary;
}

/** The ETS boundary between negligible and moderate, in proportion units. */
export const NEGLIGIBLE_PROPORTION = 0.05;
/** The boundary between moderate and large, in proportion units. */
export const LARGE_PROPORTION = 0.1;

/**
 * The standardized proportion difference.
 *
 * Within each stratum, the difference between the two groups' proportions
 * correct; across strata, a weighted average of those differences using the
 * focal group's own distribution as weights.
 *
 * It answers a different question from the odds ratio, and is worth reporting
 * beside it rather than instead of it. Mantel-Haenszel assumes one odds ratio
 * describes every stratum and reports that constant; this assumes nothing about
 * the shape and reports the average difference a focal candidate actually
 * experiences. When the two disagree, the item's effect is not constant across
 * the ability range, and the disagreement is the finding.
 */
export function standardizedDifference(
  strata: readonly DifStratum[],
): StandardizedDifferenceResult {
  requireDichotomous(strata, 'standardizedDifference');
  const usable = informativeStrata(strata);
  if (usable.length === 0) {
    throw new RangeError('standardizedDifference: no stratum holds both groups');
  }

  let weight = 0;
  let focalSum = 0;
  let referenceSum = 0;
  for (const stratum of usable) {
    const w = stratum.focalTotal;
    weight += w;
    focalSum += w * ((stratum.focal[1] as number) / stratum.focalTotal);
    referenceSum += w * ((stratum.reference[1] as number) / stratum.referenceTotal);
  }

  const focalProportion = focalSum / weight;
  const referenceProportion = referenceSum / weight;
  const value = focalProportion - referenceProportion;
  const magnitude = Math.abs(value);

  return {
    value,
    focalProportion,
    referenceProportion,
    classification:
      magnitude < NEGLIGIBLE_PROPORTION ? 'A' : magnitude < LARGE_PROPORTION ? 'B' : 'C',
    favours: value === 0 ? 'neither' : value > 0 ? 'focal' : 'reference',
    strata: summariseStrata(strata),
  };
}
