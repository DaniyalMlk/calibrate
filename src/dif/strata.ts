/**
 * Stratification of a sample by a matching criterion.
 *
 * Every matched-group method for differential item functioning rests on one
 * idea: compare the two groups only among candidates of equal ability. Ability
 * is not observed, so a matching criterion stands in for it — almost always the
 * total score on the form — and the comparison is made within each level of
 * that criterion and then pooled across levels.
 *
 * What that buys is the separation between impact and bias. Two groups can
 * differ on an item because one group is more able, which is impact and is not
 * the item's fault; or because the item is harder for one group at equal
 * ability, which is what DIF means. Stratifying first is what tells them apart,
 * and an unstratified comparison of proportions cannot.
 */

import { MISSING, type ResponseMatrix } from '../calibration/matrix.js';

/**
 * Which of the two groups a candidate belongs to.
 *
 * The naming is not decoration. Every statistic below is directional, and the
 * convention throughout is that a negative effect means the item is harder for
 * the focal group than its matched reference counterparts — the direction that
 * matters, because the focal group is by construction the one the analysis is
 * being run to protect.
 */
export type Group = 'reference' | 'focal';

export interface DifObservation {
  readonly group: Group;
  /** The matching criterion: a total score, or any other ability proxy. */
  readonly match: number;
  /** Score on the studied item: 0 or 1 if dichotomous, 0..maxScore if graded. */
  readonly score: number;
}

/**
 * One level of the matching criterion, with the studied item's score
 * distribution in each group.
 *
 * The counts are indexed by category, so `reference[2]` is the number of
 * reference-group candidates at this level who scored 2. A dichotomous item
 * uses only indices 0 and 1, which makes this the familiar two-by-two table
 * without needing a separate type for it.
 */
export interface DifStratum {
  /** Lowest matching score in the stratum. Equal to `matchHigh` unless merged. */
  readonly matchLow: number;
  /** Highest matching score in the stratum. */
  readonly matchHigh: number;
  /** Count per category in the reference group. Length is `maxScore + 1`. */
  readonly reference: readonly number[];
  /** Count per category in the focal group. */
  readonly focal: readonly number[];
  readonly referenceTotal: number;
  readonly focalTotal: number;
  /** Everyone at this level, both groups. */
  readonly total: number;
}

export interface StratifyOptions {
  /** Highest score the studied item can earn. Default 1. */
  readonly maxScore?: number;
  /**
   * Merge adjacent levels upward until each holds at least this many people.
   *
   * Thin strata are not wrong, they are just uninformative: a level holding two
   * reference candidates and one focal candidate contributes a variance so
   * small that it cannot move the pooled statistic, while still costing a degree
   * of freedom in anything that counts levels. Default 1, which merges nothing.
   */
  readonly minimumStratum?: number;
}

/**
 * Group observations into strata of the matching criterion.
 *
 * Levels present in the data become strata, in ascending order; levels nobody
 * reached do not appear at all, which is the right behaviour for a pooled
 * statistic that sums over strata and would add zero for each of them anyway.
 */
export function stratify(
  observations: readonly DifObservation[],
  options: StratifyOptions = {},
): DifStratum[] {
  const maxScore = options.maxScore ?? 1;
  const minimumStratum = options.minimumStratum ?? 1;
  if (!Number.isInteger(maxScore) || maxScore < 1) {
    throw new RangeError(`stratify: maxScore must be a positive integer, received ${maxScore}`);
  }
  if (!Number.isInteger(minimumStratum) || minimumStratum < 1) {
    throw new RangeError(
      `stratify: minimumStratum must be a positive integer, received ${minimumStratum}`,
    );
  }
  if (observations.length === 0) throw new RangeError('stratify: no observations');

  const buckets = new Map<number, { reference: number[]; focal: number[] }>();
  for (const [index, observation] of observations.entries()) {
    const { match, score, group } = observation;
    if (!Number.isFinite(match)) {
      throw new RangeError(`stratify: observation ${index} has a non-finite matching score`);
    }
    if (!Number.isInteger(score) || score < 0 || score > maxScore) {
      throw new RangeError(
        `stratify: observation ${index} scored ${score}, outside 0..${maxScore}`,
      );
    }
    let bucket = buckets.get(match);
    if (bucket === undefined) {
      bucket = {
        reference: new Array<number>(maxScore + 1).fill(0),
        focal: new Array<number>(maxScore + 1).fill(0),
      };
      buckets.set(match, bucket);
    }
    const counts = group === 'reference' ? bucket.reference : bucket.focal;
    counts[score] = (counts[score] as number) + 1;
  }

  const levels = [...buckets.keys()].sort((a, b) => a - b);
  const strata = levels.map((match) => {
    const bucket = buckets.get(match) as { reference: number[]; focal: number[] };
    return makeStratum(match, match, bucket.reference, bucket.focal);
  });

  return minimumStratum > 1 ? mergeThinStrata(strata, minimumStratum) : strata;
}

function makeStratum(
  matchLow: number,
  matchHigh: number,
  reference: readonly number[],
  focal: readonly number[],
): DifStratum {
  const referenceTotal = reference.reduce((a, b) => a + b, 0);
  const focalTotal = focal.reduce((a, b) => a + b, 0);
  return {
    matchLow,
    matchHigh,
    reference: Object.freeze([...reference]),
    focal: Object.freeze([...focal]),
    referenceTotal,
    focalTotal,
    total: referenceTotal + focalTotal,
  };
}

/**
 * Merge adjacent strata upward until each holds `minimum` people.
 *
 * Merging runs from the bottom of the scale and carries any shortfall into the
 * next level. A shortfall left over at the top is merged back down into the
 * previous stratum rather than kept, since a stratum at the top of the scale is
 * the one most likely to be all-correct and contribute nothing regardless.
 */
export function mergeThinStrata(strata: readonly DifStratum[], minimum: number): DifStratum[] {
  if (strata.length === 0) return [];
  const merged: DifStratum[] = [];
  let carry: DifStratum | null = null;

  for (const stratum of strata) {
    // Annotated because the assignment back into `carry` at the foot of the
    // loop would otherwise make the inference of this binding circular.
    const combined: DifStratum = carry === null ? stratum : combine(carry, stratum);
    if (combined.total >= minimum) {
      merged.push(combined);
      carry = null;
    } else {
      carry = combined;
    }
  }

  if (carry !== null) {
    const last = merged.pop();
    merged.push(last === undefined ? carry : combine(last, carry));
  }
  return merged;
}

function combine(lower: DifStratum, upper: DifStratum): DifStratum {
  const reference = lower.reference.map((count, index) => count + (upper.reference[index] as number));
  const focal = lower.focal.map((count, index) => count + (upper.focal[index] as number));
  return makeStratum(
    Math.min(lower.matchLow, upper.matchLow),
    Math.max(lower.matchHigh, upper.matchHigh),
    reference,
    focal,
  );
}

/**
 * Whether a stratum can move a pooled statistic at all.
 *
 * A stratum is uninformative when either group is absent from it, or when
 * everyone in it scored the same category. In both cases the within-stratum
 * variance is zero: there is no comparison to make, not a comparison that came
 * out null. Dropping them is not a robustness hack — including them adds
 * exactly zero to both the numerator and the denominator of every statistic
 * here, and the only thing they would change is a count of degrees of freedom.
 */
export function isInformative(stratum: DifStratum): boolean {
  if (stratum.referenceTotal === 0 || stratum.focalTotal === 0) return false;
  let occupied = 0;
  for (const [index, count] of stratum.reference.entries()) {
    if (count + (stratum.focal[index] as number) > 0) occupied += 1;
    if (occupied > 1) return true;
  }
  return false;
}

/** The informative strata, in ascending order of the matching criterion. */
export function informativeStrata(strata: readonly DifStratum[]): DifStratum[] {
  return strata.filter(isInformative);
}

export interface StratumSummary {
  readonly strata: number;
  readonly informative: number;
  readonly reference: number;
  readonly focal: number;
  /** People sitting in strata that cannot contribute to a pooled statistic. */
  readonly discarded: number;
}

/**
 * Counts describing a stratification, for reporting alongside a result.
 *
 * Worth printing next to any DIF statistic. A flag resting on three informative
 * strata out of forty is a different claim from the same flag resting on
 * thirty-eight, and the statistic itself does not distinguish them.
 */
export function summariseStrata(strata: readonly DifStratum[]): StratumSummary {
  let reference = 0;
  let focal = 0;
  let informative = 0;
  let discarded = 0;
  for (const stratum of strata) {
    reference += stratum.referenceTotal;
    focal += stratum.focalTotal;
    if (isInformative(stratum)) informative += 1;
    else discarded += stratum.total;
  }
  return { strata: strata.length, informative, reference, focal, discarded };
}

export interface MatchingOptions {
  /**
   * Item indices forming the matching criterion. Defaults to every item.
   *
   * Restricting this to a set of items believed free of DIF is what purification
   * does, and is the reason the option exists.
   */
  readonly anchor?: readonly number[];
  /**
   * Count the studied item in its own matching score. Default true.
   *
   * Counter-intuitive, and correct. Leaving the item out makes the criterion
   * and the studied response independent, which sounds like the cleaner choice
   * and is in fact the source of a well-known bias: the remaining items define a
   * slightly different construct, and every item is then compared against a
   * criterion it is absent from. Including it is the standard, and it costs only
   * a mild conservatism.
   */
  readonly includeStudied?: boolean;
}

export interface MatchedSample {
  readonly observations: DifObservation[];
  /** Candidates dropped for an incomplete matching score or studied response. */
  readonly dropped: number;
  /** Highest matching score attainable by anyone in the sample. */
  readonly maxMatch: number;
}

/**
 * Build the matched sample for one studied item out of a response matrix.
 *
 * A candidate who did not answer the studied item, or who is missing any anchor
 * item, is dropped rather than scored as wrong. Scoring a skipped anchor item as
 * wrong would move that candidate down the matching scale, and since candidates
 * do not skip at random, it would move the groups down by different amounts and
 * manufacture exactly the difference being tested for.
 */
export function matchedSample(
  matrix: ResponseMatrix,
  groups: readonly Group[],
  studied: number,
  options: MatchingOptions = {},
): MatchedSample {
  if (groups.length !== matrix.personCount) {
    throw new RangeError(
      `matchedSample: ${groups.length} group labels for ${matrix.personCount} people`,
    );
  }
  if (!Number.isInteger(studied) || studied < 0 || studied >= matrix.itemCount) {
    throw new RangeError(`matchedSample: no item at index ${studied}`);
  }

  const includeStudied = options.includeStudied ?? true;
  const anchor = options.anchor ?? Array.from({ length: matrix.itemCount }, (_, i) => i);
  for (const index of anchor) {
    if (!Number.isInteger(index) || index < 0 || index >= matrix.itemCount) {
      throw new RangeError(`matchedSample: anchor references no item at index ${index}`);
    }
  }
  const criterion = anchor.filter((index) => includeStudied || index !== studied);
  if (criterion.length === 0) {
    throw new RangeError(
      'matchedSample: the matching criterion is empty; widen the anchor or include the studied item',
    );
  }

  const observations: DifObservation[] = [];
  let dropped = 0;
  for (let person = 0; person < matrix.personCount; person += 1) {
    const response = matrix.at(person, studied);
    if (response === MISSING) {
      dropped += 1;
      continue;
    }
    let match = 0;
    let complete = true;
    for (const index of criterion) {
      const cell = matrix.at(person, index);
      if (cell === MISSING) {
        complete = false;
        break;
      }
      match += cell;
    }
    if (!complete) {
      dropped += 1;
      continue;
    }
    observations.push({ group: groups[person] as Group, match, score: response });
  }

  if (observations.length === 0) {
    throw new RangeError(
      `matchedSample: every candidate was dropped for item ${matrix.itemIds[studied]}`,
    );
  }
  return { observations, dropped, maxMatch: criterion.length };
}
