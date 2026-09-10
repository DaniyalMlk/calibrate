import { requireFinite } from '../core/numeric.js';
import { type Item, type ItemParameters, type Metric } from './item.js';
import {
  categoryCount,
  categoryProbabilities,
  expectedCategoryScore,
  maximumScore,
  polytomousInformation,
  validatePolytomousParameters,
  type PolytomousParameters,
} from './polytomous.js';
import { itemInformation, probabilityCorrect } from './response.js';

/**
 * An item scored into three or more ordered categories.
 *
 * Deliberately the same shape as the dichotomous `Item` — identity, parameters,
 * optional blueprint metadata — so that a pool, a blueprint or an exposure
 * table can hold both without special-casing either.
 */
export interface PolytomousItem {
  readonly id: string;
  readonly parameters: PolytomousParameters;
  /** Optional blueprint category, used by content-balanced selection. */
  readonly domain?: string;
  /** Free-form labels, e.g. skill tags. */
  readonly tags?: readonly string[];
}

/**
 * Any item the engine can administer, of either format.
 *
 * The two are distinguished by their parameters rather than by a tag field:
 * a dichotomous item has a difficulty `b`, a polytomous one has `thresholds`.
 * Discriminating on the shape that actually differs means an item cannot be
 * built with a label that contradicts its own parameters.
 */
export type AnyItem = Item | PolytomousItem;

/**
 * A response to an item of either format, scored as a category index.
 *
 * A dichotomous item takes 0 or 1, which is exactly its category index, so the
 * dichotomous `ScoredResponse` is the two-category case of this and no
 * conversion is needed in either direction.
 */
export interface CategoryResponse {
  readonly item: AnyItem;
  /** The category scored, `0..maximumScoreOf(item)`. */
  readonly category: number;
}

/** True when the item is scored into more than two ordered categories. */
export function isPolytomous(item: AnyItem): item is PolytomousItem {
  return 'thresholds' in item.parameters;
}

/** Build a validated polytomous item record. */
export function makePolytomousItem(
  id: string,
  parameters: PolytomousParameters,
  meta: { domain?: string; tags?: readonly string[] } = {},
): PolytomousItem {
  if (typeof id !== 'string' || id.length === 0) {
    throw new RangeError('item id must be a non-empty string');
  }
  const item: PolytomousItem = {
    id,
    parameters: validatePolytomousParameters(parameters),
    ...(meta.domain === undefined ? {} : { domain: meta.domain }),
    ...(meta.tags === undefined ? {} : { tags: Object.freeze([...meta.tags]) }),
  };
  return Object.freeze(item);
}

/** The metric an item of either format was calibrated on. */
export function metricOf(item: AnyItem): Metric {
  return item.parameters.metric;
}

/**
 * Number of ordered categories the item scores into.
 *
 * Two for any dichotomous item, one more than the threshold count otherwise.
 */
export function categoryCountOf(item: AnyItem): number {
  return isPolytomous(item) ? categoryCount(item.parameters) : 2;
}

/** The highest category score attainable on the item. */
export function maximumScoreOf(item: AnyItem): number {
  return isPolytomous(item) ? maximumScore(item.parameters) : 1;
}

/**
 * The full distribution over an item's categories at a given ability.
 *
 * For a dichotomous item this is `[1 - P, P]`, which is the two-category case
 * of the polytomous distribution rather than an approximation of it.
 */
export function categoryProbabilitiesOf(item: AnyItem, theta: number): number[] {
  if (isPolytomous(item)) return categoryProbabilities(item.parameters, theta);
  const p = probabilityCorrect(item.parameters as ItemParameters, theta);
  return [1 - p, p];
}

/** Probability of scoring in exactly category `k` on an item of either format. */
export function categoryProbabilityOf(item: AnyItem, theta: number, k: number): number {
  const count = categoryCountOf(item);
  if (!Number.isInteger(k) || k < 0 || k >= count) {
    throw new RangeError(
      `category must be an integer in [0, ${count - 1}] for item "${item.id}", received ${k}`,
    );
  }
  return categoryProbabilitiesOf(item, theta)[k] as number;
}

/** Fisher information contributed by an item of either format. */
export function informationOf(item: AnyItem, theta: number): number {
  return isPolytomous(item)
    ? polytomousInformation(item.parameters, theta)
    : itemInformation(item.parameters as ItemParameters, theta);
}

/**
 * Expected score on an item of either format.
 *
 * For a dichotomous item the expected score *is* the probability of a correct
 * answer, so the two formats share one definition and a mixed form can be summed
 * without a branch at the call site.
 */
export function expectedScoreOf(item: AnyItem, theta: number): number {
  return isPolytomous(item)
    ? expectedCategoryScore(item.parameters, theta)
    : probabilityCorrect(item.parameters as ItemParameters, theta);
}

/**
 * Reject a category index that the item cannot produce.
 *
 * Called wherever a scored response enters the engine. A category of 3 on a
 * two-category item is not a value to be clamped into range — it means the
 * response was scored against a different item than the one it is paired with,
 * and quietly treating it as 1 would put a wrong answer in the transcript with
 * no trace of where it came from.
 */
export function requireValidCategory(item: AnyItem, category: number): number {
  const top = maximumScoreOf(item);
  if (!Number.isInteger(category) || category < 0 || category > top) {
    throw new RangeError(
      `response to item "${item.id}" must be an integer category in [0, ${top}], ` +
        `received ${String(category)}`,
    );
  }
  return category;
}

/** Sum of item information over a mixed-format set — the test information function. */
export function testInformationOf(items: readonly AnyItem[], theta: number): number {
  requireFinite(theta, 'ability (theta)');
  let total = 0;
  for (const item of items) total += informationOf(item, theta);
  return total;
}

/**
 * The test characteristic curve: expected total score on a form at a given
 * ability, summed over items of either format.
 *
 * This is the bridge between the ability metric and a reported raw score, and
 * it is the function two forms have to agree on for their scores to mean the
 * same thing. It rises monotonically from zero to `maximumTestScore`.
 */
export function testCharacteristicCurve(items: readonly AnyItem[], theta: number): number {
  requireFinite(theta, 'ability (theta)');
  let total = 0;
  for (const item of items) total += expectedScoreOf(item, theta);
  return total;
}

/** The highest total score attainable on a form — the sum of item maxima. */
export function maximumTestScore(items: readonly AnyItem[]): number {
  let total = 0;
  for (const item of items) total += maximumScoreOf(item);
  return total;
}

/**
 * Count the items of each format in a form.
 *
 * Reported because the mix is a design fact worth surfacing: a form that is
 * nominally mixed but carries two polytomous items among sixty dichotomous ones
 * behaves, for every practical purpose, like a dichotomous form.
 */
export function formatCounts(items: readonly AnyItem[]): {
  dichotomous: number;
  polytomous: number;
  maximumScore: number;
} {
  let dichotomous = 0;
  let polytomous = 0;
  for (const item of items) {
    if (isPolytomous(item)) polytomous += 1;
    else dichotomous += 1;
  }
  return { dichotomous, polytomous, maximumScore: maximumTestScore(items) };
}
