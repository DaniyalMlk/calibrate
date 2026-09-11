import { requireFinite } from '../core/numeric.js';
import { type Item, type ItemParameters, type Metric } from './item.js';
import {
  categoryCount,
  categoryDerivatives,
  categoryProbabilities,
  categorySecondDerivatives,
  expectedCategoryScore,
  maximumScore,
  polytomousInformation,
  validatePolytomousParameters,
  type PolytomousParameters,
} from './polytomous.js';
import {
  itemInformation,
  probabilityCorrect,
  responseDerivative,
  responseSecondDerivative,
  type ScoredResponse,
} from './response.js';

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

/**
 * A response of either shape.
 *
 * The dichotomous `ScoredResponse` carries a `response` of 0 or 1; the
 * polytomous `CategoryResponse` carries a `category`. Those are the same number
 * under two names — a dichotomous item's category index *is* its score — so the
 * engine accepts either and normalises at the boundary rather than making
 * callers convert.
 */
export type AnyResponse = ScoredResponse | CategoryResponse;

/** The item a response of either shape was given to. */
export function responseItem(response: AnyResponse): AnyItem {
  return response.item;
}

/**
 * The category a response of either shape scored.
 *
 * This is the whole of the adapter between the two formats: everything
 * downstream — likelihood, score, information, boundedness — is written once
 * against a category index.
 */
export function responseCategory(response: AnyResponse): number {
  return 'category' in response ? response.category : response.response;
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

/**
 * First derivatives of every category probability with respect to ability.
 *
 * For a dichotomous item this is `[-P', P']`: the two categories are the two
 * sides of one curve, so whatever probability the correct category gains, the
 * incorrect one loses. That is the two-category case of the polytomous identity
 * that the derivatives sum to zero.
 */
export function categoryDerivativesOf(item: AnyItem, theta: number): number[] {
  if (isPolytomous(item)) return categoryDerivatives(item.parameters, theta);
  const slope = responseDerivative(item.parameters as ItemParameters, theta);
  return [-slope, slope];
}

/**
 * Second derivatives of every category probability with respect to ability.
 *
 * `[-P'', P'']` for a dichotomous item, for the same reason.
 */
export function categorySecondDerivativesOf(item: AnyItem, theta: number): number[] {
  if (isPolytomous(item)) return categorySecondDerivatives(item.parameters, theta);
  const curvature = responseSecondDerivative(item.parameters as ItemParameters, theta);
  return [-curvature, curvature];
}

/**
 * Where an item sits on the ability scale, as a single number.
 *
 * A dichotomous item reports its difficulty. A polytomous item reports the mean
 * of its thresholds, which is the standard summary location and the ability at
 * which the expected score is roughly half the maximum.
 *
 * This is a reporting convenience and nothing more: collapsing a rubric's
 * thresholds to their mean discards precisely the information that makes a
 * polytomous item worth having, so nothing in the engine selects or scores on
 * it. It exists so a transcript can put both formats in one column.
 */
export function itemLocation(item: AnyItem): number {
  if (!isPolytomous(item)) return (item.parameters as ItemParameters).b;
  const thresholds = item.parameters.thresholds;
  let total = 0;
  for (const value of thresholds) total += value;
  return total / thresholds.length;
}

/** The discrimination of an item of either format. */
export function discriminationOf(item: AnyItem): number {
  return item.parameters.a;
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

export interface ModalCategoryOptions {
  /** Ability range to search. Default -4 to 4. */
  readonly range?: readonly [number, number];
  /** Points to evaluate. Default 201. */
  readonly points?: number;
}

/**
 * Which of an item's categories are the most likely outcome somewhere on the
 * ability scale.
 *
 * The complement is the useful half. A category that is modal *nowhere* is a
 * rubric level the model never expects to be anybody's most likely score — no
 * matter how able or unable the candidate, some other level is always more
 * probable. That is what two thresholds collapsing onto each other looks like
 * from the outside, and it means the level is separating nothing: it should be
 * merged with a neighbour, and until it is, a rubric with five levels is really
 * a rubric with four and a level that only ever shows up by accident.
 *
 * Evaluated on a grid rather than solved. The modal category changes at the
 * abilities where two category curves cross, and for the generalized partial
 * credit family those crossings have no closed form — but the function being
 * sampled is a step function with at most `categories - 1` transitions over the
 * range, so a grid this dense misses a band only if that band is narrower than
 * the spacing, in which case the level in question is doing no practical work
 * either.
 *
 * Ties go to the lower category: a level that is only ever *tied* for most
 * likely is not distinguishing a band of ability from the one below it.
 */
export function modalCategories(
  item: AnyItem,
  options: ModalCategoryOptions = {},
): Set<number> {
  const [lower, upper] = options.range ?? [-4, 4];
  const points = options.points ?? 201;
  if (!(lower < upper)) {
    throw new RangeError(
      `modalCategories: range must satisfy lower < upper, received [${lower}, ${upper}]`,
    );
  }
  if (!Number.isInteger(points) || points < 2) {
    throw new RangeError(
      `modalCategories: points must be an integer of at least 2, received ${points}`,
    );
  }

  const modal = new Set<number>();
  const step = (upper - lower) / (points - 1);
  for (let i = 0; i < points; i += 1) {
    const probabilities = categoryProbabilitiesOf(item, i === points - 1 ? upper : lower + i * step);
    let best = 0;
    for (let k = 1; k < probabilities.length; k += 1) {
      if ((probabilities[k] as number) > (probabilities[best] as number)) best = k;
    }
    modal.add(best);
  }
  return modal;
}

/**
 * Categories that are modal nowhere, in order.
 *
 * The reportable form of `modalCategories`: the levels a rubric declares and
 * the model never awards as a most-likely outcome.
 */
export function deadCategories(item: AnyItem, options: ModalCategoryOptions = {}): number[] {
  const modal = modalCategories(item, options);
  const dead: number[] = [];
  for (let k = 0; k < categoryCountOf(item); k += 1) {
    if (!modal.has(k)) dead.push(k);
  }
  return dead;
}
