import { createRng, type Rng } from '../core/random.js';
import { ResponseMatrix, type Cell } from '../calibration/matrix.js';
import type { Item } from '../models/item.js';
import {
  categoryProbabilitiesOf,
  isPolytomous,
  type AnyItem,
  type CategoryResponse,
} from '../models/mixed.js';
import { probabilityCorrect, type Response, type ScoredResponse } from '../models/response.js';

/**
 * Draw a response to one item from a respondent of the given true ability.
 *
 * A Bernoulli draw against the item's response function. This is the generating
 * model the estimators assume, which makes it the right instrument for checking
 * whether they recover what they are supposed to recover — and no evidence at
 * all about whether the model fits real candidates.
 */
export function simulateResponse(item: Item, trueTheta: number, rng: Rng): Response {
  return rng.next() < probabilityCorrect(item.parameters, trueTheta) ? 1 : 0;
}

/**
 * Draw a category from a respondent of the given true ability, for an item of
 * either format.
 *
 * The general case is an inverse-CDF draw down the item's category
 * distribution. The dichotomous case delegates to `simulateResponse` rather
 * than falling out of the same loop, and that is deliberate: both consume one
 * uniform draw and both produce the same distribution, but they map a given
 * draw to opposite outcomes. Walking `[Q, P]` upward returns 1 when
 * `draw >= Q`, while the Bernoulli comparison returns 1 when `draw < P`. Either
 * is correct in isolation; having two of them in one engine would mean a seeded
 * simulation gave different answers depending on which function the caller
 * reached for, and every replayable transcript in the test suite would silently
 * change meaning. One convention, delegated to.
 *
 * The last category is returned as the fallback rather than being reached by
 * accumulation. Floating-point summation of the category probabilities need not
 * land on exactly 1, so a uniform draw of 0.9999999999 against a cumulative sum
 * that stops at 0.9999999998 would otherwise fall off the end of the loop and
 * return nothing at all.
 */
export function simulateCategory(item: AnyItem, trueTheta: number, rng: Rng): number {
  if (!isPolytomous(item)) return simulateResponse(item, trueTheta, rng);
  const probabilities = categoryProbabilitiesOf(item, trueTheta);
  const draw = rng.next();
  let cumulative = 0;
  for (let k = 0; k < probabilities.length - 1; k += 1) {
    cumulative += probabilities[k] as number;
    if (draw < cumulative) return k;
  }
  return probabilities.length - 1;
}

/** Draw categories for a whole mixed-format item set from one respondent. */
export function simulateCategories(
  items: readonly AnyItem[],
  trueTheta: number,
  rng: Rng,
): CategoryResponse[] {
  return items.map((item) => ({ item, category: simulateCategory(item, trueTheta, rng) }));
}

/** Draw responses to a whole item set from a respondent of the given true ability. */
export function simulateResponses(
  items: readonly Item[],
  trueTheta: number,
  rng: Rng,
): ScoredResponse[] {
  return items.map((item) => ({ item, response: simulateResponse(item, trueTheta, rng) }));
}

/**
 * Simulate a whole response matrix: every examinee answering every item.
 *
 * The instrument for checking calibration. Generating data from known
 * parameters and asking whether the calibrator recovers them is the only way to
 * separate a calibration bug from a bank that genuinely fits badly — with real
 * data the two are indistinguishable, because the true parameters are exactly
 * what is unknown.
 */
export function simulateMatrix(
  items: readonly Item[],
  abilities: readonly number[],
  seed = 1,
): ResponseMatrix {
  if (items.length === 0) throw new RangeError('simulateMatrix: at least one item is required');
  if (abilities.length === 0) {
    throw new RangeError('simulateMatrix: at least one examinee is required');
  }
  const rng = createRng(seed);
  const rows: Cell[][] = abilities.map((theta) =>
    items.map((item) => simulateResponse(item, theta, rng) as Cell),
  );
  return new ResponseMatrix({
    rows,
    itemIds: items.map((item) => item.id),
    personIds: abilities.map((_, index) => `p-${index}`),
  });
}
