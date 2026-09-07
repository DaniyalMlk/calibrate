import type { Rng } from '../core/random.js';
import type { Item } from '../models/item.js';
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

/** Draw responses to a whole item set from a respondent of the given true ability. */
export function simulateResponses(
  items: readonly Item[],
  trueTheta: number,
  rng: Rng,
): ScoredResponse[] {
  return items.map((item) => ({ item, response: simulateResponse(item, trueTheta, rng) }));
}
