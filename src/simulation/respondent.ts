import { createRng, type Rng } from '../core/random.js';
import { ResponseMatrix, type Cell } from '../calibration/matrix.js';
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
