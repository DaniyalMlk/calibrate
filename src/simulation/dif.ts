/**
 * Simulating two groups whose items may or may not behave the same way.
 *
 * The point of this generator is to produce the two things a DIF method has to
 * tell apart, independently of each other.
 *
 * *Impact* is a real difference in ability between the groups. It is not bias,
 * and a testing programme that removed every item showing it would be removing
 * the test. *Bias* is an item whose response function differs between the
 * groups at equal ability. Only the second is DIF. Real data always contains
 * both, which is why a method has to be shown separating them on data where the
 * truth is known — and that is what this exists for.
 */

import { createRng, type Rng } from '../core/random.js';
import type { Cell } from '../calibration/matrix.js';
import { ResponseMatrix } from '../calibration/matrix.js';
import type { Group } from '../dif/strata.js';
import { makeItem, type Item, type ItemParameters } from '../models/item.js';
import { simulateResponse } from './respondent.js';

export interface DifShift {
  /** Index of the item in the bank. */
  readonly item: number;
  /**
   * Added to the item's difficulty for focal candidates only.
   *
   * Positive means the item is harder for the focal group at equal ability,
   * which is the uniform DIF the Mantel-Haenszel family is built to detect.
   */
  readonly difficulty?: number;
  /**
   * Multiplies the item's discrimination for focal candidates only.
   *
   * This is non-uniform DIF: the two response functions cross, so the item
   * favours one group below the crossing point and the other above it. Worth
   * generating precisely because the matched-group methods handle it badly —
   * the advantages cancel when pooled across strata, and an item with a large
   * non-uniform effect can produce an odds ratio of almost exactly one.
   */
  readonly discrimination?: number;
}

export interface DifSimulationOptions {
  readonly bank: readonly Item[];
  /** Candidates in the reference group. Default 1000. */
  readonly referenceCount?: number;
  /** Candidates in the focal group. Default 1000. */
  readonly focalCount?: number;
  /** Mean ability of the reference group. Default 0. */
  readonly referenceMean?: number;
  /** Mean ability of the focal group. Default 0; set it non-zero for impact. */
  readonly focalMean?: number;
  /** Ability spread in each group. Default 1. */
  readonly abilitySd?: number;
  /** Items that behave differently for the focal group. Default none. */
  readonly shifts?: readonly DifShift[];
  /** Seed. Default 20260301. */
  readonly seed?: number;
}

export interface DifSimulation {
  readonly matrix: ResponseMatrix;
  readonly groups: Group[];
  /** The ability each simulated candidate was drawn with. */
  readonly abilities: number[];
  /** The bank as the focal group experiences it. */
  readonly focalBank: Item[];
  /** Item indices given a shift, in ascending order. */
  readonly shifted: number[];
}

function shiftItem(item: Item, shift: DifShift): Item {
  const { a, b, c, d, metric } = item.parameters;
  const parameters: ItemParameters = {
    a: a * (shift.discrimination ?? 1),
    b: b + (shift.difficulty ?? 0),
    c,
    d,
    metric,
  };
  return makeItem(item.id, parameters, {
    ...(item.domain === undefined ? {} : { domain: item.domain }),
    ...(item.tags === undefined ? {} : { tags: item.tags }),
  });
}

/**
 * Generate a two-group response matrix from a bank, with known shifts.
 *
 * Both groups answer the same items in the same order, so the resulting matrix
 * is exactly what a real administration of one form to a mixed population looks
 * like. What makes it useful is that the shifted items are returned alongside
 * it: a scan can be scored against the truth rather than eyeballed.
 */
export function simulateDif(options: DifSimulationOptions): DifSimulation {
  const { bank } = options;
  if (bank.length === 0) throw new RangeError('simulateDif: the bank is empty');
  const referenceCount = options.referenceCount ?? 1000;
  const focalCount = options.focalCount ?? 1000;
  for (const [label, count] of [
    ['referenceCount', referenceCount],
    ['focalCount', focalCount],
  ] as const) {
    if (!Number.isInteger(count) || count < 1) {
      throw new RangeError(`simulateDif: ${label} must be a positive integer, received ${count}`);
    }
  }
  const abilitySd = options.abilitySd ?? 1;
  if (!(abilitySd > 0)) {
    throw new RangeError(`simulateDif: abilitySd must be positive, received ${abilitySd}`);
  }

  const shifts = options.shifts ?? [];
  const focalBank = [...bank];
  const shifted: number[] = [];
  for (const shift of shifts) {
    const target = bank[shift.item];
    if (target === undefined) {
      throw new RangeError(`simulateDif: no item at index ${shift.item}`);
    }
    if (shift.discrimination !== undefined && !(shift.discrimination > 0)) {
      throw new RangeError(
        `simulateDif: the discrimination multiplier must be positive, received ${shift.discrimination}`,
      );
    }
    focalBank[shift.item] = shiftItem(target, shift);
    shifted.push(shift.item);
  }
  shifted.sort((a, b) => a - b);

  const rng: Rng = createRng(options.seed ?? 20260301);
  const referenceMean = options.referenceMean ?? 0;
  const focalMean = options.focalMean ?? 0;

  const groups: Group[] = [];
  const abilities: number[] = [];
  const rows: Cell[][] = [];

  const draw = (group: Group, count: number, mean: number, items: readonly Item[]): void => {
    for (let person = 0; person < count; person += 1) {
      const theta = mean + abilitySd * rng.nextNormal();
      groups.push(group);
      abilities.push(theta);
      rows.push(items.map((item) => simulateResponse(item, theta, rng) as Cell));
    }
  };

  draw('reference', referenceCount, referenceMean, bank);
  draw('focal', focalCount, focalMean, focalBank);

  return {
    matrix: new ResponseMatrix({
      rows,
      itemIds: bank.map((item) => item.id),
      personIds: rows.map((_, index) => `p-${index}`),
    }),
    groups,
    abilities,
    focalBank,
    shifted,
  };
}
