import { isPolytomous, itemLocation, type AnyItem } from '../models/mixed.js';

/**
 * One item calibrated twice, on two different scales.
 *
 * `source` carries the parameters to be moved; `target` carries the parameters
 * of the same item on the scale being moved onto.
 */
export interface CommonItemPair {
  readonly id: string;
  readonly source: AnyItem;
  readonly target: AnyItem;
}

/**
 * Match two calibrations by item identity.
 *
 * This is the common-item non-equivalent groups design: two samples that may
 * differ in ability sit two forms that share a subset of items, and the shared
 * items are the only thing that can carry information about how the two
 * resulting scales relate. Items appearing in only one calibration are silently
 * excluded, because they carry no such information — but see
 * `requireCommonItems` for the cases where finding none is an error rather than
 * an empty result.
 *
 * Order follows the source bank, so a linking is reproducible.
 */
export function commonItems(
  source: readonly AnyItem[],
  target: readonly AnyItem[],
): CommonItemPair[] {
  const byId = new Map<string, AnyItem>();
  for (const item of target) byId.set(item.id, item);

  const pairs: CommonItemPair[] = [];
  const seen = new Set<string>();
  for (const item of source) {
    if (seen.has(item.id)) {
      throw new RangeError(`commonItems: duplicate item id "${item.id}" in the source bank`);
    }
    seen.add(item.id);
    const match = byId.get(item.id);
    if (match === undefined) continue;
    if (isPolytomous(item) !== isPolytomous(match)) {
      // The same id calibrated once as dichotomous and once as polytomous is
      // not one item calibrated twice; it is two items sharing a name, and
      // linking on it would produce a coefficient with no meaning.
      throw new RangeError(
        `commonItems: item "${item.id}" is dichotomous in one calibration and ` +
          'polytomous in the other',
      );
    }
    if (isPolytomous(item) && isPolytomous(match)) {
      if (item.parameters.thresholds.length !== match.parameters.thresholds.length) {
        throw new RangeError(
          `commonItems: item "${item.id}" has ${item.parameters.thresholds.length} thresholds ` +
            `in one calibration and ${match.parameters.thresholds.length} in the other`,
        );
      }
      if (item.parameters.model !== match.parameters.model) {
        throw new RangeError(
          `commonItems: item "${item.id}" is a ${item.parameters.model} item in one ` +
            `calibration and a ${match.parameters.model} item in the other`,
        );
      }
    }
    pairs.push({ id: item.id, source: item, target: match });
  }
  return pairs;
}

/**
 * Match two calibrations, requiring at least `minimum` common items.
 *
 * Two is the arithmetic floor for the moment methods — mean/sigma needs a
 * spread of difficulties, and a single item has none — but it is nowhere near
 * an adequate anchor. Operational practice puts the anchor at roughly a fifth
 * of the form and asks it to mirror the form's content and difficulty range,
 * because linking coefficients estimated from a handful of items carry standard
 * errors that swamp the differences they are supposed to resolve.
 */
export function requireCommonItems(
  source: readonly AnyItem[],
  target: readonly AnyItem[],
  minimum = 2,
): CommonItemPair[] {
  const pairs = commonItems(source, target);
  if (pairs.length < minimum) {
    throw new RangeError(
      `linking requires at least ${minimum} common items, found ${pairs.length}`,
    );
  }
  return pairs;
}

/**
 * The location parameters a moment method should average over.
 *
 * A dichotomous item contributes its difficulty. A polytomous item contributes
 * every one of its thresholds, not their mean: each threshold is a separate
 * location on the ability scale and each carries its own evidence about where
 * the two scales sit relative to one another. Collapsing a four-threshold item
 * to a single number would give it the same weight as a binary item while
 * discarding three quarters of what it knows.
 */
export function locationParameters(items: readonly AnyItem[]): number[] {
  const locations: number[] = [];
  for (const item of items) {
    if (isPolytomous(item)) locations.push(...item.parameters.thresholds);
    else locations.push(itemLocation(item));
  }
  return locations;
}

/** The discrimination parameters a moment method should average over. */
export function discriminationParameters(items: readonly AnyItem[]): number[] {
  return items.map((item) => item.parameters.a);
}
