import { itemInformation } from '../models/response.js';
import type { Item } from '../models/item.js';
import { rankByScore, type SelectionContext, type Selector } from './selector.js';

/**
 * Randomesque exposure control (Kingsbury and Zara).
 *
 * Rank candidates with the base criterion, then pick uniformly among the top
 * `k`. The cost in measurement precision is small — the difference in
 * information between the best and fifth-best item in a well-built bank is
 * slight — and the gain in bank security is large, because no single item is
 * administered to every candidate of a given ability.
 *
 * Wrapping a base selector rather than reimplementing the criterion means it
 * composes with any of them.
 */
export function randomesque(base: Selector, k = 5): Selector {
  if (!Number.isInteger(k) || k < 1) {
    throw new RangeError(`randomesque: k must be a positive integer, received ${k}`);
  }
  return {
    name: `randomesque(${base.name}, k=${k})`,
    select(context: SelectionContext): Item | null {
      if (context.candidates.length === 0) return null;
      if (k === 1) return base.select(context);

      // Score every candidate through the base selector by asking it to choose
      // from progressively smaller sets would be O(n^2); instead rank directly
      // on the criterion the base selector exposes through a single-item probe.
      const top = topCandidates(base, context, k);
      if (top.length === 0) return null;
      return top[context.rng.nextInt(top.length)] ?? null;
    },
  };
}

/**
 * Ask a selector for its top `k` candidates by repeatedly selecting and
 * removing.
 *
 * This works for any selector rather than only those exposing a scalar
 * criterion, at a cost of `k` selections. `k` is small in practice — 3 to 10 —
 * so the extra work is bounded and predictable.
 */
function topCandidates(base: Selector, context: SelectionContext, k: number): Item[] {
  const remaining = [...context.candidates];
  const chosen: Item[] = [];
  for (let i = 0; i < k && remaining.length > 0; i += 1) {
    const pick = base.select({ ...context, candidates: remaining });
    if (pick === null) break;
    chosen.push(pick);
    const index = remaining.findIndex((item) => item.id === pick.id);
    if (index >= 0) remaining.splice(index, 1);
  }
  return chosen;
}

/** Per-item exposure control parameters, keyed by item id. */
export type ExposureParameters = ReadonlyMap<string, number>;

export interface SympsonHetterOptions {
  /**
   * Probability of administering an item once selected, by item id. Missing
   * items are administered with probability 1.
   */
  readonly parameters: ExposureParameters;
  /**
   * How many candidates to try before giving up and administering the last one
   * tried. Default 10.
   */
  readonly maxAttempts?: number;
}

/**
 * Sympson–Hetter exposure control.
 *
 * Separates *selection* from *administration*: an item chosen by the base rule
 * is administered only with probability `K_i`, and on rejection the next best
 * candidate is tried. Tuning `K_i` so that the resulting exposure rate meets a
 * target is done offline, by simulation — which is precisely what the
 * simulation harness exists for.
 *
 * After `maxAttempts` rejections the last candidate is administered anyway.
 * Falling through forever would be worse than a slightly over-exposed item: it
 * would mean a test that cannot proceed.
 */
export function sympsonHetter(base: Selector, options: SympsonHetterOptions): Selector {
  const maxAttempts = options.maxAttempts ?? 10;
  if (!Number.isInteger(maxAttempts) || maxAttempts < 1) {
    throw new RangeError(
      `sympsonHetter: maxAttempts must be a positive integer, received ${maxAttempts}`,
    );
  }
  for (const [id, value] of options.parameters) {
    if (!(value >= 0 && value <= 1)) {
      throw new RangeError(
        `sympsonHetter: exposure parameter for "${id}" must lie in [0, 1], received ${value}`,
      );
    }
  }

  return {
    name: `sympson-hetter(${base.name})`,
    select(context: SelectionContext): Item | null {
      const remaining = [...context.candidates];
      let last: Item | null = null;
      for (let attempt = 0; attempt < maxAttempts && remaining.length > 0; attempt += 1) {
        const pick = base.select({ ...context, candidates: remaining });
        if (pick === null) break;
        last = pick;
        const probability = options.parameters.get(pick.id) ?? 1;
        if (context.rng.next() < probability) return pick;
        const index = remaining.findIndex((item) => item.id === pick.id);
        if (index >= 0) remaining.splice(index, 1);
      }
      return last;
    },
  };
}

/** Observed exposure rates: the fraction of sessions in which each item appeared. */
export function exposureRates(
  administered: readonly (readonly Item[])[],
): ReadonlyMap<string, number> {
  const counts = new Map<string, number>();
  for (const session of administered) {
    const seen = new Set<string>();
    for (const item of session) {
      if (seen.has(item.id)) continue;
      seen.add(item.id);
      counts.set(item.id, (counts.get(item.id) ?? 0) + 1);
    }
  }
  const sessions = administered.length;
  const rates = new Map<string, number>();
  if (sessions === 0) return rates;
  for (const [id, count] of counts) rates.set(id, count / sessions);
  return rates;
}

/**
 * A crude but useful bank-health number: the fraction of the bank that was never
 * administered across a set of sessions.
 *
 * A maximum-information policy over a realistic bank routinely leaves more than
 * half of it untouched, which is the clearest single statement of why exposure
 * control matters.
 */
export function unusedFraction(
  bank: readonly Item[],
  administered: readonly (readonly Item[])[],
): number {
  if (bank.length === 0) return 0;
  const used = new Set<string>();
  for (const session of administered) for (const item of session) used.add(item.id);
  let unused = 0;
  for (const item of bank) if (!used.has(item.id)) unused += 1;
  return unused / bank.length;
}

/**
 * Rank a bank by information at a given ability — a helper for tuning and for
 * tests, not part of the selection path.
 */
export function rankByInformationAt(bank: readonly Item[], theta: number): Item[] {
  return rankByScore(bank, (item) => itemInformation(item.parameters, theta)).map((e) => e.item);
}
