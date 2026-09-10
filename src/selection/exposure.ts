import { informationOf } from '../models/mixed.js';
import type { AnyItem } from '../models/mixed.js';
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
    select(context: SelectionContext): AnyItem | null {
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
function topCandidates(base: Selector, context: SelectionContext, k: number): AnyItem[] {
  const remaining = [...context.candidates];
  const chosen: AnyItem[] = [];
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
    select(context: SelectionContext): AnyItem | null {
      const remaining = [...context.candidates];
      let last: AnyItem | null = null;
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

/**
 * Observed exposure rates from administered item ids: the fraction of sessions
 * in which each item appeared.
 *
 * Takes ids rather than items because that is what a stored transcript holds. A
 * study that has been serialised and read back has the ids and nothing else, and
 * exposure accounting should not require rehydrating the whole bank to do it.
 */
export function exposureRatesFromIds(
  administered: readonly (readonly string[])[],
): ReadonlyMap<string, number> {
  const counts = new Map<string, number>();
  for (const session of administered) {
    // An item appearing twice in one session still counts once: the rate is the
    // fraction of candidates who saw it, which is the number that matters for
    // both bank security and content balance.
    const seen = new Set<string>();
    for (const id of session) {
      if (seen.has(id)) continue;
      seen.add(id);
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
  }
  const sessions = administered.length;
  const rates = new Map<string, number>();
  if (sessions === 0) return rates;
  for (const [id, count] of counts) rates.set(id, count / sessions);
  return rates;
}

/** Observed exposure rates: the fraction of sessions in which each item appeared. */
export function exposureRates(
  administered: readonly (readonly AnyItem[])[],
): ReadonlyMap<string, number> {
  return exposureRatesFromIds(administered.map((session) => session.map((item) => item.id)));
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
  bank: readonly AnyItem[],
  administered: readonly (readonly AnyItem[])[],
): number {
  if (bank.length === 0) return 0;
  const used = new Set<string>();
  for (const session of administered) for (const item of session) used.add(item.id);
  let unused = 0;
  for (const item of bank) if (!used.has(item.id)) unused += 1;
  return unused / bank.length;
}

/**
 * Population variance of the exposure rates across a whole bank.
 *
 * Items that were never administered count as rate zero. Leaving them out — the
 * obvious shortcut, since they carry no entry in the rate map — would compute
 * the variance among the items the policy actually likes, which is close to zero
 * for a policy that always picks from the same narrow set, and would therefore
 * report the worst possible bank utilisation as the best.
 */
export function exposureVariance(
  bankSize: number,
  rates: ReadonlyMap<string, number>,
): number {
  if (!Number.isInteger(bankSize) || bankSize < 1) {
    throw new RangeError(`exposureVariance: bankSize must be a positive integer, received ${bankSize}`);
  }
  let total = 0;
  for (const rate of rates.values()) total += rate;
  const average = total / bankSize;
  let acc = 0;
  for (const rate of rates.values()) acc += (rate - average) * (rate - average);
  // Every item absent from the map contributes (0 - average)^2.
  acc += (bankSize - rates.size) * average * average;
  return acc / bankSize;
}

/**
 * Test overlap rate: the expected proportion of items two randomly chosen
 * examinees have in common.
 *
 * `T = (N / L) * S^2 + L / N`, for a bank of `N` items, a mean test length of
 * `L` and exposure-rate variance `S^2` (Chen, Ankenmann and Spray). The two
 * ends of the range fall straight out of the formula and are worth stating,
 * because they are what makes the number interpretable: perfectly even exposure
 * gives `S^2 = 0` and an overlap of `L / N`, the floor; a policy that
 * administers the same `L` items to everybody gives `S^2 = (L/N)(1 - L/N)` and
 * an overlap of exactly 1.
 *
 * Overlap is the security number that matters. A bank can have a respectable
 * maximum exposure rate and still be trivially harvestable if the items are
 * shared between examinees in the same combinations.
 */
export function overlapRate(bankSize: number, meanLength: number, rateVariance: number): number {
  if (!(bankSize > 0)) {
    throw new RangeError(`overlapRate: bankSize must be positive, received ${bankSize}`);
  }
  if (!(meanLength > 0)) {
    throw new RangeError(`overlapRate: meanLength must be positive, received ${meanLength}`);
  }
  if (rateVariance < 0) {
    throw new RangeError(`overlapRate: rateVariance must be non-negative, received ${rateVariance}`);
  }
  return (bankSize / meanLength) * rateVariance + meanLength / bankSize;
}

/**
 * Chi-square index of exposure skew: `sum_i (r_i - L/N)^2 / (L/N)`.
 *
 * Zero for perfectly even exposure and growing without bound as the policy
 * concentrates. Reported alongside the overlap rate because it responds to a
 * different feature of the same distribution — a bank with one badly
 * over-exposed item and one with a broadly uneven spread can share an overlap
 * rate but not a chi-square.
 */
export function exposureChiSquare(
  bankSize: number,
  meanLength: number,
  rates: ReadonlyMap<string, number>,
): number {
  const expected = meanLength / bankSize;
  if (!(expected > 0)) {
    throw new RangeError('exposureChiSquare: mean length and bank size must both be positive');
  }
  return (bankSize * exposureVariance(bankSize, rates)) / expected;
}

/**
 * Rank a bank by information at a given ability — a helper for tuning and for
 * tests, not part of the selection path.
 */
export function rankByInformationAt(bank: readonly AnyItem[], theta: number): AnyItem[] {
  return rankByScore(bank, (item) => informationOf(item, theta)).map((e) => e.item);
}
