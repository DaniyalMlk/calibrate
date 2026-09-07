import type { Item } from '../models/item.js';
import type { SelectionContext, Selector } from './selector.js';

/**
 * Target proportion of the test to devote to each domain.
 *
 * Proportions are normalised, so `{ arrays: 2, graphs: 1 }` and
 * `{ arrays: 0.667, graphs: 0.333 }` mean the same thing.
 */
export type Blueprint = ReadonlyMap<string, number>;

/** Build a blueprint from a plain object, validating and normalising it. */
export function blueprint(targets: Readonly<Record<string, number>>): Blueprint {
  const entries = Object.entries(targets);
  if (entries.length === 0) throw new RangeError('blueprint: at least one domain is required');
  let total = 0;
  for (const [domain, weight] of entries) {
    if (!Number.isFinite(weight) || weight < 0) {
      throw new RangeError(
        `blueprint: target for "${domain}" must be a non-negative finite number, received ${weight}`,
      );
    }
    total += weight;
  }
  if (total <= 0) throw new RangeError('blueprint: targets must not all be zero');
  return new Map(entries.map(([domain, weight]) => [domain, weight / total]));
}

/**
 * Content-balanced selection.
 *
 * Before applying the base rule, restrict the candidate set to the domain that
 * is furthest behind its blueprint target. This is the constrained-CAT
 * ("maximum deficit") approach: content validity is enforced as a hard
 * restriction on the candidate set, and the psychometric criterion then operates
 * inside it.
 *
 * The alternative — treating content as a soft penalty on the information score
 * — produces tests that are *nearly* balanced, which is not a claim a testing
 * programme can make in a report. A test that promises 40% data structures has
 * to deliver 40% data structures.
 *
 * Items with no `domain` are only eligible when no blueprint domain has any
 * candidate left, so an unclassified item can never displace a required one.
 */
export function contentBalanced(base: Selector, targets: Blueprint): Selector {
  if (targets.size === 0) {
    throw new RangeError('contentBalanced: blueprint must name at least one domain');
  }

  return {
    name: `content-balanced(${base.name})`,
    select(context: SelectionContext): Item | null {
      if (context.candidates.length === 0) return null;

      const administered = context.responses.length;
      const counts = new Map<string, number>();
      for (const { item } of context.responses) {
        if (item.domain === undefined) continue;
        counts.set(item.domain, (counts.get(item.domain) ?? 0) + 1);
      }

      // Deficit: how many items short of target this domain currently is.
      let bestDomain: string | null = null;
      let bestDeficit = Number.NEGATIVE_INFINITY;
      for (const [domain, proportion] of targets) {
        const hasCandidate = context.candidates.some((item) => item.domain === domain);
        if (!hasCandidate) continue;
        const target = proportion * (administered + 1);
        const deficit = target - (counts.get(domain) ?? 0);
        // Ties broken by domain name, so selection stays reproducible.
        if (deficit > bestDeficit || (deficit === bestDeficit && bestDomain !== null && domain < bestDomain)) {
          bestDeficit = deficit;
          bestDomain = domain;
        }
      }

      if (bestDomain === null) {
        // No blueprint domain has candidates left; fall back to the whole set.
        return base.select(context);
      }

      const restricted = context.candidates.filter((item) => item.domain === bestDomain);
      return base.select({ ...context, candidates: restricted });
    },
  };
}

/** Observed proportion of administered items in each domain. */
export function contentProportions(items: readonly Item[]): ReadonlyMap<string, number> {
  const counts = new Map<string, number>();
  for (const item of items) {
    const domain = item.domain ?? '(unclassified)';
    counts.set(domain, (counts.get(domain) ?? 0) + 1);
  }
  const proportions = new Map<string, number>();
  if (items.length === 0) return proportions;
  for (const [domain, count] of counts) proportions.set(domain, count / items.length);
  return proportions;
}
