import type { Rng } from '../core/random.js';
import type { AnyItem, AnyResponse } from '../models/mixed.js';

/**
 * Everything a selection policy is allowed to see when choosing the next item.
 *
 * Deliberately narrow. A policy gets the eligible candidates, the current
 * ability estimate, the responses so far and a seeded generator — and nothing
 * else, so no policy can quietly depend on global state and become unreplayable.
 */
export interface SelectionContext {
  /** Items still eligible: not yet administered, not otherwise excluded. */
  readonly candidates: readonly AnyItem[];
  /** The current ability estimate. */
  readonly theta: number;
  /** Responses given so far, in administration order. */
  readonly responses: readonly AnyResponse[];
  /** Seeded generator. Policies must use this rather than `Math.random`. */
  readonly rng: Rng;
}

/**
 * A next-item selection policy.
 *
 * Returns `null` when it cannot select — an empty candidate list, or a
 * constraint that excludes everything. Callers treat that as a stopping
 * condition rather than an error, because running out of eligible items is a
 * normal way for an adaptive test to end.
 */
export interface Selector {
  /** Stable identifier, used in transcripts and study reports. */
  readonly name: string;
  select(context: SelectionContext): AnyItem | null;
}

/**
 * Rank candidates by a scalar criterion, best first.
 *
 * Ties are broken by item id so that ranking is a total order and therefore
 * reproducible: without it, two items of identical information would be ordered
 * by whatever `Array.prototype.sort` happened to do, which is
 * implementation-defined for equal keys.
 */
export function rankByScore(
  candidates: readonly AnyItem[],
  score: (item: AnyItem) => number,
): { item: AnyItem; score: number }[] {
  return candidates
    .map((item) => ({ item, score: score(item) }))
    .sort((left, right) => {
      if (right.score !== left.score) return right.score - left.score;
      return left.item.id < right.item.id ? -1 : left.item.id > right.item.id ? 1 : 0;
    });
}
