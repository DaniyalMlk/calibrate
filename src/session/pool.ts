import type { AnyItem } from '../models/mixed.js';

/**
 * An item bank with the indexes a session needs.
 *
 * The pool is immutable. Session state — which items have been used — lives in
 * the session, not here, so one pool can serve any number of concurrent
 * sessions without any of them able to corrupt another.
 */
export class ItemPool {
  readonly #items: readonly AnyItem[];
  readonly #byId: ReadonlyMap<string, AnyItem>;
  readonly #byDomain: ReadonlyMap<string, readonly AnyItem[]>;

  constructor(items: readonly AnyItem[]) {
    if (items.length === 0) throw new RangeError('ItemPool: at least one item is required');

    const byId = new Map<string, AnyItem>();
    for (const item of items) {
      if (byId.has(item.id)) {
        // A duplicate id would let the same question be administered twice
        // under two names, and would silently break exposure accounting.
        throw new RangeError(`ItemPool: duplicate item id "${item.id}"`);
      }
      byId.set(item.id, item);
    }

    const byDomain = new Map<string, AnyItem[]>();
    for (const item of items) {
      const domain = item.domain;
      if (domain === undefined) continue;
      const bucket = byDomain.get(domain);
      if (bucket === undefined) byDomain.set(domain, [item]);
      else bucket.push(item);
    }

    this.#items = Object.freeze([...items]);
    this.#byId = byId;
    this.#byDomain = byDomain;
  }

  get size(): number {
    return this.#items.length;
  }

  /** Every item in the pool, in construction order. */
  all(): readonly AnyItem[] {
    return this.#items;
  }

  byId(id: string): AnyItem | undefined {
    return this.#byId.get(id);
  }

  /** Domains present in the pool, sorted for reproducible reporting. */
  domains(): string[] {
    return [...this.#byDomain.keys()].sort();
  }

  inDomain(domain: string): readonly AnyItem[] {
    return this.#byDomain.get(domain) ?? [];
  }

  /** Items not in `used`, preserving pool order. */
  eligible(used: ReadonlySet<string>): AnyItem[] {
    return this.#items.filter((item) => !used.has(item.id));
  }
}
