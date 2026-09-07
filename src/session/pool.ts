import type { Item } from '../models/item.js';

/**
 * An item bank with the indexes a session needs.
 *
 * The pool is immutable. Session state — which items have been used — lives in
 * the session, not here, so one pool can serve any number of concurrent
 * sessions without any of them able to corrupt another.
 */
export class ItemPool {
  readonly #items: readonly Item[];
  readonly #byId: ReadonlyMap<string, Item>;
  readonly #byDomain: ReadonlyMap<string, readonly Item[]>;

  constructor(items: readonly Item[]) {
    if (items.length === 0) throw new RangeError('ItemPool: at least one item is required');

    const byId = new Map<string, Item>();
    for (const item of items) {
      if (byId.has(item.id)) {
        // A duplicate id would let the same question be administered twice
        // under two names, and would silently break exposure accounting.
        throw new RangeError(`ItemPool: duplicate item id "${item.id}"`);
      }
      byId.set(item.id, item);
    }

    const byDomain = new Map<string, Item[]>();
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
  all(): readonly Item[] {
    return this.#items;
  }

  byId(id: string): Item | undefined {
    return this.#byId.get(id);
  }

  /** Domains present in the pool, sorted for reproducible reporting. */
  domains(): string[] {
    return [...this.#byDomain.keys()].sort();
  }

  inDomain(domain: string): readonly Item[] {
    return this.#byDomain.get(domain) ?? [];
  }

  /** Items not in `used`, preserving pool order. */
  eligible(used: ReadonlySet<string>): Item[] {
    return this.#items.filter((item) => !used.has(item.id));
  }
}
