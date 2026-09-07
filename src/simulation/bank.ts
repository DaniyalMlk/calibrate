import { createRng, type Rng } from '../core/random.js';
import { makeItem, threePL, type Item } from '../models/item.js';
import { ItemPool } from '../session/pool.js';

export interface SyntheticBankOptions {
  /** Number of items. Default 300. */
  readonly size?: number;
  /** Domains to spread items across. Default three interview-prep topics. */
  readonly domains?: readonly string[];
  /** Mean and spread of the lognormal discrimination distribution. */
  readonly discrimination?: { readonly logMean?: number; readonly logSd?: number };
  /** Standard deviation of the normal difficulty distribution. Default 1.2. */
  readonly difficultySd?: number;
  /** Lower asymptote, i.e. one over the number of options. Default 0.25. */
  readonly guessing?: number;
  /** Seed. Default 20260101. */
  readonly seed?: number;
}

/**
 * Generate a synthetic item bank with realistic parameter distributions.
 *
 * Discriminations are lognormal and difficulties normal, which is what
 * calibrated banks actually look like: a long right tail of a few very
 * discriminating items, and difficulties concentrated near the population mean
 * with thin tails. A bank of uniformly spaced, identically discriminating items
 * — the obvious thing to generate — would make every selection policy look
 * equally good, because there would be nothing to choose between.
 *
 * The guessing parameter defaults to 0.25, the chance level for a
 * four-option multiple-choice item.
 */
export function syntheticBank(options: SyntheticBankOptions = {}): Item[] {
  const size = options.size ?? 300;
  if (!Number.isInteger(size) || size < 1) {
    throw new RangeError(`syntheticBank: size must be a positive integer, received ${size}`);
  }
  const domains = options.domains ?? ['arrays', 'graphs', 'dynamic-programming'];
  if (domains.length === 0) throw new RangeError('syntheticBank: at least one domain is required');
  const logMean = options.discrimination?.logMean ?? Math.log(1.1);
  const logSd = options.discrimination?.logSd ?? 0.35;
  const difficultySd = options.difficultySd ?? 1.2;
  const guessing = options.guessing ?? 0.25;
  const rng: Rng = createRng(options.seed ?? 20260101);

  const items: Item[] = [];
  for (let index = 0; index < size; index += 1) {
    const a = Math.min(Math.exp(logMean + logSd * rng.nextNormal()), 4);
    const b = difficultySd * rng.nextNormal();
    const domain = domains[index % domains.length] as string;
    items.push(makeItem(`${domain}-${String(index).padStart(4, '0')}`, threePL(a, b, guessing), { domain }));
  }
  return items;
}

/** The same bank, wrapped in a pool. */
export function syntheticPool(options: SyntheticBankOptions = {}): ItemPool {
  return new ItemPool(syntheticBank(options));
}
