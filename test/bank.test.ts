import { describe, expect, it } from 'vitest';
import { mean, variance } from '../src/core/numeric.js';
import { describeModel } from '../src/models/item.js';
import { deadCategories, isPolytomous, type PolytomousItem } from '../src/models/mixed.js';
import { syntheticBank, syntheticMixedBank, syntheticPool } from '../src/simulation/bank.js';

describe('syntheticBank', () => {
  it('is deterministic for a given seed', () => {
    const a = syntheticBank({ size: 30, seed: 5 });
    const b = syntheticBank({ size: 30, seed: 5 });
    expect(a.map((item) => item.parameters)).toEqual(b.map((item) => item.parameters));
  });

  it('differs between seeds', () => {
    const a = syntheticBank({ size: 30, seed: 5 });
    const b = syntheticBank({ size: 30, seed: 6 });
    expect(a[0]?.parameters.b).not.toBe(b[0]?.parameters.b);
  });

  it('produces unique ids and valid 3PL parameters', () => {
    const bank = syntheticBank({ size: 200, seed: 11 });
    expect(new Set(bank.map((item) => item.id)).size).toBe(200);
    for (const item of bank) {
      expect(describeModel(item.parameters)).toBe('3PL');
      expect(item.parameters.a).toBeGreaterThan(0);
      expect(item.parameters.a).toBeLessThanOrEqual(4);
      expect(item.parameters.c).toBeCloseTo(0.25, 12);
    }
  });

  it('spreads items evenly across the requested domains', () => {
    const bank = syntheticBank({ size: 300, domains: ['a', 'b', 'c'], seed: 2 });
    const counts = new Map<string, number>();
    for (const item of bank) counts.set(item.domain as string, (counts.get(item.domain as string) ?? 0) + 1);
    expect([...counts.values()]).toEqual([100, 100, 100]);
  });

  it('draws difficulties with roughly the requested spread', () => {
    const bank = syntheticBank({ size: 4000, difficultySd: 1.2, seed: 3 });
    const difficulties = bank.map((item) => item.parameters.b);
    expect(mean(difficulties)).toBeCloseTo(0, 1);
    expect(Math.sqrt(variance(difficulties))).toBeCloseTo(1.2, 1);
  });

  it('draws discriminations with a right-skewed distribution', () => {
    // Lognormal: the mean sits above the median.
    const bank = syntheticBank({ size: 4000, seed: 4 });
    const values = bank.map((item) => item.parameters.a).sort((x, y) => x - y);
    const median = values[Math.floor(values.length / 2)] as number;
    expect(mean(values)).toBeGreaterThan(median);
    expect(median).toBeGreaterThan(0.9);
    expect(median).toBeLessThan(1.4);
  });

  it('rejects degenerate configuration', () => {
    expect(() => syntheticBank({ size: 0 })).toThrow(RangeError);
    expect(() => syntheticBank({ domains: [] })).toThrow(RangeError);
  });
});

describe('syntheticPool', () => {
  it('wraps the bank in a usable pool', () => {
    const pool = syntheticPool({ size: 60, seed: 9 });
    expect(pool.size).toBe(60);
    expect(pool.domains()).toEqual(['arrays', 'dynamic-programming', 'graphs']);
    expect(pool.eligible(new Set())).toHaveLength(60);
  });
});

describe('syntheticMixedBank threshold spacing', () => {
  /** The rubric items of a bank, with their generated parameters. */
  function rubrics(options: Parameters<typeof syntheticMixedBank>[0] = {}): PolytomousItem[] {
    return syntheticMixedBank({ size: 300, polytomousFraction: 0.2, seed: 20260101, ...options })
      .filter(isPolytomous) as PolytomousItem[];
  }

  /** Adjacent threshold gaps, measured in the item's own metric. */
  function scaledGaps(item: PolytomousItem): number[] {
    const { a, thresholds } = item.parameters;
    const gaps: number[] = [];
    for (let k = 1; k < thresholds.length; k += 1) {
      gaps.push(a * ((thresholds[k] as number) - (thresholds[k - 1] as number)));
    }
    return gaps;
  }

  it('spaces thresholds in the item’s metric, not in raw logits', () => {
    // The regression this exists for. Spacing in raw logits makes the *raw*
    // gap constant across the bank and the scaled gap vary with `a`; spacing in
    // the item's metric does the reverse, which is the one that decides whether
    // the middle categories are modal.
    const items = rubrics();
    const scaled = items.flatMap(scaledGaps);
    const raw = items.flatMap((item) => {
      const { thresholds } = item.parameters;
      const gaps: number[] = [];
      for (let k = 1; k < thresholds.length; k += 1) {
        gaps.push((thresholds[k] as number) - (thresholds[k - 1] as number));
      }
      return gaps;
    });
    expect(Math.sqrt(variance(scaled))).toBeLessThan(Math.sqrt(variance(raw)));
  });

  it('clears the 2 ln 2 threshold on every item the span cap does not bind', () => {
    const CRITICAL = 2 * Math.log(2);
    // An item whose thresholds were not narrowed by the span is spaced at the
    // requested value, so every one of its gaps must clear the critical value
    // by roughly the margin the default was chosen for.
    const items = rubrics({ thresholdJitter: 0, thresholdSpan: 99 });
    for (const item of items) {
      for (const gap of scaledGaps(item)) {
        expect(gap).toBeGreaterThan(CRITICAL);
        expect(gap).toBeCloseTo(2, 9);
      }
    }
  });

  it('honours the span cap, which is what keeps thresholds on a measurable scale', () => {
    for (const span of [3, 5, 8]) {
      for (const item of rubrics({ thresholdJitter: 0, thresholdSpan: span })) {
        const thresholds = item.parameters.thresholds;
        const width = (thresholds.at(-1) as number) - (thresholds[0] as number);
        expect(width).toBeLessThanOrEqual(span + 1e-9);
      }
    }
  });

  it('leaves far fewer levels modal nowhere than raw-logit spacing did', () => {
    // The measured consequence. `thresholdSpacing: 1` reproduces the old
    // behaviour closely enough to stand as the before.
    const dead = (options: Parameters<typeof syntheticMixedBank>[0]): number =>
      rubrics(options).filter((item) => deadCategories(item).length > 0).length;

    const before = dead({ thresholdSpacing: 1, thresholdSpan: 99 });
    const after = dead({});
    expect(before).toBeGreaterThan(50);
    expect(after).toBeLessThan(before / 2);
  });

  it('still produces some collapsed levels, because some items cannot carry five', () => {
    // Not a failure to be tuned away. A weakly discriminating item genuinely
    // cannot separate five ordered levels inside a five-logit range, and a
    // generator that always managed it would make every rubric look better than
    // rubrics are — which is the opposite of what a synthetic bank is for.
    expect(rubrics().filter((item) => deadCategories(item).length > 0).length).toBeGreaterThan(0);
  });

  it('generates collapsed levels on demand, below the critical spacing', () => {
    const tight = rubrics({ thresholdSpacing: 0.8, thresholdJitter: 0 });
    expect(tight.every((item) => deadCategories(item).length > 0)).toBe(true);
  });

  it('treats partial credit as its own metric, where discrimination is one', () => {
    // The partial credit model fixes `a` at 1 by definition, so the spacing
    // converts to itself and the raw gaps are the requested value exactly.
    for (const item of rubrics({
      polytomousModel: 'partial-credit',
      thresholdJitter: 0,
      thresholdSpan: 99,
    })) {
      expect(item.parameters.a).toBe(1);
      for (const gap of scaledGaps(item)) expect(gap).toBeCloseTo(2, 9);
    }
  });

  it('rejects a degenerate spacing, jitter or span', () => {
    expect(() => syntheticMixedBank({ thresholdSpacing: 0 })).toThrow(/thresholdSpacing/);
    expect(() => syntheticMixedBank({ thresholdSpacing: Number.NaN })).toThrow(/thresholdSpacing/);
    expect(() => syntheticMixedBank({ thresholdJitter: -1 })).toThrow(/thresholdJitter/);
    expect(() => syntheticMixedBank({ thresholdSpan: 0 })).toThrow(/thresholdSpan/);
  });
});
