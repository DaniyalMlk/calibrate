import { describe, expect, it } from 'vitest';
import { mean, variance } from '../src/core/numeric.js';
import { describeModel } from '../src/models/item.js';
import { syntheticBank, syntheticPool } from '../src/simulation/bank.js';

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
