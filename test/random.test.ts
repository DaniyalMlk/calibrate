import { describe, expect, it } from 'vitest';
import { mean, variance } from '../src/core/numeric.js';
import { createRng, normalDeviate } from '../src/core/random.js';

describe('createRng', () => {
  it('replays identically from the same seed', () => {
    const a = createRng(42);
    const b = createRng(42);
    const left = Array.from({ length: 50 }, () => a.next());
    const right = Array.from({ length: 50 }, () => b.next());
    expect(left).toEqual(right);
  });

  it('produces different streams from different seeds', () => {
    const a = createRng(1);
    const b = createRng(2);
    expect(a.next()).not.toBe(b.next());
  });

  it('stays inside [0, 1)', () => {
    const rng = createRng(7);
    for (let i = 0; i < 20000; i += 1) {
      const x = rng.next();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });

  it('is roughly uniform', () => {
    const rng = createRng(2024);
    const buckets = new Array<number>(10).fill(0);
    const draws = 100000;
    for (let i = 0; i < draws; i += 1) {
      const index = Math.floor(rng.next() * 10);
      buckets[index] = (buckets[index] as number) + 1;
    }
    for (const count of buckets) {
      // Each bucket should hold about 10000; allow a wide 5% band.
      expect(count).toBeGreaterThan(draws / 10 - draws * 0.005);
      expect(count).toBeLessThan(draws / 10 + draws * 0.005);
    }
  });

  it('handles a zero seed without collapsing', () => {
    const rng = createRng(0);
    const draws = Array.from({ length: 10 }, () => rng.next());
    expect(new Set(draws).size).toBe(10);
  });

  it('rejects a non-finite seed', () => {
    expect(() => createRng(Number.NaN)).toThrow(RangeError);
  });
});

describe('nextNormal', () => {
  it('has approximately zero mean and unit variance', () => {
    const rng = createRng(99);
    const draws = Array.from({ length: 200000 }, () => rng.nextNormal());
    expect(mean(draws)).toBeCloseTo(0, 2);
    expect(variance(draws)).toBeCloseTo(1, 2);
  });

  it('puts about 68% of mass within one standard deviation', () => {
    const rng = createRng(5);
    let inside = 0;
    const draws = 100000;
    for (let i = 0; i < draws; i += 1) {
      if (Math.abs(rng.nextNormal()) < 1) inside += 1;
    }
    expect(inside / draws).toBeCloseTo(0.6827, 2);
  });

  it('replays identically from the same seed', () => {
    const a = createRng(11);
    const b = createRng(11);
    expect(Array.from({ length: 25 }, () => a.nextNormal())).toEqual(
      Array.from({ length: 25 }, () => b.nextNormal()),
    );
  });
});

describe('normalDeviate', () => {
  it('shifts and scales', () => {
    const rng = createRng(3);
    const draws = Array.from({ length: 100000 }, () => normalDeviate(rng, 2, 0.5));
    expect(mean(draws)).toBeCloseTo(2, 1);
    expect(Math.sqrt(variance(draws))).toBeCloseTo(0.5, 2);
  });

  it('collapses to a point mass at zero spread', () => {
    const rng = createRng(3);
    expect(normalDeviate(rng, 1.5, 0)).toBe(1.5);
  });

  it('rejects a negative standard deviation', () => {
    expect(() => normalDeviate(createRng(1), 0, -1)).toThrow(RangeError);
  });
});

describe('nextInt', () => {
  it('covers the whole range', () => {
    const rng = createRng(13);
    const seen = new Set<number>();
    for (let i = 0; i < 1000; i += 1) seen.add(rng.nextInt(5));
    expect([...seen].sort()).toEqual([0, 1, 2, 3, 4]);
  });

  it('rejects a non-positive or fractional bound', () => {
    const rng = createRng(1);
    expect(() => rng.nextInt(0)).toThrow(RangeError);
    expect(() => rng.nextInt(2.5)).toThrow(RangeError);
  });
});
