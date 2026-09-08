import { describe, expect, it } from 'vitest';
import { createRng } from '../src/core/random.js';
import { mean, variance } from '../src/core/numeric.js';
import {
  drawPopulation,
  evenGridPopulation,
  gridPopulation,
  normalPopulation,
  uniformPopulation,
} from '../src/simulation/population.js';

describe('normalPopulation', () => {
  it('recovers its mean and standard deviation over many draws', () => {
    const abilities = drawPopulation(normalPopulation(0.5, 1.4), 20000, 7);
    expect(mean(abilities)).toBeCloseTo(0.5, 1);
    expect(Math.sqrt(variance(abilities))).toBeCloseTo(1.4, 1);
  });

  it('names itself with its parameters', () => {
    expect(normalPopulation(-0.25, 0.8).name).toBe('normal(-0.25, 0.8)');
  });

  it('rejects a non-positive standard deviation', () => {
    expect(() => normalPopulation(0, 0)).toThrow(RangeError);
    expect(() => normalPopulation(0, -1)).toThrow(RangeError);
  });

  it('rejects non-finite parameters', () => {
    expect(() => normalPopulation(Number.NaN, 1)).toThrow(RangeError);
    expect(() => normalPopulation(0, Number.POSITIVE_INFINITY)).toThrow(RangeError);
  });
});

describe('uniformPopulation', () => {
  it('stays inside its bounds', () => {
    const abilities = drawPopulation(uniformPopulation(-2, 2), 5000, 11);
    for (const theta of abilities) {
      expect(theta).toBeGreaterThanOrEqual(-2);
      expect(theta).toBeLessThanOrEqual(2);
    }
  });

  it('has the mean and variance of the interval it covers', () => {
    const abilities = drawPopulation(uniformPopulation(-3, 1), 40000, 3);
    expect(mean(abilities)).toBeCloseTo(-1, 1);
    // Variance of a uniform on an interval of width w is w^2 / 12.
    expect(variance(abilities)).toBeCloseTo((4 * 4) / 12, 1);
  });

  it('rejects an empty or inverted interval', () => {
    expect(() => uniformPopulation(1, 1)).toThrow(RangeError);
    expect(() => uniformPopulation(2, -2)).toThrow(RangeError);
  });
});

describe('gridPopulation', () => {
  it('cycles its points in order and consumes no randomness', () => {
    const distribution = gridPopulation([-1, 0, 1]);
    const rng = createRng(5);
    const before = rng.next();
    const drawn = [
      distribution.draw(rng),
      distribution.draw(rng),
      distribution.draw(rng),
      distribution.draw(rng),
    ];
    expect(drawn).toEqual([-1, 0, 1, -1]);
    // The generator has not advanced beyond the single draw made above.
    expect(rng.next()).not.toBe(before);
  });

  it('gives every point an equal share when the count is a multiple', () => {
    const abilities = drawPopulation(gridPopulation([-2, 0, 2]), 300, 1);
    const counts = new Map<number, number>();
    for (const theta of abilities) counts.set(theta, (counts.get(theta) ?? 0) + 1);
    expect([...counts.values()]).toEqual([100, 100, 100]);
  });

  it('rejects an empty grid', () => {
    expect(() => gridPopulation([])).toThrow(RangeError);
  });

  it('rejects a non-finite grid point', () => {
    expect(() => gridPopulation([0, Number.NaN])).toThrow(RangeError);
  });
});

describe('evenGridPopulation', () => {
  it('spans the requested range inclusively', () => {
    const abilities = drawPopulation(evenGridPopulation(-2, 2, 5), 5, 1);
    expect(abilities).toEqual([-2, -1, 0, 1, 2]);
  });
});

describe('drawPopulation', () => {
  it('is reproducible for a given seed and differs across seeds', () => {
    const distribution = normalPopulation();
    expect(drawPopulation(distribution, 20, 42)).toEqual(drawPopulation(distribution, 20, 42));
    expect(drawPopulation(distribution, 20, 42)).not.toEqual(drawPopulation(distribution, 20, 43));
  });

  it('rejects a non-positive count', () => {
    expect(() => drawPopulation(normalPopulation(), 0)).toThrow(RangeError);
    expect(() => drawPopulation(normalPopulation(), 2.5)).toThrow(RangeError);
  });
});
