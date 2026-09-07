import { describe, expect, it } from 'vitest';
import {
  clamp,
  linspace,
  logistic,
  logisticDerivative,
  mean,
  requireFinite,
  rootMeanSquareError,
  variance,
} from '../src/core/numeric.js';

describe('logistic', () => {
  it('is one half at zero', () => {
    expect(logistic(0)).toBe(0.5);
  });

  it('matches the closed form at a hand-checkable point', () => {
    // 1 / (1 + e^-1) = 0.7310585786300049
    expect(logistic(1)).toBeCloseTo(0.7310585786300049, 15);
  });

  it('is symmetric about the origin', () => {
    for (const x of [0.1, 1, 3.7, 12]) {
      expect(logistic(-x)).toBeCloseTo(1 - logistic(x), 15);
    }
  });

  it('does not overflow in the tails', () => {
    // The naive 1 / (1 + Math.exp(-x)) form produces Infinity / NaN here.
    expect(logistic(-800)).toBe(0);
    expect(logistic(800)).toBe(1);
    expect(Number.isFinite(logistic(-1e308))).toBe(true);
  });

  it('saturates at the infinities and rejects NaN', () => {
    expect(logistic(Number.POSITIVE_INFINITY)).toBe(1);
    expect(logistic(Number.NEGATIVE_INFINITY)).toBe(0);
    expect(() => logistic(Number.NaN)).toThrow(RangeError);
  });
});

describe('logisticDerivative', () => {
  it('peaks at the origin with height one quarter', () => {
    expect(logisticDerivative(0)).toBe(0.25);
    expect(logisticDerivative(0.5)).toBeLessThan(0.25);
    expect(logisticDerivative(-0.5)).toBeLessThan(0.25);
  });

  it('agrees with a central finite difference', () => {
    const h = 1e-6;
    for (const x of [-2, -0.3, 0, 1.4]) {
      const numeric = (logistic(x + h) - logistic(x - h)) / (2 * h);
      expect(logisticDerivative(x)).toBeCloseTo(numeric, 8);
    }
  });
});

describe('clamp', () => {
  it('restricts to the interval', () => {
    expect(clamp(5, 0, 1)).toBe(1);
    expect(clamp(-5, 0, 1)).toBe(0);
    expect(clamp(0.4, 0, 1)).toBe(0.4);
  });

  it('rejects an inverted interval', () => {
    expect(() => clamp(0, 1, 0)).toThrow(RangeError);
  });
});

describe('summary statistics', () => {
  it('computes mean and population variance', () => {
    expect(mean([1, 2, 3, 4])).toBe(2.5);
    expect(variance([1, 2, 3, 4])).toBeCloseTo(1.25, 12);
  });

  it('throws on empty input instead of returning NaN', () => {
    expect(() => mean([])).toThrow(RangeError);
    expect(() => variance([])).toThrow(RangeError);
  });

  it('computes RMSE and rejects mismatched lengths', () => {
    expect(rootMeanSquareError([1, 2, 3], [1, 2, 3])).toBe(0);
    expect(rootMeanSquareError([0, 0], [3, 4])).toBeCloseTo(Math.sqrt(12.5), 12);
    expect(() => rootMeanSquareError([1], [1, 2])).toThrow(RangeError);
  });
});

describe('linspace', () => {
  it('covers the interval inclusively', () => {
    expect(linspace(-1, 1, 5)).toEqual([-1, -0.5, 0, 0.5, 1]);
  });

  it('lands exactly on the endpoint despite accumulated float error', () => {
    const grid = linspace(-4, 4, 161);
    expect(grid[0]).toBe(-4);
    expect(grid[grid.length - 1]).toBe(4);
    expect(grid).toHaveLength(161);
  });

  it('requires at least two points', () => {
    expect(() => linspace(0, 1, 1)).toThrow(RangeError);
    expect(() => linspace(0, 1, 2.5)).toThrow(RangeError);
  });
});

describe('requireFinite', () => {
  it('names the offending field', () => {
    expect(() => requireFinite(Number.NaN, 'difficulty')).toThrow(/difficulty/);
    expect(() => requireFinite(Number.POSITIVE_INFINITY, 'a')).toThrow(RangeError);
    expect(requireFinite(1.5, 'a')).toBe(1.5);
  });
});
