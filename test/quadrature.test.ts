import { describe, expect, it } from 'vitest';
import { sum } from '../src/core/numeric.js';
import {
  gaussHermite,
  integrate,
  normalGaussHermiteRule,
  normalGridRule,
} from '../src/core/quadrature.js';

/** Exact value of the integral of exp(-x^2) x^k over the real line. */
function hermiteMoment(k: number): number {
  if (k % 2 === 1) return 0;
  // Gamma((k + 1) / 2); for even k this is a half-integer gamma with a closed form.
  const m = k / 2;
  let value = Math.sqrt(Math.PI);
  for (let i = 1; i <= m; i += 1) value *= i - 0.5;
  return value;
}

describe('gaussHermite', () => {
  it('reproduces the two-point rule in closed form', () => {
    const { nodes, weights } = gaussHermite(2);
    expect(nodes[0]).toBeCloseTo(-Math.SQRT1_2, 13);
    expect(nodes[1]).toBeCloseTo(Math.SQRT1_2, 13);
    expect(weights[0]).toBeCloseTo(Math.sqrt(Math.PI) / 2, 13);
    expect(weights[1]).toBeCloseTo(Math.sqrt(Math.PI) / 2, 13);
  });

  it('reproduces the three-point rule in closed form', () => {
    const { nodes, weights } = gaussHermite(3);
    expect(nodes[0]).toBeCloseTo(-Math.sqrt(1.5), 13);
    expect(nodes[1]).toBeCloseTo(0, 13);
    expect(nodes[2]).toBeCloseTo(Math.sqrt(1.5), 13);
    expect(weights[1]).toBeCloseTo((2 * Math.sqrt(Math.PI)) / 3, 13);
    expect(weights[0]).toBeCloseTo(Math.sqrt(Math.PI) / 6, 13);
    expect(weights[2]).toBeCloseTo(Math.sqrt(Math.PI) / 6, 13);
  });

  it('integrates polynomials exactly up to degree 2n - 1', () => {
    for (const n of [2, 3, 5, 8]) {
      const rule = gaussHermite(n);
      for (let degree = 0; degree <= 2 * n - 1; degree += 1) {
        const value = integrate(rule, (x) => Math.pow(x, degree));
        expect(value).toBeCloseTo(hermiteMoment(degree), 8);
      }
    }
  });

  it('weights sum to sqrt(pi)', () => {
    for (const n of [2, 5, 20, 41, 80]) {
      expect(sum([...gaussHermite(n).weights])).toBeCloseTo(Math.sqrt(Math.PI), 10);
    }
  });

  it('produces symmetric nodes', () => {
    const { nodes } = gaussHermite(21);
    for (let i = 0; i < nodes.length; i += 1) {
      expect(nodes[i]).toBeCloseTo(-(nodes[nodes.length - 1 - i] as number), 12);
    }
    expect(nodes[10]).toBeCloseTo(0, 12);
  });

  it('is stable at high order, where the raw recurrence would overflow', () => {
    const rule = gaussHermite(120);
    expect(rule.nodes.every((x) => Number.isFinite(x))).toBe(true);
    expect(rule.weights.every((w) => Number.isFinite(w) && w > 0)).toBe(true);
    expect(sum([...rule.weights])).toBeCloseTo(Math.sqrt(Math.PI), 9);
  });

  it('rejects an order below two', () => {
    expect(() => gaussHermite(1)).toThrow(RangeError);
    expect(() => gaussHermite(4.5)).toThrow(RangeError);
  });
});

describe('normalGaussHermiteRule', () => {
  it('has weights summing to one', () => {
    expect(sum([...normalGaussHermiteRule(0, 1).weights])).toBeCloseTo(1, 12);
    expect(sum([...normalGaussHermiteRule(0.5, 2, 25).weights])).toBeCloseTo(1, 12);
  });

  it('recovers the mean and variance of the prior', () => {
    const rule = normalGaussHermiteRule(0.4, 1.3, 41);
    const m = integrate(rule, (t) => t);
    const second = integrate(rule, (t) => t * t);
    expect(m).toBeCloseTo(0.4, 10);
    expect(second - m * m).toBeCloseTo(1.3 * 1.3, 9);
  });

  it('integrates a normal expectation that has a closed form', () => {
    // E[exp(t)] for t ~ N(mu, sigma^2) is exp(mu + sigma^2 / 2).
    const rule = normalGaussHermiteRule(0.2, 0.9, 41);
    expect(integrate(rule, Math.exp)).toBeCloseTo(Math.exp(0.2 + (0.9 * 0.9) / 2), 8);
  });

  it('rejects a non-positive standard deviation', () => {
    expect(() => normalGaussHermiteRule(0, 0)).toThrow(RangeError);
    expect(() => normalGaussHermiteRule(0, -1)).toThrow(RangeError);
  });
});

describe('normalGridRule', () => {
  it('has weights summing to one', () => {
    expect(sum([...normalGridRule(0, 1).weights])).toBeCloseTo(1, 14);
  });

  it('spans mean +/- halfWidth standard deviations', () => {
    const { nodes } = normalGridRule(0.5, 2, 41, 4);
    expect(nodes[0]).toBeCloseTo(0.5 - 8, 12);
    expect(nodes[nodes.length - 1]).toBeCloseTo(0.5 + 8, 12);
    expect(nodes).toHaveLength(41);
  });

  it('recovers the prior mean, and the variance up to truncation', () => {
    const rule = normalGridRule(0.3, 1, 81, 5);
    const m = integrate(rule, (t) => t);
    const second = integrate(rule, (t) => t * t);
    expect(m).toBeCloseTo(0.3, 10);
    expect(second - m * m).toBeCloseTo(1, 4);
  });

  it('agrees with the Gauss-Hermite rule on a smooth integrand', () => {
    const grid = normalGridRule(0, 1, 81, 5);
    const gh = normalGaussHermiteRule(0, 1, 41);
    const f = (t: number): number => 1 / (1 + Math.exp(-1.4 * (t - 0.3)));
    expect(integrate(grid, f)).toBeCloseTo(integrate(gh, f), 6);
  });

  it('rejects degenerate parameters', () => {
    expect(() => normalGridRule(0, 1, 1)).toThrow(RangeError);
    expect(() => normalGridRule(0, 1, 41, 0)).toThrow(RangeError);
    expect(() => normalGridRule(0, -1)).toThrow(RangeError);
  });
});

describe('integrate', () => {
  it('rejects a malformed rule', () => {
    expect(() => integrate({ nodes: [0, 1], weights: [1] }, (x) => x)).toThrow(RangeError);
  });
});
