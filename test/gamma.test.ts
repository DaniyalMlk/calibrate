import { describe, expect, it } from 'vitest';
import {
  chiSquareUpperTail,
  logGamma,
  lowerGamma,
  normalTwoSidedTail,
  normalUpperTail,
  upperGamma,
} from '../src/core/gamma.js';

describe('logGamma', () => {
  it('reproduces the factorials at integer arguments', () => {
    // gamma(n) = (n-1)!, so the log of the function is the log of the factorial.
    let logFactorial = 0;
    for (let n = 1; n <= 20; n += 1) {
      expect(logGamma(n)).toBeCloseTo(logFactorial, 10);
      logFactorial += Math.log(n);
    }
  });

  it('gives the half-integer values from the duplication of sqrt(pi)', () => {
    // gamma(1/2) = sqrt(pi), and each step up multiplies by the current argument.
    expect(logGamma(0.5)).toBeCloseTo(Math.log(Math.sqrt(Math.PI)), 12);
    expect(logGamma(1.5)).toBeCloseTo(Math.log(0.5 * Math.sqrt(Math.PI)), 12);
    expect(logGamma(2.5)).toBeCloseTo(Math.log(0.75 * Math.sqrt(Math.PI)), 12);
  });

  it('satisfies the reflection formula below a half', () => {
    for (const x of [0.1, 0.25, 0.4, 0.49]) {
      const reflected = Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
      expect(logGamma(x)).toBeCloseTo(reflected, 12);
    }
  });

  it('stays finite far past where the function itself overflows', () => {
    // gamma(172) is not representable; its log is about 711.
    expect(logGamma(100)).toBeCloseTo(359.1342053695754, 8);
    expect(Number.isFinite(logGamma(1e6))).toBe(true);
  });

  it('rejects a non-positive or non-finite argument', () => {
    expect(() => logGamma(0)).toThrow(/must be positive/);
    expect(() => logGamma(-2)).toThrow(/must be positive/);
    expect(() => logGamma(Number.NaN)).toThrow(/logGamma: x/);
  });
});

describe('the regularized incomplete gamma function', () => {
  it('partitions unity across the crossover between the two series', () => {
    // The series are chosen by whether x is below a + 1, so walking x across
    // that boundary exercises both sides of the branch.
    for (const a of [0.5, 1, 2.5, 7, 30]) {
      for (const x of [0, 0.3, 1, a, a + 1, a + 3, 4 * a + 6]) {
        expect(lowerGamma(a, x) + upperGamma(a, x)).toBeCloseTo(1, 12);
      }
    }
  });

  it('agrees with the elementary form at a = 1', () => {
    // P(1, x) = 1 - exp(-x) exactly.
    for (const x of [0, 0.25, 1, 2, 5, 12]) {
      expect(lowerGamma(1, x)).toBeCloseTo(1 - Math.exp(-x), 12);
      expect(upperGamma(1, x)).toBeCloseTo(Math.exp(-x), 12);
    }
  });

  it('matches a published value away from both limits', () => {
    expect(lowerGamma(2.5, 3)).toBeCloseTo(0.6937810815867212, 10);
    expect(upperGamma(2.5, 3)).toBeCloseTo(0.30621891841327875, 10);
  });

  it('runs from zero to one monotonically', () => {
    let previous = -1;
    for (let x = 0; x <= 20; x += 0.25) {
      const value = lowerGamma(3, x);
      expect(value).toBeGreaterThanOrEqual(previous);
      previous = value;
    }
    expect(lowerGamma(3, 0)).toBe(0);
    expect(lowerGamma(3, 200)).toBeCloseTo(1, 12);
  });

  it('rejects a non-positive shape or a negative argument', () => {
    expect(() => lowerGamma(0, 1)).toThrow(/a must be positive/);
    expect(() => upperGamma(-1, 1)).toThrow(/a must be positive/);
    expect(() => lowerGamma(1, -0.5)).toThrow(/must not be negative/);
    expect(() => upperGamma(1, -0.5)).toThrow(/must not be negative/);
  });
});

describe('chiSquareUpperTail', () => {
  it('is exactly the exponential tail on two degrees of freedom', () => {
    // The chi-square on 2 df is an exponential with mean 2, so its survival
    // function has a closed form to check against at every point.
    for (const x of [0, 0.5, 1, 3, 5.991464547107979, 20, 60]) {
      expect(chiSquareUpperTail(x, 2)).toBeCloseTo(Math.exp(-x / 2), 12);
    }
  });

  it('is twice the normal tail on one degree of freedom', () => {
    // A chi-square on 1 df is the square of a standard normal, so exceeding x
    // is the same event as |Z| exceeding sqrt(x).
    for (const x of [0.1, 1, 3.841458820694124, 10, 40]) {
      expect(chiSquareUpperTail(x, 1)).toBeCloseTo(normalTwoSidedTail(Math.sqrt(x)), 12);
    }
  });

  it('reproduces the five percent critical values', () => {
    const critical: Readonly<Record<number, number>> = {
      1: 3.841458820694124,
      2: 5.991464547107979,
      3: 7.814727903251179,
      4: 9.487729036781154,
      5: 11.070497693516351,
      10: 18.307038053275146,
    };
    for (const [df, value] of Object.entries(critical)) {
      expect(chiSquareUpperTail(value, Number(df))).toBeCloseTo(0.05, 10);
    }
  });

  it('keeps its significant digits in the far tail', () => {
    // Computed as 1 - P these would both round to zero. Read from the
    // continued fraction directly they carry full precision, which is what a
    // p-value reported in scientific notation needs.
    expect(chiSquareUpperTail(100, 1)).toBeCloseTo(1.5239706048320995e-23, 33);
    expect(chiSquareUpperTail(50, 3)).toBeCloseTo(7.989179244951495e-11, 20);
    expect(chiSquareUpperTail(100, 1)).toBeGreaterThan(0);
  });

  it('matches reference values at assorted degrees of freedom', () => {
    expect(chiSquareUpperTail(1.2, 4)).toBeCloseTo(0.8780986177504424, 10);
    expect(chiSquareUpperTail(9.5, 7)).toBeCloseTo(0.21872185112315715, 10);
  });

  it('is one at a zero statistic and decreasing in the statistic', () => {
    expect(chiSquareUpperTail(0, 3)).toBe(1);
    let previous = 1.1;
    for (let x = 0; x <= 30; x += 0.5) {
      const value = chiSquareUpperTail(x, 4);
      expect(value).toBeLessThanOrEqual(previous);
      previous = value;
    }
  });

  it('rejects a negative statistic or a non-integer degrees of freedom', () => {
    expect(() => chiSquareUpperTail(-1, 1)).toThrow(/must not be negative/);
    expect(() => chiSquareUpperTail(1, 0)).toThrow(/positive integer/);
    expect(() => chiSquareUpperTail(1, 1.5)).toThrow(/positive integer/);
  });
});

describe('the normal tails', () => {
  it('is a half at zero and symmetric about it', () => {
    expect(normalUpperTail(0)).toBeCloseTo(0.5, 14);
    for (const z of [0.3, 1, 2.5, 4]) {
      expect(normalUpperTail(z) + normalUpperTail(-z)).toBeCloseTo(1, 13);
    }
  });

  it('reproduces the familiar critical points', () => {
    expect(normalUpperTail(1.6448536269514722)).toBeCloseTo(0.05, 12);
    expect(normalUpperTail(1.959963984540054)).toBeCloseTo(0.025, 12);
    expect(normalUpperTail(1.96)).toBeCloseTo(0.024997895148220435, 12);
    expect(normalUpperTail(0.5)).toBeCloseTo(0.3085375387259869, 12);
  });

  it('stays accurate eight deviations out', () => {
    expect(normalUpperTail(8)).toBeCloseTo(6.22096057427174e-16, 24);
  });

  it('doubles into the two-sided tail', () => {
    for (const z of [-3, -0.7, 0, 0.7, 3]) {
      expect(normalTwoSidedTail(z)).toBeCloseTo(2 * normalUpperTail(Math.abs(z)), 14);
    }
    expect(normalTwoSidedTail(1.959963984540054)).toBeCloseTo(0.05, 12);
  });

  it('rejects a non-finite argument', () => {
    expect(() => normalUpperTail(Number.NaN)).toThrow(/normalUpperTail: z/);
  });
});
