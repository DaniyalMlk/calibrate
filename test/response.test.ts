import { describe, expect, it } from 'vitest';
import { linspace, NORMAL_METRIC_SCALE } from '../src/core/numeric.js';
import { fourPL, makeItem, rasch, threePL, twoPL } from '../src/models/item.js';
import {
  expectedScore,
  informationPeak,
  itemInformation,
  probabilityCorrect,
  probabilityIncorrect,
  probabilityOfResponse,
  responseDerivative,
  standardError,
  testInformation,
} from '../src/models/response.js';

describe('response function, closed-form checks', () => {
  it('is exactly one half at theta = b for any model without asymptotes', () => {
    for (const b of [-2, -0.5, 0, 1.3, 3]) {
      expect(probabilityCorrect(twoPL(1.7, b), b)).toBeCloseTo(0.5, 15);
      expect(probabilityCorrect(rasch(b), b)).toBeCloseTo(0.5, 15);
    }
  });

  it('is exactly (1 + c) / 2 at theta = b for the 3PL', () => {
    for (const c of [0, 0.1, 0.25, 0.33]) {
      expect(probabilityCorrect(threePL(1.4, 0.6, c), 0.6)).toBeCloseTo((1 + c) / 2, 14);
    }
  });

  it('is exactly (c + d) / 2 at theta = b for the 4PL', () => {
    expect(probabilityCorrect(fourPL(1.4, -0.2, 0.2, 0.9), -0.2)).toBeCloseTo(0.55, 14);
  });

  it('reproduces a hand-computed 2PL value', () => {
    // a = 1.5, b = 0.5, theta = 1.0 -> exponent 0.75
    // P = 1 / (1 + e^-0.75) = 0.679178699175393
    expect(probabilityCorrect(twoPL(1.5, 0.5), 1)).toBeCloseTo(0.679178699175393, 12);
  });

  it('reproduces a hand-computed 3PL value', () => {
    // a = 1.0, b = 0.0, c = 0.2, theta = 1.0
    // P = 0.2 + 0.8 / (1 + e^-1) = 0.2 + 0.8 * 0.7310585786300049 = 0.784846862904004
    expect(probabilityCorrect(threePL(1, 0, 0.2), 1)).toBeCloseTo(0.784846862904004, 12);
  });

  it('approaches its asymptotes in the tails', () => {
    const item = fourPL(1.2, 0, 0.15, 0.93);
    expect(probabilityCorrect(item, -40)).toBeCloseTo(0.15, 12);
    expect(probabilityCorrect(item, 40)).toBeCloseTo(0.93, 12);
  });

  it('is strictly increasing in ability', () => {
    const item = threePL(0.8, 0.4, 0.2);
    const grid = linspace(-6, 6, 241);
    for (let i = 1; i < grid.length; i += 1) {
      const previous = probabilityCorrect(item, grid[i - 1] as number);
      const current = probabilityCorrect(item, grid[i] as number);
      expect(current).toBeGreaterThan(previous);
    }
  });

  it('is strictly decreasing in difficulty at fixed ability', () => {
    const easier = probabilityCorrect(twoPL(1, -1), 0);
    const harder = probabilityCorrect(twoPL(1, 1), 0);
    expect(easier).toBeGreaterThan(harder);
  });

  it('complements correctly', () => {
    const item = threePL(1.1, 0.3, 0.18);
    for (const theta of linspace(-4, 4, 33)) {
      expect(probabilityCorrect(item, theta) + probabilityIncorrect(item, theta)).toBeCloseTo(1, 14);
      expect(probabilityOfResponse(item, theta, 1)).toBeCloseTo(
        probabilityCorrect(item, theta),
        15,
      );
      expect(probabilityOfResponse(item, theta, 0)).toBeCloseTo(
        probabilityIncorrect(item, theta),
        15,
      );
    }
  });

  it('applies the normal metric scaling to the exponent', () => {
    const logisticItem = twoPL(1, 0, 'logistic');
    const normalItem = twoPL(1, 0, 'normal');
    expect(probabilityCorrect(normalItem, 1)).toBeCloseTo(
      probabilityCorrect(logisticItem, NORMAL_METRIC_SCALE),
      14,
    );
  });

  it('rejects a non-finite ability', () => {
    expect(() => probabilityCorrect(twoPL(1, 0), Number.NaN)).toThrow(RangeError);
  });
});

describe('response derivative', () => {
  it('agrees with a central finite difference across models', () => {
    const items = [rasch(0.2), twoPL(1.6, -0.4), threePL(1.1, 0.5, 0.25), fourPL(0.9, 0, 0.2, 0.94)];
    const h = 1e-6;
    for (const item of items) {
      for (const theta of [-2, -0.5, 0, 0.7, 2.5]) {
        const numeric =
          (probabilityCorrect(item, theta + h) - probabilityCorrect(item, theta - h)) / (2 * h);
        expect(responseDerivative(item, theta)).toBeCloseTo(numeric, 7);
      }
    }
  });

  it('peaks at theta = b with height D * a * (d - c) / 4', () => {
    const item = fourPL(1.3, 0.4, 0.2, 0.9);
    const peak = responseDerivative(item, 0.4);
    expect(peak).toBeCloseTo((1.3 * (0.9 - 0.2)) / 4, 14);
    expect(responseDerivative(item, 0.9)).toBeLessThan(peak);
    expect(responseDerivative(item, -0.1)).toBeLessThan(peak);
  });
});

describe('item information', () => {
  it('matches the 2PL shortcut D^2 a^2 P Q', () => {
    const item = twoPL(1.45, -0.2);
    for (const theta of linspace(-4, 4, 41)) {
      const p = probabilityCorrect(item, theta);
      const shortcut = item.a * item.a * p * (1 - p);
      expect(itemInformation(item, theta)).toBeCloseTo(shortcut, 12);
    }
  });

  it('matches the Birnbaum 3PL form D^2 a^2 (Q/P) [(P - c)/(1 - c)]^2', () => {
    const item = threePL(1.2, 0.3, 0.22);
    for (const theta of linspace(-4, 4, 41)) {
      const p = probabilityCorrect(item, theta);
      const q = 1 - p;
      const birnbaum =
        item.a * item.a * (q / p) * Math.pow((p - item.c) / (1 - item.c), 2);
      expect(itemInformation(item, theta)).toBeCloseTo(birnbaum, 12);
    }
  });

  it('peaks at theta = b with height a^2 / 4 for the 2PL', () => {
    for (const a of [0.5, 1, 1.7, 2.4]) {
      const item = twoPL(a, 0.75);
      expect(informationPeak(item)).toBeCloseTo(0.75, 9);
      expect(itemInformation(item, 0.75)).toBeCloseTo((a * a) / 4, 13);
    }
  });

  it('places the 3PL information peak strictly above b', () => {
    const item = threePL(1.3, 0.2, 0.25);
    const peak = informationPeak(item);
    expect(peak).toBeGreaterThan(0.2);
    // Closed form: b + ln[(1 + sqrt(1 + 8c)) / 2] / (D a)
    const expected = 0.2 + Math.log((1 + Math.sqrt(1 + 8 * 0.25)) / 2) / 1.3;
    expect(peak).toBeCloseTo(expected, 9);
  });

  it('finds the 4PL peak numerically and it dominates its neighbourhood', () => {
    const item = fourPL(1.1, -0.3, 0.2, 0.9);
    const peak = informationPeak(item);
    const atPeak = itemInformation(item, peak);
    for (const offset of [-1.5, -0.4, -0.05, 0.05, 0.4, 1.5]) {
      expect(itemInformation(item, peak + offset)).toBeLessThanOrEqual(atPeak + 1e-12);
    }
  });

  it('is reduced by guessing at the point where the guess-free item is best', () => {
    const clean = twoPL(1.5, 0);
    const guessy = threePL(1.5, 0, 0.25);
    expect(itemInformation(guessy, 0)).toBeLessThan(itemInformation(clean, 0));
  });

  it('is never negative and decays to zero in the tails', () => {
    const item = threePL(1.2, 0, 0.2);
    for (const theta of linspace(-8, 8, 81)) {
      expect(itemInformation(item, theta)).toBeGreaterThanOrEqual(0);
    }
    expect(itemInformation(item, -200)).toBeCloseTo(0, 12);
    expect(itemInformation(item, 200)).toBeCloseTo(0, 12);
  });

  it('returns zero rather than NaN once P * Q underflows', () => {
    // For a 2PL item the response probability itself underflows to exactly 0
    // in the far tail, so the naive ratio would be 0 / 0.
    const item = twoPL(1.2, 0);
    expect(itemInformation(item, -2000)).toBe(0);
    expect(itemInformation(item, 2000)).toBe(0);
    expect(Number.isNaN(itemInformation(item, -2000))).toBe(false);
  });

  it('scales with the square of discrimination', () => {
    const single = itemInformation(twoPL(1, 0), 0);
    const double = itemInformation(twoPL(2, 0), 0);
    expect(double / single).toBeCloseTo(4, 12);
  });
});

describe('test information and standard error', () => {
  const bank = [
    makeItem('a', twoPL(1.2, -1)),
    makeItem('b', twoPL(1.0, 0)),
    makeItem('c', twoPL(1.4, 1)),
  ];

  it('is additive over items', () => {
    const theta = 0.3;
    const manual =
      itemInformation(bank[0]!.parameters, theta) +
      itemInformation(bank[1]!.parameters, theta) +
      itemInformation(bank[2]!.parameters, theta);
    expect(testInformation(bank, theta)).toBeCloseTo(manual, 14);
  });

  it('is zero for an empty test', () => {
    expect(testInformation([], 0)).toBe(0);
  });

  it('converts to a standard error of 1 / sqrt(I)', () => {
    expect(standardError(4)).toBe(0.5);
    expect(standardError(1)).toBe(1);
    expect(standardError(0)).toBe(Number.POSITIVE_INFINITY);
    expect(() => standardError(-1)).toThrow(RangeError);
  });

  it('tightens the standard error as items accumulate', () => {
    const one = standardError(testInformation(bank.slice(0, 1), 0));
    const all = standardError(testInformation(bank, 0));
    expect(all).toBeLessThan(one);
  });
});

describe('expected score', () => {
  const bank = [makeItem('a', twoPL(1, -1)), makeItem('b', twoPL(1, 0)), makeItem('c', twoPL(1, 1))];

  it('is the sum of response probabilities', () => {
    // At theta = 0 the three items give P = sigma(1), 0.5, sigma(-1); the outer
    // two are symmetric so the total is exactly 1.5.
    expect(expectedScore(bank, 0)).toBeCloseTo(1.5, 13);
  });

  it('is monotone in ability and bounded by the test length', () => {
    expect(expectedScore(bank, -3)).toBeLessThan(expectedScore(bank, 0));
    expect(expectedScore(bank, 0)).toBeLessThan(expectedScore(bank, 3));
    expect(expectedScore(bank, 40)).toBeCloseTo(3, 10);
    expect(expectedScore(bank, -40)).toBeCloseTo(0, 10);
  });
});
