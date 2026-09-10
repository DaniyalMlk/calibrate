import { describe, expect, it } from 'vitest';
import { linspace } from '../src/core/numeric.js';
import { rasch, twoPL } from '../src/models/item.js';
import {
  boundaryProbability,
  categoryCount,
  categoryDerivatives,
  categoryProbabilities,
  categoryProbability,
  categoryScoreVariance,
  categorySecondDerivatives,
  describePolytomousModel,
  expectedCategoryScore,
  generalizedPartialCredit,
  graded,
  maximumScore,
  partialCredit,
  partialCreditInformation,
  polytomousInformation,
  polytomousInformationPeak,
  validatePolytomousParameters,
  type PolytomousParameters,
} from '../src/models/polytomous.js';
import { itemInformation, probabilityCorrect } from '../src/models/response.js';

/** A representative item from each family, used across the shared invariants. */
const FAMILIES: readonly PolytomousParameters[] = [
  graded(1.2, [-2, -0.5, 0.4, 1.8]),
  partialCredit([-1.1, 0.2, 1.4]),
  generalizedPartialCredit(1.7, [-1.5, 0.2, -0.3, 1.1]),
];

const GRID = linspace(-4, 4, 33);

/** Central difference, accurate to O(h^2). */
function centralDifference(f: (x: number) => number, x: number, h = 1e-5): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}

/** Second central difference, accurate to O(h^2). */
function secondDifference(f: (x: number) => number, x: number, h = 1e-4): number {
  return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}

describe('polytomous constructors', () => {
  it('pin the parameters each model holds fixed', () => {
    expect(graded(1.3, [-1, 1])).toMatchObject({ a: 1.3, model: 'graded' });
    expect(partialCredit([-1, 1])).toMatchObject({ a: 1, model: 'partial-credit' });
    expect(generalizedPartialCredit(2.1, [-1, 1])).toMatchObject({
      a: 2.1,
      model: 'generalized-partial-credit',
    });
  });

  it('score into one more category than they have thresholds', () => {
    expect(categoryCount(graded(1, [-1, 0, 1]))).toBe(4);
    expect(maximumScore(graded(1, [-1, 0, 1]))).toBe(3);
    expect(categoryCount(partialCredit([0]))).toBe(2);
  });

  it('default to the logistic metric and freeze what they return', () => {
    const parameters = graded(1, [-1, 1]);
    expect(parameters.metric).toBe('logistic');
    expect(graded(1, [-1, 1], 'normal').metric).toBe('normal');
    expect(Object.isFrozen(parameters)).toBe(true);
    expect(Object.isFrozen(parameters.thresholds)).toBe(true);
  });

  it('copy the thresholds rather than aliasing the caller array', () => {
    const supplied = [-1, 0, 1];
    const parameters = graded(1, supplied);
    supplied[1] = 99;
    expect(parameters.thresholds[1]).toBe(0);
  });

  it('name the model compactly for reports', () => {
    expect(describePolytomousModel(graded(1, [-1, 0, 1]))).toBe('GRM(4)');
    expect(describePolytomousModel(partialCredit([-1, 1]))).toBe('PCM(3)');
    expect(describePolytomousModel(generalizedPartialCredit(2, [0]))).toBe('GPCM(2)');
  });
});

describe('polytomous validation', () => {
  const base = {
    a: 1,
    thresholds: [-1, 0, 1],
    model: 'graded' as const,
    metric: 'logistic' as const,
  };

  it('names the offending field', () => {
    expect(() => validatePolytomousParameters({ ...base, a: Number.NaN })).toThrow(
      /discrimination \(a\)/,
    );
    expect(() =>
      validatePolytomousParameters({ ...base, thresholds: [-1, Number.NaN, 1] }),
    ).toThrow(/threshold 2/);
  });

  it('rejects non-positive and implausibly large discriminations', () => {
    expect(() => validatePolytomousParameters({ ...base, a: 0 })).toThrow(RangeError);
    expect(() => validatePolytomousParameters({ ...base, a: -1 })).toThrow(RangeError);
    expect(() => validatePolytomousParameters({ ...base, a: 25 })).toThrow(/step function/);
    expect(() => validatePolytomousParameters({ ...base, a: 20 })).not.toThrow();
  });

  it('rejects thresholds outside the admissible window', () => {
    expect(() => validatePolytomousParameters({ ...base, thresholds: [-1, 0, 21] })).toThrow(
      RangeError,
    );
    expect(() => validatePolytomousParameters({ ...base, thresholds: [-21, 0, 1] })).toThrow(
      RangeError,
    );
  });

  it('requires at least one threshold', () => {
    expect(() => validatePolytomousParameters({ ...base, thresholds: [] })).toThrow(
      /at least one threshold/,
    );
  });

  it('rejects an unknown model or metric', () => {
    expect(() =>
      validatePolytomousParameters({
        ...base,
        model: 'nominal' as unknown as PolytomousParameters['model'],
      }),
    ).toThrow(/model must be/);
    expect(() =>
      validatePolytomousParameters({
        ...base,
        metric: 'probit' as unknown as PolytomousParameters['metric'],
      }),
    ).toThrow(/metric must be/);
  });

  it('caps the number of categories', () => {
    const many = linspace(-5, 5, 64);
    expect(() => validatePolytomousParameters({ ...base, thresholds: many })).toThrow(
      /at most 64 categories/,
    );
  });

  it('holds the partial credit discrimination at one', () => {
    expect(() =>
      validatePolytomousParameters({ ...base, model: 'partial-credit', a: 1.5 }),
    ).toThrow(/fixes discrimination at 1/);
    expect(() =>
      validatePolytomousParameters({ ...base, model: 'partial-credit', a: 1 }),
    ).not.toThrow();
  });

  it('requires strictly increasing thresholds under the graded model', () => {
    expect(() => validatePolytomousParameters({ ...base, thresholds: [1, 0, -1] })).toThrow(
      /strictly increasing/,
    );
    expect(() => validatePolytomousParameters({ ...base, thresholds: [-1, 0, 0] })).toThrow(
      /strictly increasing/,
    );
  });

  it('permits threshold reversals under the partial credit models', () => {
    // A reversal is a real result: it says the category is never modal.
    const reversed = partialCredit([1, -1, 2]);
    const probabilities = categoryProbabilities(reversed, 0);
    for (const p of probabilities) expect(p).toBeGreaterThan(0);
    expect(probabilities.reduce((total, p) => total + p, 0)).toBeCloseTo(1, 14);

    // Confirm the substantive claim: the reversed category never wins anywhere.
    const everModal = GRID.some((theta) => {
      const row = categoryProbabilities(reversed, theta);
      const best = Math.max(...row);
      return row[1] === best;
    });
    expect(everModal).toBe(false);
  });
});

describe('category probabilities', () => {
  it('form a distribution over categories at every ability', () => {
    for (const parameters of FAMILIES) {
      for (const theta of GRID) {
        const probabilities = categoryProbabilities(parameters, theta);
        expect(probabilities).toHaveLength(categoryCount(parameters));
        for (const p of probabilities) {
          expect(p).toBeGreaterThanOrEqual(0);
          expect(p).toBeLessThanOrEqual(1);
        }
        expect(probabilities.reduce((total, p) => total + p, 0)).toBeCloseTo(1, 12);
      }
    }
  });

  it('stay a proper distribution at abilities that overflow the naive forms', () => {
    // exp() of the running partial credit exponent overflows here without the
    // softmax shift, and the graded differences cancel to nothing without expm1.
    const wide = generalizedPartialCredit(2, linspace(-4, 4, 12));
    for (const theta of [-20, -15, 15, 20]) {
      for (const parameters of [...FAMILIES, wide]) {
        const probabilities = categoryProbabilities(parameters, theta);
        for (const p of probabilities) expect(Number.isFinite(p)).toBe(true);
        expect(probabilities.reduce((total, p) => total + p, 0)).toBeCloseTo(1, 12);
      }
    }
  });

  it('put all their mass in the extreme categories in the tails', () => {
    // Not exactly all of it: at theta = -20 a unit-slope step at -1.1 still
    // leaves exp(-18.9) ~ 6e-9 in the category above. The assertion is that the
    // leakage is of that order and no larger.
    for (const parameters of FAMILIES) {
      const top = categoryCount(parameters) - 1;
      expect(1 - categoryProbability(parameters, -20, 0)).toBeLessThan(1e-7);
      expect(1 - categoryProbability(parameters, 20, top)).toBeLessThan(1e-7);
    }
  });

  it('reject a category index outside the item', () => {
    const parameters = graded(1, [-1, 0, 1]);
    expect(() => categoryProbability(parameters, 0, 4)).toThrow(/category must be/);
    expect(() => categoryProbability(parameters, 0, -1)).toThrow(/category must be/);
    expect(() => categoryProbability(parameters, 0, 1.5)).toThrow(/category must be/);
  });

  it('reject a non-finite ability', () => {
    expect(() => categoryProbabilities(FAMILIES[0] as PolytomousParameters, Number.NaN)).toThrow(
      /ability \(theta\)/,
    );
    expect(() =>
      categoryProbabilities(FAMILIES[0] as PolytomousParameters, Number.POSITIVE_INFINITY),
    ).toThrow(/ability \(theta\)/);
  });
});

describe('reduction to the dichotomous models', () => {
  it('makes a two-category graded item a 2PL item exactly', () => {
    const polytomous = graded(1.4, [0.3]);
    const dichotomous = twoPL(1.4, 0.3);
    for (const theta of GRID) {
      const probabilities = categoryProbabilities(polytomous, theta);
      expect(probabilities[1]).toBeCloseTo(probabilityCorrect(dichotomous, theta), 14);
      expect(probabilities[0]).toBeCloseTo(1 - probabilityCorrect(dichotomous, theta), 14);
      expect(polytomousInformation(polytomous, theta)).toBeCloseTo(
        itemInformation(dichotomous, theta),
        14,
      );
    }
  });

  it('makes a two-category partial credit item a Rasch item exactly', () => {
    const polytomous = partialCredit([0.7]);
    const dichotomous = rasch(0.7);
    for (const theta of GRID) {
      expect(categoryProbability(polytomous, theta, 1)).toBeCloseTo(
        probabilityCorrect(dichotomous, theta),
        14,
      );
      expect(polytomousInformation(polytomous, theta)).toBeCloseTo(
        itemInformation(dichotomous, theta),
        14,
      );
    }
  });

  it('agrees across families when a two-category item has the same slope', () => {
    // With one threshold the graded and generalized partial credit forms are
    // the same curve written two ways, so they must coincide exactly.
    const asGraded = graded(1.6, [-0.4]);
    const asGpcm = generalizedPartialCredit(1.6, [-0.4]);
    for (const theta of GRID) {
      expect(categoryProbability(asGraded, theta, 1)).toBeCloseTo(
        categoryProbability(asGpcm, theta, 1),
        14,
      );
    }
  });

  it('respects the metric, matching the dichotomous scaling', () => {
    const polytomous = graded(1.1, [0.2], 'normal');
    const dichotomous = twoPL(1.1, 0.2, 'normal');
    for (const theta of GRID) {
      expect(categoryProbability(polytomous, theta, 1)).toBeCloseTo(
        probabilityCorrect(dichotomous, theta),
        14,
      );
    }
  });
});

describe('threshold identities', () => {
  it('puts the graded cumulative probability at exactly one half on each boundary', () => {
    const parameters = graded(1.3, [-1.7, -0.2, 0.9, 2.1]);
    parameters.thresholds.forEach((threshold, index) => {
      const k = index + 1;
      const probabilities = categoryProbabilities(parameters, threshold);
      let atOrAbove = 0;
      for (let category = k; category < probabilities.length; category += 1) {
        atOrAbove += probabilities[category] as number;
      }
      expect(atOrAbove).toBeCloseTo(0.5, 13);
      expect(boundaryProbability(parameters, threshold, k)).toBeCloseTo(0.5, 14);
    });
  });

  it('equalises adjacent categories at each partial credit step', () => {
    for (const parameters of [partialCredit([-1, 0.5, 1.4]), generalizedPartialCredit(1.9, [-2, 0, 1])]) {
      parameters.thresholds.forEach((threshold, index) => {
        const k = index + 1;
        const probabilities = categoryProbabilities(parameters, threshold);
        expect(probabilities[k - 1]).toBeCloseTo(probabilities[k] as number, 13);
      });
    }
  });

  it('fixes the graded boundaries at the ends by definition', () => {
    const parameters = graded(1, [-1, 0, 1]);
    expect(boundaryProbability(parameters, 0.3, 0)).toBe(1);
    expect(boundaryProbability(parameters, 0.3, categoryCount(parameters))).toBe(0);
  });

  it('makes graded boundaries decrease in the category index', () => {
    const parameters = graded(1.2, [-2, -0.5, 0.4, 1.8]);
    for (const theta of GRID) {
      for (let k = 0; k < categoryCount(parameters); k += 1) {
        expect(boundaryProbability(parameters, theta, k)).toBeGreaterThanOrEqual(
          boundaryProbability(parameters, theta, k + 1),
        );
      }
    }
  });

  it('refuses boundary probabilities for the partial credit models', () => {
    expect(() => boundaryProbability(partialCredit([-1, 1]), 0, 1)).toThrow(
      /graded response model only/,
    );
  });

  it('rejects a boundary index outside the item', () => {
    const parameters = graded(1, [-1, 0, 1]);
    expect(() => boundaryProbability(parameters, 0, 5)).toThrow(/boundary index/);
    expect(() => boundaryProbability(parameters, 0, -1)).toThrow(/boundary index/);
  });
});

describe('category derivatives', () => {
  it('match central differences across the ability range', () => {
    for (const parameters of FAMILIES) {
      for (const theta of linspace(-3, 3, 13)) {
        const analytic = categoryDerivatives(parameters, theta);
        for (let k = 0; k < analytic.length; k += 1) {
          const numeric = centralDifference(
            (x) => categoryProbabilities(parameters, x)[k] as number,
            theta,
          );
          expect(analytic[k]).toBeCloseTo(numeric, 8);
        }
      }
    }
  });

  it('sum to zero, because the probabilities always sum to one', () => {
    for (const parameters of FAMILIES) {
      for (const theta of GRID) {
        const total = categoryDerivatives(parameters, theta).reduce((acc, d) => acc + d, 0);
        expect(total).toBeCloseTo(0, 12);
      }
    }
  });

  it('push the lowest category down and the highest category up', () => {
    for (const parameters of FAMILIES) {
      for (const theta of linspace(-2, 2, 9)) {
        const derivatives = categoryDerivatives(parameters, theta);
        expect(derivatives[0]).toBeLessThan(0);
        expect(derivatives[derivatives.length - 1]).toBeGreaterThan(0);
      }
    }
  });

  it('match second central differences across the ability range', () => {
    for (const parameters of FAMILIES) {
      for (const theta of linspace(-3, 3, 13)) {
        const analytic = categorySecondDerivatives(parameters, theta);
        for (let k = 0; k < analytic.length; k += 1) {
          const numeric = secondDifference(
            (x) => categoryProbabilities(parameters, x)[k] as number,
            theta,
          );
          expect(analytic[k]).toBeCloseTo(numeric, 5);
        }
      }
    }
  });

  it('sum the second derivatives to zero as well', () => {
    for (const parameters of FAMILIES) {
      for (const theta of GRID) {
        const total = categorySecondDerivatives(parameters, theta).reduce((acc, d) => acc + d, 0);
        expect(total).toBeCloseTo(0, 11);
      }
    }
  });

  it('rejects a non-finite ability', () => {
    expect(() => categoryDerivatives(FAMILIES[0] as PolytomousParameters, Number.NaN)).toThrow(
      /ability \(theta\)/,
    );
    expect(() =>
      categorySecondDerivatives(FAMILIES[0] as PolytomousParameters, Number.NaN),
    ).toThrow(/ability \(theta\)/);
  });
});

describe('expected category score', () => {
  it('rises monotonically from zero to the maximum score', () => {
    for (const parameters of FAMILIES) {
      const top = maximumScore(parameters);
      expect(expectedCategoryScore(parameters, -20)).toBeLessThan(1e-7);
      expect(top - expectedCategoryScore(parameters, 20)).toBeLessThan(1e-7);

      let previous = Number.NEGATIVE_INFINITY;
      for (const theta of GRID) {
        const score = expectedCategoryScore(parameters, theta);
        expect(score).toBeGreaterThan(previous);
        previous = score;
      }
    }
  });

  it('agrees with the probability of a correct answer for a two-category item', () => {
    const parameters = graded(1.5, [0.4]);
    const dichotomous = twoPL(1.5, 0.4);
    for (const theta of GRID) {
      expect(expectedCategoryScore(parameters, theta)).toBeCloseTo(
        probabilityCorrect(dichotomous, theta),
        14,
      );
    }
  });

  it('has the sum of the category derivatives weighted by score as its slope', () => {
    for (const parameters of FAMILIES) {
      for (const theta of linspace(-3, 3, 13)) {
        const slope = centralDifference((x) => expectedCategoryScore(parameters, x), theta);
        const analytic = categoryDerivatives(parameters, theta).reduce(
          (acc, d, k) => acc + k * d,
          0,
        );
        expect(analytic).toBeCloseTo(slope, 8);
      }
    }
  });
});

describe('polytomous information', () => {
  it('matches the closed form for the partial credit models', () => {
    // Substituting P'_k = D a P_k (k - E[X]) into sum_k (P'_k)^2 / P_k leaves
    // D^2 a^2 Var(X), so the general sum and the variance form must agree.
    for (const parameters of [FAMILIES[1], FAMILIES[2]] as PolytomousParameters[]) {
      for (const theta of GRID) {
        expect(polytomousInformation(parameters, theta)).toBeCloseTo(
          partialCreditInformation(parameters, theta),
          12,
        );
      }
    }
  });

  it('scales quadratically with discrimination under the partial credit model', () => {
    const thresholds = [-1, 0.3, 1.2];
    const single = generalizedPartialCredit(1, thresholds);
    const double = generalizedPartialCredit(2, thresholds);
    // Var(X) is not itself invariant to the slope, so compare at the point where
    // the curves coincide: the same variance times four times the squared slope.
    const theta = 0.5;
    const ratio =
      partialCreditInformation(double, theta) /
      (categoryScoreVariance(double, theta) * 4);
    expect(ratio).toBeCloseTo(1, 12);
    expect(partialCreditInformation(single, theta)).toBeCloseTo(
      categoryScoreVariance(single, theta),
      12,
    );
  });

  it('has no variance form for the graded model', () => {
    expect(() => partialCreditInformation(graded(1, [-1, 1]), 0)).toThrow(
      /no variance form/,
    );
  });

  it('is non-negative everywhere and vanishes in the tails', () => {
    for (const parameters of FAMILIES) {
      for (const theta of GRID) {
        expect(polytomousInformation(parameters, theta)).toBeGreaterThanOrEqual(0);
      }
      expect(polytomousInformation(parameters, -25)).toBeCloseTo(0, 6);
      expect(polytomousInformation(parameters, 25)).toBeCloseTo(0, 6);
    }
  });

  it('equals the negative expected curvature of the log-likelihood', () => {
    // Fisher information is E[-d2 log P / dtheta2] over categories, which is
    // an independent route to the same number:
    //   sum_k P_k * [ (P'_k / P_k)^2 - P''_k / P_k ]
    for (const parameters of FAMILIES) {
      for (const theta of linspace(-2.5, 2.5, 11)) {
        const probabilities = categoryProbabilities(parameters, theta);
        const first = categoryDerivatives(parameters, theta);
        const second = categorySecondDerivatives(parameters, theta);
        let expected = 0;
        for (let k = 0; k < probabilities.length; k += 1) {
          const p = probabilities[k] as number;
          if (p <= 1e-280) continue;
          const d = first[k] as number;
          const dd = second[k] as number;
          expected += p * ((d / p) ** 2 - dd / p);
        }
        expect(expected).toBeCloseTo(polytomousInformation(parameters, theta), 9);
      }
    }
  });

  it('carries more information than a dichotomous item of the same slope', () => {
    // Four ordered categories resolve more of the ability continuum than a
    // single cut through the middle of the same range.
    const polytomous = graded(1.4, [-1.5, -0.5, 0.5, 1.5]);
    const dichotomous = twoPL(1.4, 0);
    expect(polytomousInformation(polytomous, 0)).toBeGreaterThan(
      itemInformation(dichotomous, 0),
    );
  });

  it('finds a peak where the information is at least as large as anywhere nearby', () => {
    for (const parameters of FAMILIES) {
      const peak = polytomousInformationPeak(parameters);
      const best = polytomousInformation(parameters, peak);
      for (const offset of [-0.5, -0.1, -0.01, 0.01, 0.1, 0.5]) {
        expect(best).toBeGreaterThanOrEqual(
          polytomousInformation(parameters, peak + offset) - 1e-12,
        );
      }
    }
  });

  it('peaks at the threshold for a two-category item, matching the dichotomous case', () => {
    expect(polytomousInformationPeak(graded(1.5, [0.6]))).toBeCloseTo(0.6, 6);
    expect(polytomousInformationPeak(partialCredit([-0.9]))).toBeCloseTo(-0.9, 6);
  });
});

describe('category score variance', () => {
  it('is the variance of the category distribution', () => {
    for (const parameters of FAMILIES) {
      for (const theta of linspace(-2, 2, 9)) {
        const probabilities = categoryProbabilities(parameters, theta);
        const expected = expectedCategoryScore(parameters, theta);
        let direct = 0;
        for (let k = 0; k < probabilities.length; k += 1) {
          direct += (probabilities[k] as number) * (k - expected) ** 2;
        }
        expect(categoryScoreVariance(parameters, theta)).toBeCloseTo(direct, 12);
      }
    }
  });

  it('collapses to zero in the tails, where the outcome is certain', () => {
    for (const parameters of FAMILIES) {
      expect(categoryScoreVariance(parameters, -20)).toBeLessThan(1e-7);
      expect(categoryScoreVariance(parameters, 20)).toBeLessThan(1e-7);
    }
  });
});
