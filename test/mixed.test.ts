import { describe, expect, it } from 'vitest';
import { linspace } from '../src/core/numeric.js';
import { makeItem, threePL, twoPL } from '../src/models/item.js';
import {
  categoryCountOf,
  categoryProbabilitiesOf,
  categoryProbabilityOf,
  expectedScoreOf,
  formatCounts,
  informationOf,
  isPolytomous,
  makePolytomousItem,
  maximumScoreOf,
  maximumTestScore,
  metricOf,
  requireValidCategory,
  testCharacteristicCurve,
  testInformationOf,
  type AnyItem,
} from '../src/models/mixed.js';
import { graded, generalizedPartialCredit, partialCredit } from '../src/models/polytomous.js';
import { itemInformation, probabilityCorrect, testInformation } from '../src/models/response.js';

const DICHOTOMOUS = makeItem('mc-1', twoPL(1.4, 0.2), { domain: 'algebra' });
const GUESSABLE = makeItem('mc-2', threePL(1.1, -0.3, 0.25), { domain: 'algebra' });
const ESSAY = makePolytomousItem('essay-1', graded(1.2, [-1.5, -0.2, 0.9]), {
  domain: 'writing',
  tags: ['rubric'],
});
const TASK = makePolytomousItem('task-1', generalizedPartialCredit(1.6, [-1, 0.4, 1.3]), {
  domain: 'modelling',
});
const FORM: readonly AnyItem[] = [DICHOTOMOUS, GUESSABLE, ESSAY, TASK];

const GRID = linspace(-4, 4, 33);

describe('format discrimination', () => {
  it('tells the formats apart by the shape of their parameters', () => {
    expect(isPolytomous(DICHOTOMOUS)).toBe(false);
    expect(isPolytomous(GUESSABLE)).toBe(false);
    expect(isPolytomous(ESSAY)).toBe(true);
    expect(isPolytomous(TASK)).toBe(true);
  });

  it('reports category counts and maxima for both formats', () => {
    expect(categoryCountOf(DICHOTOMOUS)).toBe(2);
    expect(maximumScoreOf(DICHOTOMOUS)).toBe(1);
    expect(categoryCountOf(ESSAY)).toBe(4);
    expect(maximumScoreOf(ESSAY)).toBe(3);
  });

  it('carries the metric through from either format', () => {
    expect(metricOf(DICHOTOMOUS)).toBe('logistic');
    expect(metricOf(makePolytomousItem('p', graded(1, [-1, 1], 'normal')))).toBe('normal');
  });
});

describe('polytomous item records', () => {
  it('keep identity and blueprint metadata', () => {
    expect(ESSAY.id).toBe('essay-1');
    expect(ESSAY.domain).toBe('writing');
    expect(ESSAY.tags).toEqual(['rubric']);
  });

  it('are frozen, parameters included', () => {
    expect(Object.isFrozen(ESSAY)).toBe(true);
    expect(Object.isFrozen(ESSAY.parameters)).toBe(true);
  });

  it('omit absent metadata rather than storing undefined', () => {
    const bare = makePolytomousItem('bare', partialCredit([-1, 1]));
    expect('domain' in bare).toBe(false);
    expect('tags' in bare).toBe(false);
  });

  it('reject an empty id and invalid parameters', () => {
    expect(() => makePolytomousItem('', graded(1, [-1, 1]))).toThrow(/non-empty string/);
    expect(() =>
      makePolytomousItem('bad', {
        a: 1,
        thresholds: [1, -1],
        model: 'graded',
        metric: 'logistic',
      }),
    ).toThrow(/strictly increasing/);
  });
});

describe('the dichotomous case of the mixed functions', () => {
  it('gives a dichotomous item the category distribution [1 - P, P]', () => {
    for (const theta of GRID) {
      const p = probabilityCorrect(DICHOTOMOUS.parameters, theta);
      expect(categoryProbabilitiesOf(DICHOTOMOUS, theta)).toEqual([1 - p, p]);
    }
  });

  it('reproduces the dichotomous information function exactly', () => {
    for (const item of [DICHOTOMOUS, GUESSABLE]) {
      for (const theta of GRID) {
        expect(informationOf(item, theta)).toBe(itemInformation(item.parameters, theta));
      }
    }
  });

  it('makes the expected score of a dichotomous item its probability correct', () => {
    for (const theta of GRID) {
      expect(expectedScoreOf(GUESSABLE, theta)).toBe(
        probabilityCorrect(GUESSABLE.parameters, theta),
      );
    }
  });

  it('agrees with the dichotomous test information over an all-dichotomous form', () => {
    const dichotomousOnly = [DICHOTOMOUS, GUESSABLE];
    for (const theta of GRID) {
      expect(testInformationOf(dichotomousOnly, theta)).toBeCloseTo(
        testInformation(dichotomousOnly, theta),
        14,
      );
    }
  });
});

describe('mixed-format information', () => {
  it('sums item information over both formats', () => {
    for (const theta of GRID) {
      const expected =
        informationOf(DICHOTOMOUS, theta) +
        informationOf(GUESSABLE, theta) +
        informationOf(ESSAY, theta) +
        informationOf(TASK, theta);
      expect(testInformationOf(FORM, theta)).toBeCloseTo(expected, 14);
    }
  });

  it('is non-negative and vanishes far outside the item locations', () => {
    for (const theta of GRID) expect(testInformationOf(FORM, theta)).toBeGreaterThanOrEqual(0);
    expect(testInformationOf(FORM, 25)).toBeLessThan(1e-6);
  });

  it('is zero over an empty form', () => {
    expect(testInformationOf([], 0)).toBe(0);
  });

  it('rejects a non-finite ability', () => {
    expect(() => testInformationOf(FORM, Number.NaN)).toThrow(/ability \(theta\)/);
    expect(() => testCharacteristicCurve(FORM, Number.NaN)).toThrow(/ability \(theta\)/);
  });
});

describe('the test characteristic curve', () => {
  it('rises monotonically from the guessing floor to the maximum total score', () => {
    // Not from zero: the 3PL item on this form contributes its lower asymptote
    // at any ability, so the floor of the whole curve is the sum of those.
    const floor = GUESSABLE.parameters.c;
    expect(testCharacteristicCurve(FORM, -30) - floor).toBeLessThan(0.02);
    expect(maximumTestScore(FORM) - testCharacteristicCurve(FORM, 30)).toBeLessThan(0.02);

    let previous = Number.NEGATIVE_INFINITY;
    for (const theta of GRID) {
      const score = testCharacteristicCurve(FORM, theta);
      expect(score).toBeGreaterThan(previous);
      previous = score;
    }
  });

  it('is bounded below by the guessing floor the form carries', () => {
    // The 3PL item contributes at least its lower asymptote at any ability, so
    // the whole curve sits above it however low the ability goes.
    const floor = GUESSABLE.parameters.c;
    for (const theta of GRID) {
      expect(testCharacteristicCurve(FORM, theta)).toBeGreaterThan(floor);
    }
  });

  it('sums the maximum score across formats', () => {
    // 1 + 1 from the dichotomous items, 3 + 3 from the polytomous ones.
    expect(maximumTestScore(FORM)).toBe(8);
    expect(maximumTestScore([])).toBe(0);
  });

  it('stays within [0, maximum] everywhere', () => {
    const top = maximumTestScore(FORM);
    for (const theta of GRID) {
      const score = testCharacteristicCurve(FORM, theta);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(top);
    }
  });
});

describe('category access and validation', () => {
  it('reads a single category probability for either format', () => {
    for (const theta of [-1, 0, 1.5]) {
      expect(categoryProbabilityOf(DICHOTOMOUS, theta, 1)).toBeCloseTo(
        probabilityCorrect(DICHOTOMOUS.parameters, theta),
        14,
      );
      const row = categoryProbabilitiesOf(ESSAY, theta);
      for (let k = 0; k < row.length; k += 1) {
        expect(categoryProbabilityOf(ESSAY, theta, k)).toBeCloseTo(row[k] as number, 14);
      }
    }
  });

  it('names the item when a category is out of range', () => {
    expect(() => categoryProbabilityOf(DICHOTOMOUS, 0, 2)).toThrow(/"mc-1"/);
    expect(() => categoryProbabilityOf(ESSAY, 0, 4)).toThrow(/"essay-1"/);
  });

  it('accepts every category an item can produce', () => {
    for (let k = 0; k <= maximumScoreOf(ESSAY); k += 1) {
      expect(requireValidCategory(ESSAY, k)).toBe(k);
    }
    expect(requireValidCategory(DICHOTOMOUS, 0)).toBe(0);
    expect(requireValidCategory(DICHOTOMOUS, 1)).toBe(1);
  });

  it('refuses a category the item cannot produce rather than clamping it', () => {
    expect(() => requireValidCategory(DICHOTOMOUS, 2)).toThrow(/in \[0, 1\]/);
    expect(() => requireValidCategory(ESSAY, 4)).toThrow(/in \[0, 3\]/);
    expect(() => requireValidCategory(ESSAY, -1)).toThrow(RangeError);
    expect(() => requireValidCategory(ESSAY, 1.5)).toThrow(RangeError);
    expect(() => requireValidCategory(ESSAY, Number.NaN)).toThrow(RangeError);
  });
});

describe('format counts', () => {
  it('reports the mix and the total score of a form', () => {
    expect(formatCounts(FORM)).toEqual({ dichotomous: 2, polytomous: 2, maximumScore: 8 });
  });

  it('handles single-format and empty forms', () => {
    expect(formatCounts([DICHOTOMOUS])).toEqual({
      dichotomous: 1,
      polytomous: 0,
      maximumScore: 1,
    });
    expect(formatCounts([ESSAY])).toEqual({ dichotomous: 0, polytomous: 1, maximumScore: 3 });
    expect(formatCounts([])).toEqual({ dichotomous: 0, polytomous: 0, maximumScore: 0 });
  });
});
