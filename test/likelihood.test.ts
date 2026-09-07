import { describe, expect, it } from 'vitest';
import { linspace } from '../src/core/numeric.js';
import { makeItem, rasch, threePL, twoPL } from '../src/models/item.js';
import { probabilityCorrect, type ScoredResponse } from '../src/models/response.js';
import {
  expectedInformation,
  logLikelihood,
  observedInformation,
  patternBoundedness,
  scoreFunction,
} from '../src/estimation/likelihood.js';

const twoPlPattern: ScoredResponse[] = [
  { item: makeItem('a', twoPL(1.2, -1)), response: 1 },
  { item: makeItem('b', twoPL(0.9, 0)), response: 1 },
  { item: makeItem('c', twoPL(1.5, 1)), response: 0 },
  { item: makeItem('d', twoPL(1.1, 1.5)), response: 0 },
];

const threePlPattern: ScoredResponse[] = [
  { item: makeItem('a', threePL(1.2, -1, 0.2)), response: 1 },
  { item: makeItem('b', threePL(0.9, 0, 0.25)), response: 0 },
  { item: makeItem('c', threePL(1.5, 1, 0.2)), response: 1 },
  { item: makeItem('d', threePL(1.1, 1.5, 0.25)), response: 0 },
];

describe('logLikelihood', () => {
  it('equals the hand-summed log probabilities', () => {
    const theta = 0.4;
    let expected = 0;
    for (const { item, response } of twoPlPattern) {
      const p = probabilityCorrect(item.parameters, theta);
      expected += Math.log(response === 1 ? p : 1 - p);
    }
    expect(logLikelihood(twoPlPattern, theta)).toBeCloseTo(expected, 13);
  });

  it('is zero for an empty pattern', () => {
    expect(logLikelihood([], 0)).toBe(0);
  });

  it('stays finite where a response probability underflows', () => {
    // A correct answer to a very hard, very discriminating item, evaluated far
    // below its difficulty: P underflows to exactly zero.
    const pattern: ScoredResponse[] = [{ item: makeItem('x', twoPL(2, 5)), response: 1 }];
    const value = logLikelihood(pattern, -400);
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeLessThan(-600);
  });

  it('prefers the ability that generated the pattern', () => {
    // The pattern above is two correct on easy items, two incorrect on hard
    // ones, so the likelihood should favour the middle of the range.
    const middle = logLikelihood(twoPlPattern, 0.3);
    expect(middle).toBeGreaterThan(logLikelihood(twoPlPattern, -3));
    expect(middle).toBeGreaterThan(logLikelihood(twoPlPattern, 3));
  });
});

describe('scoreFunction', () => {
  it('agrees with a central finite difference of the log-likelihood', () => {
    const h = 1e-5;
    for (const pattern of [twoPlPattern, threePlPattern]) {
      for (const theta of linspace(-3, 3, 25)) {
        const numeric =
          (logLikelihood(pattern, theta + h) - logLikelihood(pattern, theta - h)) / (2 * h);
        expect(scoreFunction(pattern, theta)).toBeCloseTo(numeric, 6);
      }
    }
  });

  it('is positive below the optimum and negative above it', () => {
    expect(scoreFunction(twoPlPattern, -2)).toBeGreaterThan(0);
    expect(scoreFunction(twoPlPattern, 2)).toBeLessThan(0);
  });

  it('is a sum of residuals: an item answered exactly as predicted contributes nothing', () => {
    // A Rasch item at theta = b predicts P = 0.5; a correct and an incorrect
    // response to two such items cancel exactly.
    const balanced: ScoredResponse[] = [
      { item: makeItem('a', rasch(0)), response: 1 },
      { item: makeItem('b', rasch(0)), response: 0 },
    ];
    expect(scoreFunction(balanced, 0)).toBeCloseTo(0, 14);
  });

  it('is zero for an empty pattern', () => {
    expect(scoreFunction([], 0)).toBe(0);
  });
});

describe('observedInformation', () => {
  it('agrees with a central finite difference of the score function', () => {
    const h = 1e-5;
    for (const pattern of [twoPlPattern, threePlPattern]) {
      for (const theta of linspace(-3, 3, 25)) {
        const numeric = (scoreFunction(pattern, theta + h) - scoreFunction(pattern, theta - h)) / (2 * h);
        expect(observedInformation(pattern, theta)).toBeCloseTo(-numeric, 5);
      }
    }
  });

  it('equals the expected information for 1PL and 2PL patterns', () => {
    // The logistic link is canonical for these models, so the second derivative
    // of the log-likelihood does not depend on the responses.
    const raschPattern: ScoredResponse[] = [
      { item: makeItem('a', rasch(-0.5)), response: 1 },
      { item: makeItem('b', rasch(0.5)), response: 0 },
    ];
    for (const pattern of [twoPlPattern, raschPattern]) {
      for (const theta of linspace(-3, 3, 25)) {
        expect(observedInformation(pattern, theta)).toBeCloseTo(
          expectedInformation(pattern, theta),
          11,
        );
      }
    }
  });

  it('differs from the expected information under the 3PL', () => {
    const differences = linspace(-3, 3, 25).map((theta) =>
      Math.abs(observedInformation(threePlPattern, theta) - expectedInformation(threePlPattern, theta)),
    );
    expect(Math.max(...differences)).toBeGreaterThan(0.01);
  });

  it('can go negative under the 3PL, which is why a raw Newton step is unsafe', () => {
    // A correct answer to a hard 3PL item, evaluated well below its difficulty,
    // sits on the convex part of the log-likelihood.
    const pattern: ScoredResponse[] = [{ item: makeItem('x', threePL(1.8, 2, 0.25)), response: 1 }];
    const values = linspace(-3, 3, 61).map((theta) => observedInformation(pattern, theta));
    expect(Math.min(...values)).toBeLessThan(0);
  });
});

describe('expectedInformation', () => {
  it('is never negative', () => {
    for (const theta of linspace(-5, 5, 41)) {
      expect(expectedInformation(threePlPattern, theta)).toBeGreaterThanOrEqual(0);
    }
  });

  it('does not depend on the responses given', () => {
    const flipped = threePlPattern.map((r) => ({ item: r.item, response: (1 - r.response) as 0 | 1 }));
    expect(expectedInformation(flipped, 0.3)).toBeCloseTo(expectedInformation(threePlPattern, 0.3), 14);
  });
});

describe('patternBoundedness', () => {
  const item = makeItem('a', twoPL(1, 0));
  const other = makeItem('b', twoPL(1, 1));

  it('flags all-correct and all-incorrect patterns', () => {
    expect(
      patternBoundedness([
        { item, response: 1 },
        { item: other, response: 1 },
      ]),
    ).toBe('unbounded-high');
    expect(
      patternBoundedness([
        { item, response: 0 },
        { item: other, response: 0 },
      ]),
    ).toBe('unbounded-low');
  });

  it('reports a mixed pattern as bounded', () => {
    expect(
      patternBoundedness([
        { item, response: 1 },
        { item: other, response: 0 },
      ]),
    ).toBe('bounded');
  });

  it('reports an empty pattern separately', () => {
    expect(patternBoundedness([])).toBe('empty');
  });

  it('flags a single response as unbounded in its direction', () => {
    expect(patternBoundedness([{ item, response: 1 }])).toBe('unbounded-high');
    expect(patternBoundedness([{ item, response: 0 }])).toBe('unbounded-low');
  });
});
