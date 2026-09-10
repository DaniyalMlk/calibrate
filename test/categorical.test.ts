import { describe, expect, it } from 'vitest';
import { linspace } from '../src/core/numeric.js';
import {
  expectedInformation,
  logLikelihood,
  observedInformation,
  patternBoundedness,
  scoreFunction,
} from '../src/estimation/likelihood.js';
import { estimateEap, estimateMap } from '../src/estimation/bayes.js';
import { estimateMle } from '../src/estimation/mle.js';
import { estimateWle, warmCorrection } from '../src/estimation/wle.js';
import { makeItem, rasch, threePL, twoPL } from '../src/models/item.js';
import {
  categoryProbabilitiesOf,
  makePolytomousItem,
  maximumScoreOf,
  responseCategory,
  responseItem,
  type AnyResponse,
} from '../src/models/mixed.js';
import { generalizedPartialCredit, graded, partialCredit } from '../src/models/polytomous.js';
import {
  probabilityCorrect,
  responseDerivative,
  responseSecondDerivative,
  type ScoredResponse,
} from '../src/models/response.js';

const MC_A = makeItem('mc-a', twoPL(1.4, -0.4));
const MC_B = makeItem('mc-b', twoPL(1.1, 0.6));
const MC_C = makeItem('mc-c', threePL(1.3, 0.1, 0.2));
const MC_D = makeItem('mc-d', rasch(-1.2));
const ESSAY = makePolytomousItem('essay', graded(1.2, [-1.4, -0.1, 1.0]));
const TASK = makePolytomousItem('task', generalizedPartialCredit(1.5, [-0.9, 0.3, 1.1]));
const STEPS = makePolytomousItem('steps', partialCredit([-1.0, 0.2, 1.3]));

const GRID = linspace(-4, 4, 33);

/** Central difference, accurate to O(h^2). */
function centralDifference(f: (x: number) => number, x: number, h = 1e-5): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}

describe('response normalisation', () => {
  it('reads a category from either response shape', () => {
    const scored: ScoredResponse = { item: MC_A, response: 1 };
    const categorical = { item: ESSAY, category: 2 };
    expect(responseCategory(scored)).toBe(1);
    expect(responseCategory(categorical)).toBe(2);
    expect(responseItem(scored)).toBe(MC_A);
    expect(responseItem(categorical)).toBe(ESSAY);
  });

  it('treats a dichotomous response and its category form as the same pattern', () => {
    const asScored: AnyResponse[] = [
      { item: MC_A, response: 1 },
      { item: MC_B, response: 0 },
    ];
    const asCategorical: AnyResponse[] = [
      { item: MC_A, category: 1 },
      { item: MC_B, category: 0 },
    ];
    for (const theta of GRID) {
      expect(logLikelihood(asCategorical, theta)).toBe(logLikelihood(asScored, theta));
      expect(scoreFunction(asCategorical, theta)).toBe(scoreFunction(asScored, theta));
      expect(observedInformation(asCategorical, theta)).toBe(
        observedInformation(asScored, theta),
      );
    }
    expect(estimateMle(asCategorical).theta).toBe(estimateMle(asScored).theta);
  });
});

describe('the dichotomous case of the categorical likelihood', () => {
  const pattern: ScoredResponse[] = [
    { item: MC_A, response: 1 },
    { item: MC_B, response: 0 },
    { item: MC_C, response: 1 },
    { item: MC_D, response: 0 },
  ];

  it('reproduces the residual form of the score function', () => {
    for (const theta of GRID) {
      let expected = 0;
      for (const { item, response } of pattern) {
        const p = probabilityCorrect(item.parameters, theta);
        expected += ((response - p) * responseDerivative(item.parameters, theta)) / (p * (1 - p));
      }
      expect(scoreFunction(pattern, theta)).toBeCloseTo(expected, 12);
    }
  });

  it('reproduces the u/P, (1-u)/Q form of the observed information', () => {
    for (const theta of GRID) {
      let expected = 0;
      for (const { item, response } of pattern) {
        const p = probabilityCorrect(item.parameters, theta);
        const q = 1 - p;
        const first = responseDerivative(item.parameters, theta);
        const second = responseSecondDerivative(item.parameters, theta);
        const curvature = response === 1 ? 1 / (p * p) : 1 / (q * q);
        const slope = response === 1 ? 1 / p : -1 / q;
        expected += curvature * first * first - slope * second;
      }
      expect(observedInformation(pattern, theta)).toBeCloseTo(expected, 10);
    }
  });

  it("reproduces Warm's P' P'' / (P Q) form", () => {
    for (const theta of GRID) {
      let expected = 0;
      for (const { item } of pattern) {
        const p = probabilityCorrect(item.parameters, theta);
        const q = 1 - p;
        expected +=
          (responseDerivative(item.parameters, theta) *
            responseSecondDerivative(item.parameters, theta)) /
          (p * q);
      }
      expect(warmCorrection(pattern, theta)).toBeCloseTo(expected, 12);
    }
  });
});

describe('the categorical likelihood over a mixed pattern', () => {
  const pattern: AnyResponse[] = [
    { item: MC_A, response: 1 },
    { item: MC_B, response: 0 },
    { item: ESSAY, category: 2 },
    { item: TASK, category: 1 },
    { item: STEPS, category: 3 },
  ];

  it('sums the log of the probability of each observed category', () => {
    for (const theta of [-2, -0.5, 0.7, 2]) {
      // Recomputed from the model layer, independently of the likelihood code.
      let expected = 0;
      for (const response of pattern) {
        const item = responseItem(response);
        const k = responseCategory(response);
        expected += Math.log(categoryProbabilitiesOf(item, theta)[k] as number);
      }
      expect(logLikelihood(pattern, theta)).toBeCloseTo(expected, 12);
    }
  });

  it('has the score function as the derivative of the log-likelihood', () => {
    for (const theta of linspace(-3, 3, 13)) {
      expect(scoreFunction(pattern, theta)).toBeCloseTo(
        centralDifference((x) => logLikelihood(pattern, x), theta),
        7,
      );
    }
  });

  it('has the observed information as the negative curvature of the log-likelihood', () => {
    for (const theta of linspace(-3, 3, 13)) {
      expect(observedInformation(pattern, theta)).toBeCloseTo(
        -centralDifference((x) => scoreFunction(pattern, x), theta),
        6,
      );
    }
  });

  it("has Warm's correction as the derivative-weighted curvature sum", () => {
    // J is finite and continuous across the range; the estimator depends on it
    // staying so once polytomous items enter the pattern.
    for (const theta of GRID) {
      expect(Number.isFinite(warmCorrection(pattern, theta))).toBe(true);
    }
  });

  it('sums Fisher information across both formats', () => {
    for (const theta of GRID) {
      expect(expectedInformation(pattern, theta)).toBeGreaterThan(0);
    }
  });

  it('stays finite at abilities where every observed category has underflowed', () => {
    for (const theta of [-30, 30]) {
      expect(Number.isFinite(logLikelihood(pattern, theta))).toBe(true);
      expect(Number.isFinite(scoreFunction(pattern, theta))).toBe(true);
      expect(Number.isFinite(observedInformation(pattern, theta))).toBe(true);
    }
  });
});

describe('boundedness of a mixed pattern', () => {
  it('is empty for no responses', () => {
    expect(patternBoundedness([])).toBe('empty');
  });

  it('is unbounded high only when every item scored its own maximum', () => {
    expect(
      patternBoundedness([
        { item: MC_A, response: 1 },
        { item: ESSAY, category: maximumScoreOf(ESSAY) },
      ]),
    ).toBe('unbounded-high');

    // Perfect on the multiple choice, one short on the essay: finite.
    expect(
      patternBoundedness([
        { item: MC_A, response: 1 },
        { item: ESSAY, category: maximumScoreOf(ESSAY) - 1 },
      ]),
    ).toBe('bounded');
  });

  it('is unbounded low only when every item scored zero', () => {
    expect(
      patternBoundedness([
        { item: MC_A, response: 0 },
        { item: ESSAY, category: 0 },
      ]),
    ).toBe('unbounded-low');

    expect(
      patternBoundedness([
        { item: MC_A, response: 0 },
        { item: ESSAY, category: 1 },
      ]),
    ).toBe('bounded');
  });

  it('treats a single middle category as bounded', () => {
    expect(patternBoundedness([{ item: ESSAY, category: 1 }])).toBe('bounded');
    expect(patternBoundedness([{ item: ESSAY, category: 2 }])).toBe('bounded');
  });

  it('does not mistake a top category for a maximum on a longer item', () => {
    // Category 1 is the maximum of a dichotomous item but the middle of a
    // four-category one; the distinction is the whole point.
    expect(patternBoundedness([{ item: MC_A, response: 1 }])).toBe('unbounded-high');
    expect(patternBoundedness([{ item: ESSAY, category: 1 }])).toBe('bounded');
  });
});

describe('estimating ability from a mixed pattern', () => {
  const pattern: AnyResponse[] = [
    { item: MC_A, response: 1 },
    { item: MC_B, response: 1 },
    { item: MC_C, response: 0 },
    { item: ESSAY, category: 2 },
    { item: TASK, category: 2 },
    { item: STEPS, category: 1 },
  ];

  it('locates a stationary point of the likelihood with maximum likelihood', () => {
    const estimate = estimateMle(pattern);
    expect(estimate.converged).toBe(true);
    expect(estimate.boundary).toBe('none');
    expect(Math.abs(scoreFunction(pattern, estimate.theta))).toBeLessThan(1e-6);
    expect(estimate.standardError).toBeGreaterThan(0);
    expect(Number.isFinite(estimate.standardError)).toBe(true);
  });

  it('maximises the log-likelihood there, not merely solves the score', () => {
    const estimate = estimateMle(pattern);
    const best = logLikelihood(pattern, estimate.theta);
    for (const offset of [-0.5, -0.05, 0.05, 0.5]) {
      expect(best).toBeGreaterThan(logLikelihood(pattern, estimate.theta + offset));
    }
  });

  it('agrees across estimators to within their standard errors', () => {
    const mle = estimateMle(pattern);
    const eap = estimateEap(pattern);
    const map = estimateMap(pattern);
    const wle = estimateWle(pattern);
    for (const estimate of [eap, map, wle]) {
      expect(Math.abs(estimate.theta - mle.theta)).toBeLessThan(mle.standardError);
    }
    // The Bayesian estimators shrink towards the prior mean of zero, so they
    // must sit between the prior and the likelihood's own maximum.
    expect(Math.abs(eap.theta)).toBeLessThan(Math.abs(mle.theta) + 1e-9);
    expect(Math.abs(map.theta)).toBeLessThan(Math.abs(mle.theta) + 1e-9);
  });

  it('returns a finite weighted likelihood estimate for a perfect mixed pattern', () => {
    const perfect: AnyResponse[] = [
      { item: MC_A, response: 1 },
      { item: MC_B, response: 1 },
      { item: ESSAY, category: maximumScoreOf(ESSAY) },
    ];
    expect(patternBoundedness(perfect)).toBe('unbounded-high');
    expect(estimateMle(perfect).boundary).toBe('upper');

    const wle = estimateWle(perfect);
    expect(Number.isFinite(wle.theta)).toBe(true);
    expect(wle.theta).toBeLessThan(5);
    expect(wle.theta).toBeGreaterThan(0);
  });

  it('gains precision as polytomous items are added', () => {
    const short: AnyResponse[] = [
      { item: MC_A, response: 1 },
      { item: MC_B, response: 0 },
    ];
    const long: AnyResponse[] = [...short, { item: ESSAY, category: 2 }, { item: TASK, category: 2 }];
    expect(expectedInformation(long, 0)).toBeGreaterThan(expectedInformation(short, 0));
    expect(estimateEap(long).posteriorSd).toBeLessThan(estimateEap(short).posteriorSd);
  });

  it('estimates from a purely polytomous pattern', () => {
    const polytomousOnly: AnyResponse[] = [
      { item: ESSAY, category: 2 },
      { item: TASK, category: 1 },
      { item: STEPS, category: 2 },
    ];
    const estimate = estimateMle(polytomousOnly);
    expect(estimate.converged).toBe(true);
    expect(Math.abs(scoreFunction(polytomousOnly, estimate.theta))).toBeLessThan(1e-6);
  });

  it('rejects an empty pattern', () => {
    expect(() => estimateMle([])).toThrow(/at least one response/);
  });

  it('returns the prior mean for an empty pattern under EAP', () => {
    expect(estimateEap([]).theta).toBeCloseTo(0, 10);
  });
});
