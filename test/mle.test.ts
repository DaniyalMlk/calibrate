import { describe, expect, it } from 'vitest';
import { linspace, mean, rootMeanSquareError } from '../src/core/numeric.js';
import { createRng } from '../src/core/random.js';
import { makeItem, rasch, threePL, twoPL, type Item } from '../src/models/item.js';
import type { ScoredResponse } from '../src/models/response.js';
import { logLikelihood, scoreFunction } from '../src/estimation/likelihood.js';
import { estimateMle } from '../src/estimation/mle.js';
import { simulateResponses } from '../src/simulation/respondent.js';

/** A 40-item bank spread evenly over the ability range. */
function uniformBank(count: number, a = 1.2): Item[] {
  return linspace(-2.5, 2.5, count).map((b, index) => makeItem(`i-${index}`, twoPL(a, b)));
}

describe('estimateMle', () => {
  it('recovers the generating ability to within three standard errors', () => {
    // A single replication is a random draw, so the tolerance has to be stated
    // in the estimator's own units of uncertainty rather than as a fixed number
    // of decimal places.
    const bank = uniformBank(60);
    for (const trueTheta of [-1.5, -0.5, 0, 0.75, 1.5]) {
      const responses = simulateResponses(bank, trueTheta, createRng(1234 + trueTheta * 10));
      const estimate = estimateMle(responses);
      expect(estimate.converged).toBe(true);
      expect(estimate.boundary).toBe('none');
      expect(estimate.standardError).toBeLessThan(0.45);
      expect(Math.abs(estimate.theta - trueTheta)).toBeLessThan(3 * estimate.standardError);
    }
  });

  it('recovers the generating ability on average across the range', () => {
    const bank = uniformBank(60);
    for (const trueTheta of [-1.5, -0.5, 0, 0.75, 1.5]) {
      const estimates: number[] = [];
      for (let replication = 0; replication < 200; replication += 1) {
        const responses = simulateResponses(bank, trueTheta, createRng(replication * 104729 + 3));
        const estimate = estimateMle(responses);
        if (estimate.boundary === 'none') estimates.push(estimate.theta);
      }
      expect(mean(estimates)).toBeCloseTo(trueTheta, 1);
    }
  });

  it('places the estimate at a stationary point of the log-likelihood', () => {
    const bank = uniformBank(30);
    const responses = simulateResponses(bank, 0.6, createRng(9));
    const { theta } = estimateMle(responses);
    expect(scoreFunction(responses, theta)).toBeCloseTo(0, 6);
    // And it is a maximum, not just any stationary point.
    expect(logLikelihood(responses, theta)).toBeGreaterThan(logLikelihood(responses, theta - 0.05));
    expect(logLikelihood(responses, theta)).toBeGreaterThan(logLikelihood(responses, theta + 0.05));
  });

  it('reproduces a hand-checkable Rasch case', () => {
    // For a Rasch pattern the MLE solves sum_i P_i(theta) = number correct.
    // With items at b = -1, 0, 1 and two correct answers the solution of
    // sigma(t+1) + sigma(t) + sigma(t-1) = 2 is theta = 0.7645..., found here by
    // an independent bisection on that equation.
    const items = [makeItem('a', rasch(-1)), makeItem('b', rasch(0)), makeItem('c', rasch(1))];
    const responses: ScoredResponse[] = [
      { item: items[0]!, response: 1 },
      { item: items[1]!, response: 1 },
      { item: items[2]!, response: 0 },
    ];
    const sigma = (x: number): number => 1 / (1 + Math.exp(-x));
    const g = (t: number): number => sigma(t + 1) + sigma(t) + sigma(t - 1) - 2;
    let lo = -5;
    let hi = 5;
    for (let i = 0; i < 200; i += 1) {
      const mid = (lo + hi) / 2;
      if (g(lo) * g(mid) <= 0) hi = mid;
      else lo = mid;
    }
    const reference = (lo + hi) / 2;
    expect(estimateMle(responses).theta).toBeCloseTo(reference, 7);
  });

  it('is unbiased in the mean over many replications at a fixed ability', () => {
    const bank = uniformBank(50);
    const trueTheta = 0.5;
    const estimates: number[] = [];
    for (let replication = 0; replication < 300; replication += 1) {
      const responses = simulateResponses(bank, trueTheta, createRng(replication * 7919 + 1));
      const estimate = estimateMle(responses);
      if (estimate.boundary === 'none') estimates.push(estimate.theta);
    }
    expect(estimates.length).toBeGreaterThan(280);
    // Maximum likelihood is biased outward in finite samples; over a 50-item
    // test the bias should still be small.
    expect(mean(estimates)).toBeCloseTo(trueTheta, 1);
    const rmse = rootMeanSquareError(estimates, estimates.map(() => trueTheta));
    expect(rmse).toBeLessThan(0.4);
  });

  it('reports a tighter standard error for a longer test', () => {
    const shortTest = simulateResponses(uniformBank(10), 0, createRng(3));
    const longTest = simulateResponses(uniformBank(60), 0, createRng(3));
    expect(estimateMle(longTest).standardError).toBeLessThan(estimateMle(shortTest).standardError);
  });

  it('agrees with the analytic standard error 1 / sqrt(I)', () => {
    const responses = simulateResponses(uniformBank(40), 0.2, createRng(17));
    const estimate = estimateMle(responses);
    let information = 0;
    for (const { item } of responses) {
      const p = 1 / (1 + Math.exp(-item.parameters.a * (estimate.theta - item.parameters.b)));
      information += item.parameters.a * item.parameters.a * p * (1 - p);
    }
    expect(estimate.standardError).toBeCloseTo(1 / Math.sqrt(information), 8);
  });

  it('handles 3PL patterns, where the likelihood may not be unimodal', () => {
    const bank = linspace(-2, 2, 40).map((b, index) =>
      makeItem(`t-${index}`, threePL(1.3, b, 0.25)),
    );
    const responses = simulateResponses(bank, 0.8, createRng(555));
    const estimate = estimateMle(responses);
    expect(estimate.converged).toBe(true);
    expect(scoreFunction(responses, estimate.theta)).toBeCloseTo(0, 5);
  });
});

describe('estimateMle boundary handling', () => {
  const bank = uniformBank(8);

  it('flags an all-correct pattern as unbounded above rather than estimating it', () => {
    const responses: ScoredResponse[] = bank.map((item) => ({ item, response: 1 }));
    const estimate = estimateMle(responses);
    expect(estimate.boundary).toBe('upper');
    expect(estimate.converged).toBe(false);
    expect(estimate.theta).toBe(5);
  });

  it('flags an all-incorrect pattern as unbounded below', () => {
    const responses: ScoredResponse[] = bank.map((item) => ({ item, response: 0 }));
    const estimate = estimateMle(responses);
    expect(estimate.boundary).toBe('lower');
    expect(estimate.converged).toBe(false);
    expect(estimate.theta).toBe(-5);
  });

  it('respects a custom search window at the boundary', () => {
    const responses: ScoredResponse[] = bank.map((item) => ({ item, response: 1 }));
    expect(estimateMle(responses, { min: -4, max: 4 }).theta).toBe(4);
  });

  it('flags a maximum lying outside a narrow window', () => {
    // Every easy item correct, every hard item wrong, but the window is pinned
    // far below where the likelihood peaks.
    const responses: ScoredResponse[] = [
      { item: makeItem('a', twoPL(1.2, 0)), response: 1 },
      { item: makeItem('b', twoPL(1.2, 0.5)), response: 1 },
      { item: makeItem('c', twoPL(1.2, 1)), response: 0 },
    ];
    const estimate = estimateMle(responses, { min: -5, max: -3 });
    expect(estimate.boundary).toBe('upper');
    expect(estimate.theta).toBe(-3);
    expect(estimate.converged).toBe(false);
  });

  it('rejects an empty response pattern', () => {
    expect(() => estimateMle([])).toThrow(/at least one response/);
  });

  it('rejects an empty search window', () => {
    const responses = simulateResponses(bank, 0, createRng(1));
    expect(() => estimateMle(responses, { min: 1, max: 1 })).toThrow(/empty search window/);
  });
});
