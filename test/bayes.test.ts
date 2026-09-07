import { describe, expect, it } from 'vitest';
import { linspace, mean } from '../src/core/numeric.js';
import { createRng } from '../src/core/random.js';
import { makeItem, rasch, twoPL, type Item } from '../src/models/item.js';
import type { ScoredResponse } from '../src/models/response.js';
import { estimateEap, estimateMap } from '../src/estimation/bayes.js';
import { estimateMle } from '../src/estimation/mle.js';
import { normalPrior, priorInformation, priorLogDensity, priorScore } from '../src/estimation/prior.js';
import { simulateResponses } from '../src/simulation/respondent.js';

function uniformBank(count: number, a = 1.2): Item[] {
  return linspace(-2.5, 2.5, count).map((b, index) => makeItem(`i-${index}`, twoPL(a, b)));
}

describe('normalPrior', () => {
  it('validates its parameters', () => {
    expect(() => normalPrior(0, 0)).toThrow(RangeError);
    expect(() => normalPrior(0, -1)).toThrow(RangeError);
    expect(() => normalPrior(Number.NaN, 1)).toThrow(RangeError);
    expect(normalPrior()).toEqual({ mean: 0, sd: 1 });
  });

  it('has a log density matching the closed form', () => {
    // At the mean the standard normal density is 1 / sqrt(2 pi).
    expect(priorLogDensity(normalPrior(0, 1), 0)).toBeCloseTo(Math.log(1 / Math.sqrt(2 * Math.PI)), 13);
    // One standard deviation out costs exactly 1/2 in log density.
    expect(priorLogDensity(normalPrior(0, 1), 1)).toBeCloseTo(
      priorLogDensity(normalPrior(0, 1), 0) - 0.5,
      13,
    );
  });

  it('has a score agreeing with a finite difference of the log density', () => {
    const prior = normalPrior(0.3, 1.4);
    const h = 1e-6;
    for (const theta of [-2, 0, 1.5]) {
      const numeric = (priorLogDensity(prior, theta + h) - priorLogDensity(prior, theta - h)) / (2 * h);
      expect(priorScore(prior, theta)).toBeCloseTo(numeric, 7);
    }
  });

  it('contributes information equal to the reciprocal variance', () => {
    expect(priorInformation(normalPrior(0, 2))).toBeCloseTo(0.25, 14);
  });
});

describe('estimateEap', () => {
  it('returns the prior mean for an empty pattern', () => {
    expect(estimateEap([]).theta).toBeCloseTo(0, 10);
    expect(estimateEap([], { prior: normalPrior(0.7, 1) }).theta).toBeCloseTo(0.7, 10);
  });

  it('returns the prior standard deviation for an empty pattern', () => {
    expect(estimateEap([], { prior: normalPrior(0, 1.5) }).posteriorSd).toBeCloseTo(1.5, 6);
  });

  it('shrinks towards the prior mean relative to maximum likelihood', () => {
    const responses = simulateResponses(uniformBank(12), 1.4, createRng(21));
    const eap = estimateEap(responses);
    const mle = estimateMle(responses);
    expect(Math.abs(eap.theta)).toBeLessThan(Math.abs(mle.theta));
  });

  it('shrinks less as evidence accumulates', () => {
    const shortTest = simulateResponses(uniformBank(6), 1.5, createRng(4));
    const longTest = simulateResponses(uniformBank(70), 1.5, createRng(4));
    const shortGap = Math.abs(estimateEap(shortTest).theta - estimateMle(shortTest).theta);
    const longGap = Math.abs(estimateEap(longTest).theta - estimateMle(longTest).theta);
    expect(longGap).toBeLessThan(shortGap);
  });

  it('converges on the maximum likelihood estimate for a long test', () => {
    const responses = simulateResponses(uniformBank(150), 0.4, createRng(77));
    expect(estimateEap(responses).theta).toBeCloseTo(estimateMle(responses).theta, 1);
  });

  it('tightens the posterior as items accumulate', () => {
    const shortTest = simulateResponses(uniformBank(5), 0, createRng(8));
    const longTest = simulateResponses(uniformBank(60), 0, createRng(8));
    expect(estimateEap(longTest).posteriorSd).toBeLessThan(estimateEap(shortTest).posteriorSd);
    expect(estimateEap(shortTest).posteriorSd).toBeLessThan(1);
  });

  it('agrees between the two quadrature families', () => {
    const responses = simulateResponses(uniformBank(25), 0.9, createRng(31));
    const gh = estimateEap(responses, { quadrature: 'gauss-hermite', points: 41 });
    const grid = estimateEap(responses, { quadrature: 'grid', points: 81 });
    expect(gh.theta).toBeCloseTo(grid.theta, 4);
    expect(gh.posteriorSd).toBeCloseTo(grid.posteriorSd, 4);
  });

  it('is stable for a long test, where unshifted likelihoods would underflow', () => {
    // 400 items give a likelihood around 1e-120 at the posterior mode; without
    // the log-sum shift both moments underflow to zero.
    const responses = simulateResponses(uniformBank(400), 0.5, createRng(2));
    const estimate = estimateEap(responses);
    expect(Number.isFinite(estimate.theta)).toBe(true);
    expect(estimate.theta).toBeCloseTo(0.5, 0);
    expect(estimate.posteriorSd).toBeGreaterThan(0);
  });

  it('handles an all-correct pattern, where maximum likelihood has no answer', () => {
    const responses: ScoredResponse[] = uniformBank(8).map((item) => ({ item, response: 1 }));
    const estimate = estimateEap(responses);
    expect(estimate.boundary).toBe('none');
    expect(estimate.theta).toBeGreaterThan(1);
    expect(Number.isFinite(estimate.theta)).toBe(true);
  });

  it('validates the prior it is given', () => {
    expect(() => estimateEap([], { prior: { mean: 0, sd: 0 } })).toThrow(RangeError);
  });
});

describe('estimateMap', () => {
  it('sits between the prior mean and the maximum likelihood estimate', () => {
    const responses = simulateResponses(uniformBank(12), 1.4, createRng(21));
    const map = estimateMap(responses).theta;
    const mle = estimateMle(responses).theta;
    expect(map).toBeGreaterThan(0);
    expect(map).toBeLessThan(mle);
  });

  it('places the estimate at a stationary point of the posterior', () => {
    const responses = simulateResponses(uniformBank(30), 0.6, createRng(12));
    const prior = normalPrior(0, 1);
    const { theta } = estimateMap(responses, { prior });
    const posteriorScore =
      responses.reduce((acc, { item, response }) => {
        const p = 1 / (1 + Math.exp(-item.parameters.a * (theta - item.parameters.b)));
        return acc + (response - p) * item.parameters.a;
      }, 0) + priorScore(prior, theta);
    expect(posteriorScore).toBeCloseTo(0, 6);
  });

  it('has a posterior standard deviation of 1 / sqrt(I + 1/sd^2)', () => {
    const responses = simulateResponses(uniformBank(20), 0.3, createRng(6));
    const prior = normalPrior(0, 1.2);
    const estimate = estimateMap(responses, { prior });
    let information = 0;
    for (const { item } of responses) {
      const p = 1 / (1 + Math.exp(-item.parameters.a * (estimate.theta - item.parameters.b)));
      information += item.parameters.a * item.parameters.a * p * (1 - p);
    }
    expect(estimate.posteriorSd).toBeCloseTo(1 / Math.sqrt(information + priorInformation(prior)), 7);
  });

  it('is finite for all-correct and all-incorrect patterns', () => {
    const bank = uniformBank(8);
    const allCorrect: ScoredResponse[] = bank.map((item) => ({ item, response: 1 }));
    const allWrong: ScoredResponse[] = bank.map((item) => ({ item, response: 0 }));
    const high = estimateMap(allCorrect);
    const low = estimateMap(allWrong);
    expect(high.converged).toBe(true);
    expect(low.converged).toBe(true);
    expect(high.theta).toBeGreaterThan(1);
    expect(low.theta).toBeLessThan(-1);
    expect(high.theta).toBeCloseTo(-low.theta, 6);
  });

  it('converges on the maximum likelihood estimate for a long test', () => {
    const responses = simulateResponses(uniformBank(200), -0.7, createRng(43));
    expect(estimateMap(responses).theta).toBeCloseTo(estimateMle(responses).theta, 1);
  });

  it('respects a shifted prior', () => {
    const responses = simulateResponses(uniformBank(6), 0, createRng(15));
    const centred = estimateMap(responses, { prior: normalPrior(0, 1) }).theta;
    const shifted = estimateMap(responses, { prior: normalPrior(1, 1) }).theta;
    expect(shifted).toBeGreaterThan(centred);
  });

  it('is pulled less by a diffuse prior', () => {
    const responses = simulateResponses(uniformBank(10), 1.6, createRng(19));
    const tight = estimateMap(responses, { prior: normalPrior(0, 0.5) }).theta;
    const diffuse = estimateMap(responses, { prior: normalPrior(0, 3) }).theta;
    expect(diffuse).toBeGreaterThan(tight);
  });

  it('rejects a non-positive search width', () => {
    const responses = simulateResponses(uniformBank(5), 0, createRng(1));
    expect(() => estimateMap(responses, { searchWidth: 0 })).toThrow(RangeError);
  });

  it('agrees closely with EAP for a symmetric posterior', () => {
    // With a symmetric bank and a symmetric prior the posterior is close to
    // normal, so its mode and mean nearly coincide.
    const responses: ScoredResponse[] = [
      { item: makeItem('a', rasch(-1)), response: 1 },
      { item: makeItem('b', rasch(0)), response: 1 },
      { item: makeItem('c', rasch(1)), response: 0 },
      { item: makeItem('d', rasch(2)), response: 0 },
    ];
    expect(estimateMap(responses).theta).toBeCloseTo(estimateEap(responses).theta, 1);
  });
});

describe('estimator agreement over replications', () => {
  it('EAP is closer to the truth on short tests than maximum likelihood', () => {
    const bank = uniformBank(8);
    const trueTheta = 0.5;
    const eapErrors: number[] = [];
    const mleErrors: number[] = [];
    for (let replication = 0; replication < 400; replication += 1) {
      const responses = simulateResponses(bank, trueTheta, createRng(replication * 65537 + 11));
      const mle = estimateMle(responses);
      if (mle.boundary !== 'none') continue;
      eapErrors.push(Math.pow(estimateEap(responses).theta - trueTheta, 2));
      mleErrors.push(Math.pow(mle.theta - trueTheta, 2));
    }
    expect(eapErrors.length).toBeGreaterThan(200);
    // Shrinkage trades a little bias for a lot of variance on a short test.
    expect(mean(eapErrors)).toBeLessThan(mean(mleErrors));
  });
});
