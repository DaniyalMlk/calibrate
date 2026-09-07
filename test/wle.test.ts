import { describe, expect, it } from 'vitest';
import { linspace, mean } from '../src/core/numeric.js';
import { createRng } from '../src/core/random.js';
import { makeItem, rasch, twoPL, type Item } from '../src/models/item.js';
import type { ScoredResponse } from '../src/models/response.js';
import { estimateMle } from '../src/estimation/mle.js';
import { estimateWle, warmCorrection, weightedScore } from '../src/estimation/wle.js';
import { simulateResponses } from '../src/simulation/respondent.js';

function uniformBank(count: number, a = 1.2): Item[] {
  return linspace(-2.5, 2.5, count).map((b, index) => makeItem(`i-${index}`, twoPL(a, b)));
}

describe('warmCorrection', () => {
  it('vanishes where the response function is at its inflection point', () => {
    // P'' is zero at theta = b for a 2PL item, so a single item contributes
    // nothing to the correction there.
    expect(warmCorrection([{ item: makeItem('a', twoPL(1.3, 0.5)), response: 1 }], 0.5)).toBeCloseTo(
      0,
      13,
    );
  });

  it('does not depend on the responses given', () => {
    const items = uniformBank(6);
    const correct: ScoredResponse[] = items.map((item) => ({ item, response: 1 }));
    const wrong: ScoredResponse[] = items.map((item) => ({ item, response: 0 }));
    expect(warmCorrection(correct, 0.4)).toBeCloseTo(warmCorrection(wrong, 0.4), 14);
  });

  it('is negative above the bank and positive below it, pulling estimates inward', () => {
    const responses: ScoredResponse[] = uniformBank(20).map((item) => ({ item, response: 1 }));
    expect(warmCorrection(responses, 2.5)).toBeLessThan(0);
    expect(warmCorrection(responses, -2.5)).toBeGreaterThan(0);
  });
});

describe('weightedScore', () => {
  it('reduces to 1.5 - 2 P for a single correct Rasch item', () => {
    const responses: ScoredResponse[] = [{ item: makeItem('a', rasch(0)), response: 1 }];
    for (const theta of linspace(-3, 3, 25)) {
      const p = 1 / (1 + Math.exp(-theta));
      expect(weightedScore(responses, theta)).toBeCloseTo(1.5 - 2 * p, 12);
    }
  });
});

describe('estimateWle', () => {
  it('gives b + ln 3 for a single correct Rasch item', () => {
    // The weighted score 1.5 - 2 P vanishes at P = 0.75, i.e. theta = b + ln 3.
    const responses: ScoredResponse[] = [{ item: makeItem('a', rasch(0.4)), response: 1 }];
    expect(estimateWle(responses).theta).toBeCloseTo(0.4 + Math.log(3), 7);
  });

  it('gives b - ln 3 for a single incorrect Rasch item', () => {
    const responses: ScoredResponse[] = [{ item: makeItem('a', rasch(-0.2)), response: 0 }];
    expect(estimateWle(responses).theta).toBeCloseTo(-0.2 - Math.log(3), 7);
  });

  it('is finite for all-correct and all-incorrect patterns', () => {
    const bank = uniformBank(10);
    const allCorrect: ScoredResponse[] = bank.map((item) => ({ item, response: 1 }));
    const allWrong: ScoredResponse[] = bank.map((item) => ({ item, response: 0 }));

    expect(estimateMle(allCorrect).boundary).toBe('upper');
    const high = estimateWle(allCorrect);
    const low = estimateWle(allWrong);
    expect(high.converged).toBe(true);
    expect(high.boundary).toBe('none');
    expect(Number.isFinite(high.theta)).toBe(true);
    expect(high.theta).toBeGreaterThan(2);
    expect(high.theta).toBeCloseTo(-low.theta, 5);
  });

  it('places the estimate at a root of the weighted score', () => {
    const responses = simulateResponses(uniformBank(25), 0.7, createRng(88));
    const { theta } = estimateWle(responses);
    expect(weightedScore(responses, theta)).toBeCloseTo(0, 6);
  });

  it('is less outwardly biased than maximum likelihood in the tails', () => {
    // Maximum likelihood over-disperses: at a true ability of 1.5 on a short
    // test its mean estimate sits above 1.5. The weighted estimator should sit
    // closer to the truth.
    const bank = uniformBank(15);
    const trueTheta = 1.5;
    const mleEstimates: number[] = [];
    const wleEstimates: number[] = [];
    for (let replication = 0; replication < 500; replication += 1) {
      const responses = simulateResponses(bank, trueTheta, createRng(replication * 15485863 + 5));
      const mle = estimateMle(responses);
      if (mle.boundary !== 'none') continue;
      mleEstimates.push(mle.theta);
      wleEstimates.push(estimateWle(responses).theta);
    }
    expect(mleEstimates.length).toBeGreaterThan(200);
    const mleBias = mean(mleEstimates) - trueTheta;
    const wleBias = mean(wleEstimates) - trueTheta;
    expect(mleBias).toBeGreaterThan(0);
    expect(Math.abs(wleBias)).toBeLessThan(Math.abs(mleBias));
  });

  it('converges on the maximum likelihood estimate for a long test', () => {
    const responses = simulateResponses(uniformBank(200), 0.3, createRng(64));
    expect(estimateWle(responses).theta).toBeCloseTo(estimateMle(responses).theta, 1);
  });

  it('reports the same standard error basis as maximum likelihood', () => {
    const responses = simulateResponses(uniformBank(30), 0.2, createRng(2));
    const wle = estimateWle(responses);
    let information = 0;
    for (const { item } of responses) {
      const p = 1 / (1 + Math.exp(-item.parameters.a * (wle.theta - item.parameters.b)));
      information += item.parameters.a * item.parameters.a * p * (1 - p);
    }
    expect(wle.standardError).toBeCloseTo(1 / Math.sqrt(information), 8);
  });

  it('rejects an empty pattern and an empty window', () => {
    expect(() => estimateWle([])).toThrow(/at least one response/);
    const responses = simulateResponses(uniformBank(4), 0, createRng(1));
    expect(() => estimateWle(responses, { min: 2, max: 2 })).toThrow(/empty search window/);
  });

  it('flags a root lying outside a narrow window', () => {
    const responses: ScoredResponse[] = uniformBank(10).map((item) => ({ item, response: 1 }));
    const estimate = estimateWle(responses, { min: -5, max: -3 });
    expect(estimate.boundary).toBe('upper');
    expect(estimate.converged).toBe(false);
  });
});
