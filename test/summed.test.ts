import { describe, expect, it } from 'vitest';
import { normalGaussHermiteRule } from '../src/core/quadrature.js';
import { createRng } from '../src/core/random.js';
import { makeItem, onePL, twoPL, type Item } from '../src/models/item.js';
import { graded, partialCredit } from '../src/models/polytomous.js';
import {
  categoryProbabilitiesOf,
  expectedScoreOf,
  makePolytomousItem,
  maximumScoreOf,
  maximumTestScore,
  testCharacteristicCurve,
  type AnyItem,
} from '../src/models/mixed.js';
import { categoryScoreVariance } from '../src/models/polytomous.js';
import { probabilityCorrect } from '../src/models/response.js';
import {
  marginalScoreDistribution,
  scoreDistribution,
  scoreLikelihoods,
} from '../src/scoring/summed.js';

const binary: Item[] = [
  makeItem('b0', twoPL(1.2, -1)),
  makeItem('b1', onePL(0)),
  makeItem('b2', twoPL(0.8, 0.7)),
  makeItem('b3', twoPL(1.6, 1.4)),
];

const mixed: AnyItem[] = [
  makeItem('b0', twoPL(1.1, -0.5)),
  makePolytomousItem('r0', graded(1.3, [-1, 0.2, 1.1])),
  makeItem('b1', onePL(0.6)),
  makePolytomousItem('r1', partialCredit([-0.5, 0.8])),
];

/**
 * The same distribution by brute force: enumerate every response pattern the
 * form admits, multiply the category probabilities, and add the product to the
 * bucket for its total. Exponential in the number of items, which is why the
 * recursion exists, but exact and obviously correct on a short form.
 */
function enumerateScores(items: readonly AnyItem[], theta: number): number[] {
  const totals = new Array<number>(maximumTestScore(items) + 1).fill(0);
  const probabilities = items.map((item) => categoryProbabilitiesOf(item, theta));

  const walk = (index: number, score: number, mass: number): void => {
    if (index === items.length) {
      totals[score] = (totals[score] as number) + mass;
      return;
    }
    const row = probabilities[index] as number[];
    for (let category = 0; category <= maximumScoreOf(items[index] as AnyItem); category += 1) {
      walk(index + 1, score + category, mass * (row[category] as number));
    }
  };
  walk(0, 0, 1);
  return totals;
}

function totalOf(values: readonly number[] | Float64Array): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

describe('scoreDistribution', () => {
  it('matches brute-force enumeration on a binary form', () => {
    for (const theta of [-2, -0.5, 0, 0.9, 2.5]) {
      const recursed = scoreDistribution(binary, theta);
      const enumerated = enumerateScores(binary, theta);
      expect(recursed).toHaveLength(enumerated.length);
      for (const [score, mass] of enumerated.entries()) {
        expect(recursed[score] as number).toBeCloseTo(mass, 13);
      }
    }
  });

  it('matches brute-force enumeration on a form mixing both item formats', () => {
    for (const theta of [-1.5, 0, 1.7]) {
      const recursed = scoreDistribution(mixed, theta);
      const enumerated = enumerateScores(mixed, theta);
      for (const [score, mass] of enumerated.entries()) {
        expect(recursed[score] as number).toBeCloseTo(mass, 13);
      }
    }
  });

  it('is a distribution over exactly the attainable totals', () => {
    for (const items of [binary, mixed]) {
      for (const theta of [-3, 0, 3]) {
        const distribution = scoreDistribution(items, theta);
        expect(distribution).toHaveLength(maximumTestScore(items) + 1);
        expect(totalOf(distribution)).toBeCloseTo(1, 12);
        for (const mass of distribution) expect(mass).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('has the test characteristic curve as its mean', () => {
    // The expected total is the sum of the expected item scores, whatever the
    // dependence structure of the totals. A recursion that got the convolution
    // wrong would still sum to one but would not land here.
    for (const items of [binary, mixed]) {
      for (const theta of [-2.2, -0.3, 0.8, 2.1]) {
        const distribution = scoreDistribution(items, theta);
        let mean = 0;
        for (const [score, mass] of distribution.entries()) mean += score * mass;
        expect(mean).toBeCloseTo(testCharacteristicCurve(items, theta), 11);
      }
    }
  });

  it('has the sum of the item score variances as its variance', () => {
    // Local independence again: variances add only because the item scores are
    // independent given ability.
    for (const theta of [-1.4, 0, 1.9]) {
      const distribution = scoreDistribution(mixed, theta);
      let mean = 0;
      for (const [score, mass] of distribution.entries()) mean += score * mass;
      let variance = 0;
      for (const [score, mass] of distribution.entries()) {
        variance += mass * (score - mean) * (score - mean);
      }

      let expected = 0;
      for (const item of mixed) {
        if ('thresholds' in item.parameters) {
          expected += categoryScoreVariance(item.parameters, theta);
        } else {
          const p = probabilityCorrect(item.parameters, theta);
          expected += p * (1 - p);
        }
      }
      expect(variance).toBeCloseTo(expected, 11);
    }
  });

  it('puts almost all its mass at the floor and the ceiling in the tails', () => {
    const low = scoreDistribution(binary, -8);
    const high = scoreDistribution(binary, 8);
    expect(low[0] as number).toBeGreaterThan(0.99);
    expect(high[high.length - 1] as number).toBeGreaterThan(0.99);
  });

  it('shifts upward as ability rises, in the stochastic ordering sense', () => {
    // Every upper tail probability must increase with ability: a more able
    // candidate is at least as likely to reach any given total.
    const lower = scoreDistribution(mixed, -0.5);
    const upper = scoreDistribution(mixed, 0.5);
    let lowerTail = 0;
    let upperTail = 0;
    for (let score = lower.length - 1; score >= 0; score -= 1) {
      lowerTail += lower[score] as number;
      upperTail += upper[score] as number;
      expect(upperTail).toBeGreaterThanOrEqual(lowerTail - 1e-12);
    }
  });

  it('reduces to a single item correctly', () => {
    const single: Item[] = [makeItem('only', twoPL(1.4, 0.3))];
    const distribution = scoreDistribution(single, 0.6);
    const p = probabilityCorrect((single[0] as Item).parameters, 0.6);
    expect(distribution).toHaveLength(2);
    expect(distribution[1] as number).toBeCloseTo(p, 14);
    expect(distribution[0] as number).toBeCloseTo(1 - p, 14);
  });

  it('rejects an empty form and a non-finite ability', () => {
    expect(() => scoreDistribution([], 0)).toThrow(RangeError);
    expect(() => scoreDistribution(binary, Number.NaN)).toThrow(RangeError);
  });
});

describe('marginalScoreDistribution', () => {
  const rule = normalGaussHermiteRule(0, 1, 41);

  it('is a distribution over the attainable totals', () => {
    const distribution = marginalScoreDistribution(mixed, rule);
    expect(distribution).toHaveLength(maximumTestScore(mixed) + 1);
    expect(totalOf(distribution)).toBeCloseTo(1, 12);
  });

  it('has the population-averaged test characteristic curve as its mean', () => {
    const distribution = marginalScoreDistribution(binary, rule);
    let mean = 0;
    for (const [score, mass] of distribution.entries()) mean += score * mass;

    let expected = 0;
    for (const [k, node] of rule.nodes.entries()) {
      expected += (rule.weights[k] as number) * testCharacteristicCurve(binary, node);
    }
    expect(mean).toBeCloseTo(expected, 10);
  });

  it('matches the totals a simulated cohort actually produces', () => {
    const rng = createRng(606060);
    const counts = new Array<number>(maximumTestScore(binary) + 1).fill(0);
    const draws = 60000;
    for (let person = 0; person < draws; person += 1) {
      const theta = rng.nextNormal();
      let score = 0;
      for (const item of binary) {
        if (rng.next() < probabilityCorrect(item.parameters, theta)) score += 1;
      }
      counts[score] = (counts[score] as number) + 1;
    }
    const predicted = marginalScoreDistribution(binary, rule);
    for (const [score, count] of counts.entries()) {
      expect(count / draws).toBeCloseTo(predicted[score] as number, 2);
    }
  });

  it('spreads a wider population over a wider range of totals', () => {
    const narrow = marginalScoreDistribution(binary, normalGaussHermiteRule(0, 0.4, 41));
    const wide = marginalScoreDistribution(binary, normalGaussHermiteRule(0, 2, 41));
    const spread = (distribution: readonly number[]): number => {
      let mean = 0;
      for (const [score, mass] of distribution.entries()) mean += score * mass;
      let variance = 0;
      for (const [score, mass] of distribution.entries()) {
        variance += mass * (score - mean) * (score - mean);
      }
      return variance;
    };
    expect(spread(wide)).toBeGreaterThan(spread(narrow));
  });

  it('rejects a malformed rule', () => {
    expect(() => marginalScoreDistribution(binary, { nodes: [0, 1], weights: [1] })).toThrow(
      RangeError,
    );
  });
});

describe('scoreLikelihoods', () => {
  const rule = normalGaussHermiteRule(0, 1, 21);

  it('holds one row per attainable total and one column per node', () => {
    const table = scoreLikelihoods(mixed, rule);
    expect(table).toHaveLength(maximumTestScore(mixed) + 1);
    for (const row of table) expect(row).toHaveLength(rule.nodes.length);
  });

  it('has columns that are the conditional distributions themselves', () => {
    const table = scoreLikelihoods(binary, rule);
    for (const [k, node] of rule.nodes.entries()) {
      const conditional = scoreDistribution(binary, node);
      let column = 0;
      for (const [score, row] of table.entries()) {
        expect(row[k] as number).toBeCloseTo(conditional[score] as number, 14);
        column += row[k] as number;
      }
      expect(column).toBeCloseTo(1, 12);
    }
  });

  it('is monotone in ability at the extreme totals', () => {
    // The likelihood of a perfect score rises with ability everywhere, and that
    // of a zero score falls. Neither has an interior mode, which is why extreme
    // scores have no finite maximum likelihood estimate.
    const table = scoreLikelihoods(binary, rule);
    const perfect = table[table.length - 1] as Float64Array;
    const zero = table[0] as Float64Array;
    for (let k = 1; k < rule.nodes.length; k += 1) {
      expect(perfect[k] as number).toBeGreaterThan(perfect[k - 1] as number);
      expect(zero[k] as number).toBeLessThan(zero[k - 1] as number);
    }
  });
});

describe('the expected score of a form', () => {
  it('agrees between the item sum and the score distribution, at every node', () => {
    const rule = normalGaussHermiteRule(0, 1, 15);
    for (const node of rule.nodes) {
      let itemwise = 0;
      for (const item of mixed) itemwise += expectedScoreOf(item, node);
      const distribution = scoreDistribution(mixed, node);
      let distributional = 0;
      for (const [score, mass] of distribution.entries()) distributional += score * mass;
      expect(distributional).toBeCloseTo(itemwise, 11);
    }
  });
});
