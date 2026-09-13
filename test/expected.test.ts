import { describe, expect, it } from 'vitest';
import { logistic } from '../src/core/numeric.js';
import { normalGaussHermiteRule, normalGridRule } from '../src/core/quadrature.js';
import { createRng } from '../src/core/random.js';
import { makeItem, twoPL, type Item } from '../src/models/item.js';
import { simulateMatrix } from '../src/simulation/respondent.js';
import { MISSING, ResponseMatrix, type Cell } from '../src/calibration/matrix.js';
import {
  expectationStep,
  marginalLogLikelihood,
  patternLogLikelihood,
  pointProbability,
  posteriorOverNodes,
  type ItemPoint,
} from '../src/calibration/expected.js';

const points: ItemPoint[] = [
  { discrimination: 1, difficulty: -1 },
  { discrimination: 1.4, difficulty: 0 },
  { discrimination: 0.8, difficulty: 1 },
];

const bank: Item[] = points.map((p, index) =>
  makeItem(`i-${index}`, twoPL(p.discrimination, p.difficulty)),
);

const rule = normalGaussHermiteRule(0, 1, 41);

function totalOf(values: Float64Array | readonly number[]): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

describe('patternLogLikelihood', () => {
  it('matches the log of the product of the response probabilities', () => {
    const row: Cell[] = [1, 0, 1];
    const theta = 0.4;
    let expected = 0;
    for (const [index, cell] of row.entries()) {
      const p = pointProbability(points[index] as ItemPoint, theta);
      expected += Math.log(cell === 1 ? p : 1 - p);
    }
    expect(patternLogLikelihood(row, points, theta)).toBeCloseTo(expected, 12);
  });

  it('skips missing responses entirely', () => {
    const full = patternLogLikelihood([1, MISSING, 1], points, 0.2);
    const dropped =
      Math.log(pointProbability(points[0] as ItemPoint, 0.2)) +
      Math.log(pointProbability(points[2] as ItemPoint, 0.2));
    expect(full).toBeCloseTo(dropped, 12);
  });

  it('is zero for a pattern with no responses at all', () => {
    expect(patternLogLikelihood([MISSING, MISSING, MISSING], points, 1.3)).toBe(0);
  });

  it('stays finite where the naive product underflows', () => {
    // Forty confident items answered against the grain: the probability of the
    // pattern is around 1e-90, which a product of doubles still holds, but at
    // theta = -12 it is below 1e-300 and the product is exactly zero.
    const hard: ItemPoint[] = Array.from({ length: 40 }, () => ({
      discrimination: 2,
      difficulty: 2,
    }));
    const row: Cell[] = Array.from({ length: 40 }, () => 1 as Cell);
    const value = patternLogLikelihood(row, hard, -12);
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeLessThan(-500);

    let naive = 1;
    for (const item of hard) naive *= pointProbability(item, -12);
    expect(naive).toBe(0);
  });

  it('is symmetric under flipping every response and every ability sign', () => {
    const mirrored: ItemPoint[] = points.map((p) => ({
      discrimination: p.discrimination,
      difficulty: -p.difficulty,
    }));
    const original = patternLogLikelihood([1, 0, 1], points, 0.7);
    const flipped = patternLogLikelihood([0, 1, 0], mirrored, -0.7);
    expect(flipped).toBeCloseTo(original, 12);
  });
});

describe('posteriorOverNodes', () => {
  it('returns a normalised posterior', () => {
    const into = new Float64Array(rule.nodes.length);
    posteriorOverNodes([1, 1, 0], points, rule, into);
    expect(totalOf(into)).toBeCloseTo(1, 12);
    for (const mass of into) expect(mass).toBeGreaterThanOrEqual(0);
  });

  it('reduces to the population weights when there are no responses', () => {
    const into = new Float64Array(rule.nodes.length);
    const marginal = posteriorOverNodes([MISSING, MISSING, MISSING], points, rule, into);
    for (const [k, weight] of rule.weights.entries()) {
      expect(into[k] as number).toBeCloseTo(weight, 12);
    }
    // The marginal likelihood of the empty pattern is the total prior mass,
    // which a rule representing a probability distribution puts at one.
    expect(marginal).toBeCloseTo(0, 9);
  });

  it('returns the log of the weighted average conditional likelihood', () => {
    const into = new Float64Array(rule.nodes.length);
    const row: Cell[] = [1, 0, 1];
    const marginal = posteriorOverNodes(row, points, rule, into);

    let direct = 0;
    for (const [k, node] of rule.nodes.entries()) {
      direct += (rule.weights[k] as number) * Math.exp(patternLogLikelihood(row, points, node));
    }
    expect(marginal).toBeCloseTo(Math.log(direct), 10);
  });

  it('puts a symmetric pattern at a posterior mean of zero', () => {
    const symmetric: ItemPoint[] = [
      { discrimination: 1, difficulty: -1 },
      { discrimination: 1, difficulty: 1 },
    ];
    const into = new Float64Array(rule.nodes.length);
    posteriorOverNodes([1, 0], symmetric, rule, into);
    let mean = 0;
    for (const [k, node] of rule.nodes.entries()) mean += (into[k] as number) * node;
    expect(mean).toBeCloseTo(0, 10);
  });

  it('moves the posterior up for a correct answer and down for a wrong one', () => {
    const into = new Float64Array(rule.nodes.length);
    const meanOf = (row: readonly Cell[]): number => {
      posteriorOverNodes(row, points, rule, into);
      let mean = 0;
      for (const [k, node] of rule.nodes.entries()) mean += (into[k] as number) * node;
      return mean;
    };
    expect(meanOf([1, 1, 1])).toBeGreaterThan(meanOf([1, 1, 0]));
    expect(meanOf([1, 1, 0])).toBeGreaterThan(meanOf([0, 0, 0]));
  });

  it('stays normalised for a perfect score on a long confident test', () => {
    const long: ItemPoint[] = Array.from({ length: 60 }, (_, index) => ({
      discrimination: 2.5,
      difficulty: -2 + (4 * index) / 59,
    }));
    const into = new Float64Array(rule.nodes.length);
    const marginal = posteriorOverNodes(
      Array.from({ length: 60 }, () => 1 as Cell),
      long,
      rule,
      into,
    );
    expect(totalOf(into)).toBeCloseTo(1, 12);
    expect(Number.isFinite(marginal)).toBe(true);
    for (const mass of into) expect(Number.isNaN(mass)).toBe(false);
  });
});

describe('expectationStep', () => {
  const rng = createRng(20260913);
  const abilities = Array.from({ length: 300 }, () => rng.nextNormal());
  const matrix = simulateMatrix(bank, abilities, 4471);

  it('conserves the observed totals across the nodes', () => {
    // The single strongest invariant available: spreading each respondent over
    // the ability range redistributes their response but cannot create or
    // destroy it, so each item's counts must still sum to what was observed.
    const counts = expectationStep(matrix, points, rule);
    for (let item = 0; item < matrix.itemCount; item += 1) {
      const { correct, answered } = matrix.itemScore(item);
      expect(totalOf(counts.answered[item] as Float64Array)).toBeCloseTo(answered, 8);
      expect(totalOf(counts.correct[item] as Float64Array)).toBeCloseTo(correct, 8);
    }
  });

  it('spreads exactly one unit of mass per respondent', () => {
    const counts = expectationStep(matrix, points, rule);
    expect(totalOf(counts.density)).toBeCloseTo(matrix.personCount, 8);
  });

  it('never counts more correct answers than responses at any node', () => {
    const counts = expectationStep(matrix, points, rule);
    for (let item = 0; item < matrix.itemCount; item += 1) {
      const answered = counts.answered[item] as Float64Array;
      const correct = counts.correct[item] as Float64Array;
      for (let k = 0; k < answered.length; k += 1) {
        expect(correct[k] as number).toBeLessThanOrEqual((answered[k] as number) + 1e-12);
        expect(correct[k] as number).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('agrees with the marginal log-likelihood computed on its own', () => {
    const counts = expectationStep(matrix, points, rule);
    expect(counts.logLikelihood).toBeCloseTo(marginalLogLikelihood(matrix, points, rule), 9);
  });

  it('does not depend on the order the respondents are stored in', () => {
    const rows = matrix.toArray();
    const reversed = new ResponseMatrix({
      rows: [...rows].reverse(),
      itemIds: matrix.itemIds,
    });
    const forward = expectationStep(matrix, points, rule);
    const backward = expectationStep(reversed, points, rule);
    expect(backward.logLikelihood).toBeCloseTo(forward.logLikelihood, 8);
    for (let item = 0; item < matrix.itemCount; item += 1) {
      const a = forward.answered[item] as Float64Array;
      const b = backward.answered[item] as Float64Array;
      for (let k = 0; k < a.length; k += 1) {
        expect(b[k] as number).toBeCloseTo(a[k] as number, 8);
      }
    }
  });

  it('leaves a skipped item out of its own counts but not out of the others', () => {
    const rows: Cell[][] = [
      [1, MISSING, 0],
      [1, 1, 1],
    ];
    const skipped = new ResponseMatrix({ rows });
    const counts = expectationStep(skipped, points, rule);
    expect(totalOf(counts.answered[0] as Float64Array)).toBeCloseTo(2, 10);
    expect(totalOf(counts.answered[1] as Float64Array)).toBeCloseTo(1, 10);
    expect(totalOf(counts.answered[2] as Float64Array)).toBeCloseTo(2, 10);
    expect(totalOf(counts.correct[1] as Float64Array)).toBeCloseTo(1, 10);
  });

  it('keeps perfect and zero scorers, which joint estimation cannot', () => {
    const rows: Cell[][] = [
      [1, 1, 1],
      [0, 0, 0],
      [1, 0, 0],
    ];
    const counts = expectationStep(new ResponseMatrix({ rows }), points, rule);
    expect(Number.isFinite(counts.logLikelihood)).toBe(true);
    expect(totalOf(counts.density)).toBeCloseTo(3, 10);
    // The perfect scorer's mass sits above the middle of the scale and the zero
    // scorer's below it, both finite, neither discarded.
    expect(totalOf(counts.answered[0] as Float64Array)).toBeCloseTo(3, 10);
  });

  it('rejects a parameter vector of the wrong length', () => {
    expect(() => expectationStep(matrix, points.slice(0, 2), rule)).toThrow(RangeError);
  });

  it('reaches the same counts on a grid rule as on a Gauss-Hermite rule', () => {
    // Different nodes, so the per-node counts differ, but the conserved totals
    // and the marginal likelihood should agree to quadrature accuracy.
    const grid = normalGridRule(0, 1, 61, 5);
    const gauss = expectationStep(matrix, points, rule);
    const gridded = expectationStep(matrix, points, grid);
    expect(gridded.logLikelihood).toBeCloseTo(gauss.logLikelihood, 4);
  });
});

describe('pointProbability', () => {
  it('agrees with the logistic form used elsewhere in the engine', () => {
    for (const theta of [-3, -0.5, 0, 1.2, 4]) {
      for (const item of points) {
        expect(pointProbability(item, theta)).toBeCloseTo(
          logistic(item.discrimination * (theta - item.difficulty)),
          14,
        );
      }
    }
  });
});
