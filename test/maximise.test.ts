import { describe, expect, it } from 'vitest';
import { normalGaussHermiteRule } from '../src/core/quadrature.js';
import { pointProbability, type ItemPoint } from '../src/calibration/expected.js';
import {
  countDerivatives,
  countStandardErrors,
  maximiseItem,
  type MaximiseOptions,
} from '../src/calibration/maximise.js';

const rule = normalGaussHermiteRule(0, 1, 41);

const twoPlOptions: MaximiseOptions = {
  model: '2pl',
  discriminationBounds: [0.05, 4],
  difficultyBounds: [-6, 6],
};
const raschOptions: MaximiseOptions = { ...twoPlOptions, model: 'rasch' };

/**
 * Counts with no sampling noise in them at all: every node sees `perNode`
 * respondents and exactly the fraction the truth predicts answers correctly.
 * The maximiser must then return the generating parameters, because they are
 * the exact maximiser of this likelihood and not merely close to it.
 */
function exactCounts(
  truth: ItemPoint,
  perNode = 100,
): { answered: Float64Array; correct: Float64Array } {
  const answered = new Float64Array(rule.nodes.length);
  const correct = new Float64Array(rule.nodes.length);
  for (const [k, node] of rule.nodes.entries()) {
    const n = perNode * (rule.weights[k] as number) * rule.nodes.length;
    answered[k] = n;
    correct[k] = n * pointProbability(truth, node);
  }
  return { answered, correct };
}

describe('countDerivatives', () => {
  const truth: ItemPoint = { discrimination: 1.3, difficulty: 0.4 };
  const { answered, correct } = exactCounts(truth);
  const at: ItemPoint = { discrimination: 0.9, difficulty: -0.3 };

  it('has a gradient that matches a central difference of the value', () => {
    const h = 1e-5;
    const base = countDerivatives(at, rule, answered, correct);

    const upA = countDerivatives(
      { discrimination: at.discrimination + h, difficulty: at.difficulty },
      rule,
      answered,
      correct,
    ).value;
    const downA = countDerivatives(
      { discrimination: at.discrimination - h, difficulty: at.difficulty },
      rule,
      answered,
      correct,
    ).value;
    expect(base.gradientA).toBeCloseTo((upA - downA) / (2 * h), 5);

    const upB = countDerivatives(
      { discrimination: at.discrimination, difficulty: at.difficulty + h },
      rule,
      answered,
      correct,
    ).value;
    const downB = countDerivatives(
      { discrimination: at.discrimination, difficulty: at.difficulty - h },
      rule,
      answered,
      correct,
    ).value;
    expect(base.gradientB).toBeCloseTo((upB - downB) / (2 * h), 5);
  });

  it('has a Hessian that matches a central difference of the gradient', () => {
    const h = 1e-5;
    const base = countDerivatives(at, rule, answered, correct);

    const gradAUpA = countDerivatives(
      { discrimination: at.discrimination + h, difficulty: at.difficulty },
      rule,
      answered,
      correct,
    ).gradientA;
    const gradADownA = countDerivatives(
      { discrimination: at.discrimination - h, difficulty: at.difficulty },
      rule,
      answered,
      correct,
    ).gradientA;
    expect(base.hessianAA).toBeCloseTo((gradAUpA - gradADownA) / (2 * h), 4);

    const gradBUpB = countDerivatives(
      { discrimination: at.discrimination, difficulty: at.difficulty + h },
      rule,
      answered,
      correct,
    ).gradientB;
    const gradBDownB = countDerivatives(
      { discrimination: at.discrimination, difficulty: at.difficulty - h },
      rule,
      answered,
      correct,
    ).gradientB;
    expect(base.hessianBB).toBeCloseTo((gradBUpB - gradBDownB) / (2 * h), 4);

    // The cross derivative, taken both ways round: equality is Clairaut's
    // theorem, and disagreement would mean one of the two is wrong.
    const gradAUpB = countDerivatives(
      { discrimination: at.discrimination, difficulty: at.difficulty + h },
      rule,
      answered,
      correct,
    ).gradientA;
    const gradADownB = countDerivatives(
      { discrimination: at.discrimination, difficulty: at.difficulty - h },
      rule,
      answered,
      correct,
    ).gradientA;
    expect(base.hessianAB).toBeCloseTo((gradAUpB - gradADownB) / (2 * h), 4);
  });

  it('vanishes at the generating parameters when the counts carry no noise', () => {
    const d = countDerivatives(truth, rule, answered, correct);
    expect(Math.abs(d.gradientA)).toBeLessThan(1e-9);
    expect(Math.abs(d.gradientB)).toBeLessThan(1e-9);
  });

  it('ignores nodes that nobody reached', () => {
    const sparse = new Float64Array(rule.nodes.length);
    const sparseCorrect = new Float64Array(rule.nodes.length);
    sparse[20] = 10;
    sparseCorrect[20] = 6;
    const d = countDerivatives(at, rule, sparse, sparseCorrect);
    expect(Number.isFinite(d.value)).toBe(true);
    expect(Number.isFinite(d.gradientA)).toBe(true);
  });
});

describe('maximiseItem', () => {
  it('recovers the generating difficulty exactly under the Rasch model', () => {
    for (const b of [-1.5, -0.4, 0, 0.7, 2.1]) {
      const truth: ItemPoint = { discrimination: 1, difficulty: b };
      const { answered, correct } = exactCounts(truth);
      const result = maximiseItem(
        { discrimination: 1, difficulty: 0 },
        rule,
        answered,
        correct,
        raschOptions,
      );
      expect(result.stationary).toBe(true);
      expect(result.item.difficulty).toBeCloseTo(b, 8);
      expect(result.item.discrimination).toBe(1);
    }
  });

  it('recovers both generating parameters exactly under the 2PL', () => {
    for (const truth of [
      { discrimination: 0.7, difficulty: -1.2 },
      { discrimination: 1.0, difficulty: 0 },
      { discrimination: 1.8, difficulty: 0.9 },
      { discrimination: 2.4, difficulty: -0.5 },
    ] satisfies ItemPoint[]) {
      const { answered, correct } = exactCounts(truth);
      const result = maximiseItem(
        { discrimination: 1, difficulty: 0 },
        rule,
        answered,
        correct,
        twoPlOptions,
      );
      expect(result.stationary).toBe(true);
      expect(result.item.discrimination).toBeCloseTo(truth.discrimination, 6);
      expect(result.item.difficulty).toBeCloseTo(truth.difficulty, 6);
    }
  });

  it('reaches the same optimum from very different starting points', () => {
    const truth: ItemPoint = { discrimination: 1.5, difficulty: 0.6 };
    const { answered, correct } = exactCounts(truth);
    const starts: ItemPoint[] = [
      { discrimination: 0.2, difficulty: -3 },
      { discrimination: 3.5, difficulty: 3 },
      { discrimination: 1, difficulty: 0 },
    ];
    for (const start of starts) {
      const result = maximiseItem(start, rule, answered, correct, {
        ...twoPlOptions,
        maxIterations: 200,
      });
      expect(result.item.discrimination).toBeCloseTo(truth.discrimination, 4);
      expect(result.item.difficulty).toBeCloseTo(truth.difficulty, 4);
    }
  });

  it('does not decrease the log-likelihood it is maximising', () => {
    const truth: ItemPoint = { discrimination: 1.2, difficulty: -0.8 };
    const { answered, correct } = exactCounts(truth, 40);
    const start: ItemPoint = { discrimination: 2.8, difficulty: 2.5 };
    const before = countDerivatives(start, rule, answered, correct).value;
    const result = maximiseItem(start, rule, answered, correct, twoPlOptions);
    const after = countDerivatives(result.item, rule, answered, correct).value;
    expect(after).toBeGreaterThan(before);
  });

  it('sits at a maximum, not merely a stationary point', () => {
    const truth: ItemPoint = { discrimination: 1.1, difficulty: 0.3 };
    const { answered, correct } = exactCounts(truth);
    const result = maximiseItem(
      { discrimination: 1, difficulty: 0 },
      rule,
      answered,
      correct,
      twoPlOptions,
    );
    const peak = countDerivatives(result.item, rule, answered, correct).value;
    for (const [da, db] of [
      [0.05, 0],
      [-0.05, 0],
      [0, 0.05],
      [0, -0.05],
      [0.04, 0.04],
      [-0.04, -0.04],
    ]) {
      const nudged = countDerivatives(
        {
          discrimination: result.item.discrimination + (da as number),
          difficulty: result.item.difficulty + (db as number),
        },
        rule,
        answered,
        correct,
      ).value;
      expect(nudged).toBeLessThan(peak);
    }
  });

  it('reports a parameter pinned to a bound rather than pretending it converged', () => {
    // Counts that no admissible item can fit: everyone passes below the middle
    // of the scale and nobody passes above it, which wants a large negative
    // discrimination the bounds do not allow.
    const answered = new Float64Array(rule.nodes.length).fill(20);
    const correct = new Float64Array(rule.nodes.length);
    for (const [k, node] of rule.nodes.entries()) correct[k] = node < 0 ? 20 : 0;
    const result = maximiseItem(
      { discrimination: 1, difficulty: 0 },
      rule,
      answered,
      correct,
      twoPlOptions,
    );
    expect(result.atBound).toBe(true);
    expect(result.item.discrimination).toBeGreaterThanOrEqual(0.05);
    expect(result.item.difficulty).toBeGreaterThanOrEqual(-6);
  });

  it('holds the discrimination at one under the Rasch model whatever it starts at', () => {
    const truth: ItemPoint = { discrimination: 1, difficulty: 0.5 };
    const { answered, correct } = exactCounts(truth);
    const result = maximiseItem(
      { discrimination: 2.7, difficulty: 0 },
      rule,
      answered,
      correct,
      raschOptions,
    );
    expect(result.item.discrimination).toBe(1);
  });
});

describe('countStandardErrors', () => {
  const truth: ItemPoint = { discrimination: 1.4, difficulty: 0.2 };

  it('shrinks like one over the square root of the sample', () => {
    const small = exactCounts(truth, 100);
    const large = exactCounts(truth, 400);
    const a = countStandardErrors(truth, rule, small.answered, small.correct, '2pl');
    const b = countStandardErrors(truth, rule, large.answered, large.correct, '2pl');
    expect(b.difficulty).toBeCloseTo(a.difficulty / 2, 6);
    expect(b.discrimination).toBeCloseTo(a.discrimination / 2, 6);
  });

  it('reports no discrimination error under the Rasch model, where it is not estimated', () => {
    const { answered, correct } = exactCounts({ discrimination: 1, difficulty: 0.2 });
    const errors = countStandardErrors(
      { discrimination: 1, difficulty: 0.2 },
      rule,
      answered,
      correct,
      'rasch',
    );
    expect(errors.discrimination).toBe(0);
    expect(errors.difficulty).toBeGreaterThan(0);
    expect(Number.isFinite(errors.difficulty)).toBe(true);
  });

  it('is infinite where no counts constrain the item at all', () => {
    const empty = new Float64Array(rule.nodes.length);
    const errors = countStandardErrors(truth, rule, empty, empty, '2pl');
    expect(errors.difficulty).toBe(Number.POSITIVE_INFINITY);
  });

  it('agrees with the inverse curvature computed by finite differences', () => {
    const rasch: ItemPoint = { discrimination: 1, difficulty: 0.2 };
    const { answered, correct } = exactCounts(rasch, 250);
    const errors = countStandardErrors(rasch, rule, answered, correct, 'rasch');
    const h = 1e-4;
    const at = (difficulty: number): number =>
      countDerivatives({ discrimination: 1, difficulty }, rule, answered, correct).value;
    const curvature =
      -(at(rasch.difficulty + h) - 2 * at(rasch.difficulty) + at(rasch.difficulty - h)) / (h * h);
    expect(errors.difficulty).toBeCloseTo(1 / Math.sqrt(curvature), 6);
  });

  it('agrees with the inverse Hessian block under the 2PL', () => {
    const { answered, correct } = exactCounts(truth, 250);
    const errors = countStandardErrors(truth, rule, answered, correct, '2pl');
    const d = countDerivatives(truth, rule, answered, correct);
    // Invert the 2x2 observed information by hand and compare the diagonal.
    const [iaa, ibb, iab] = [-d.hessianAA, -d.hessianBB, -d.hessianAB];
    const determinant = iaa * ibb - iab * iab;
    expect(errors.discrimination).toBeCloseTo(Math.sqrt(ibb / determinant), 12);
    expect(errors.difficulty).toBeCloseTo(Math.sqrt(iaa / determinant), 12);
  });
});
