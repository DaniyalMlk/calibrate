import { describe, expect, it } from 'vitest';
import { minimiseSimplex } from '../src/core/simplex.js';

/** Rosenbrock's banana: a curved, narrow valley with its minimum at (1, 1). */
function rosenbrock(point: readonly number[]): number {
  const x = point[0] as number;
  const y = point[1] as number;
  return (1 - x) ** 2 + 100 * (y - x * x) ** 2;
}

describe('minimiseSimplex', () => {
  it('finds the minimum of a quadratic', () => {
    const result = minimiseSimplex(
      (point) => ((point[0] as number) - 3) ** 2 + ((point[1] as number) + 2) ** 2 + 5,
      [0, 0],
    );
    expect(result.point[0]).toBeCloseTo(3, 6);
    expect(result.point[1]).toBeCloseTo(-2, 6);
    expect(result.value).toBeCloseTo(5, 9);
    expect(result.converged).toBe(true);
  });

  it('follows the Rosenbrock valley from the standard hard start', () => {
    const result = minimiseSimplex(rosenbrock, [-1.2, 1], {
      maxEvaluations: 20000,
      tolerance: 1e-16,
      pointTolerance: 1e-12,
    });
    expect(result.point[0]).toBeCloseTo(1, 6);
    expect(result.point[1]).toBeCloseTo(1, 6);
    expect(result.value).toBeLessThan(1e-12);
    expect(result.converged).toBe(true);
  });

  it('does not stop where the vertices straddle the optimum', () => {
    // The regression this exists for. A one-dimensional simplex reaching
    // {1.4, 1.6} on (x - 1.5)^2 has a value spread of exactly zero while
    // sitting a tenth away from the answer; terminating on the value spread
    // alone reports convergence there.
    const result = minimiseSimplex((point) => ((point[0] as number) - 1.5) ** 2, [10]);
    expect(result.point[0]).toBeCloseTo(1.5, 8);
    expect(result.converged).toBe(true);
  });

  it('works in one dimension and in four', () => {
    const one = minimiseSimplex((point) => Math.abs((point[0] as number) - 2) ** 1.5, [-5]);
    expect(one.point[0]).toBeCloseTo(2, 4);

    const target = [1, -2, 3, 0.5];
    const four = minimiseSimplex(
      (point) => point.reduce((total, value, index) => total + (value - (target[index] as number)) ** 2, 0),
      [0, 0, 0, 0],
      { maxEvaluations: 20000 },
    );
    for (let axis = 0; axis < target.length; axis += 1) {
      expect(four.point[axis]).toBeCloseTo(target[axis] as number, 5);
    }
  });

  it('steps away from a region where the objective is not finite', () => {
    // A trial point straying outside the domain must not poison the ordering.
    const result = minimiseSimplex(
      (point) =>
        (point[0] as number) < 0
          ? Number.NaN
          : ((point[0] as number) - 2) ** 2 + ((point[1] as number) - 1) ** 2,
      [5, 5],
    );
    expect(result.point[0]).toBeCloseTo(2, 5);
    expect(result.point[1]).toBeCloseTo(1, 5);
  });

  it('reports non-convergence rather than a wrong answer when the budget runs out', () => {
    const result = minimiseSimplex(rosenbrock, [-1.2, 1], { maxEvaluations: 10 });
    expect(result.converged).toBe(false);
    expect(result.evaluations).toBeLessThanOrEqual(10);
  });

  it('stops on a plateau, where the points never converge but the values do', () => {
    // A flat objective has no unique minimiser; terminating on the value spread
    // is what makes this return at all.
    const result = minimiseSimplex(() => 7, [0, 0]);
    expect(result.value).toBe(7);
    expect(result.converged).toBe(true);
  });

  it('never reports a value worse than the starting point', () => {
    const objective = (point: readonly number[]): number =>
      ((point[0] as number) - 1) ** 2 + ((point[1] as number) - 1) ** 2;
    const start = [8, -6];
    const result = minimiseSimplex(objective, start);
    expect(result.value).toBeLessThanOrEqual(objective(start));
  });

  it('returns a copy, not the internal vertex', () => {
    const result = minimiseSimplex((point) => (point[0] as number) ** 2, [1]);
    const mutable = result.point as number[];
    const before = mutable[0] as number;
    mutable[0] = 99;
    expect(minimiseSimplex((point) => (point[0] as number) ** 2, [1]).point[0]).toBeCloseTo(
      before,
      9,
    );
  });

  it('validates its arguments', () => {
    expect(() => minimiseSimplex((point) => point[0] as number, [])).toThrow(/must not be empty/);
    expect(() => minimiseSimplex((point) => point[0] as number, [Number.NaN])).toThrow(
      /start\[0\]/,
    );
    expect(() => minimiseSimplex((point) => point[0] as number, [0], { step: 0 })).toThrow(
      /step must be positive/,
    );
    expect(() =>
      minimiseSimplex((point) => point[0] as number, [0, 0], { maxEvaluations: 2 }),
    ).toThrow(/at least 3 evaluations/);
  });
});
