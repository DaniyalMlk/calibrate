import { requireFinite } from './numeric.js';

export interface SimplexOptions {
  /** Absolute tolerance on the spread of function values across the simplex. Default 1e-12. */
  readonly tolerance?: number;
  /**
   * Absolute tolerance on the width of the simplex in the domain. Default 1e-10.
   *
   * Checked *alongside* the value tolerance, not instead of it. See the note on
   * termination in `minimiseSimplex`.
   */
  readonly pointTolerance?: number;
  /** Maximum function evaluations. Default 4000. */
  readonly maxEvaluations?: number;
  /**
   * Initial step away from the starting point along each coordinate.
   *
   * The simplex is built by displacing the start along each axis in turn. Too
   * small a step and the search starts inside a region the objective cannot
   * distinguish; too large and it steps outside the domain the objective is
   * defined on. Default 0.1.
   */
  readonly step?: number;
}

export interface SimplexResult {
  /** The best point found. */
  readonly point: readonly number[];
  /** The objective there. */
  readonly value: number;
  /** True when the simplex collapsed to within tolerance before the budget ran out. */
  readonly converged: boolean;
  /** Function evaluations performed. */
  readonly evaluations: number;
}

/** Standard Nelder-Mead coefficients: reflection, expansion, contraction, shrink. */
const ALPHA = 1;
const GAMMA = 2;
const RHO = 0.5;
const SIGMA = 0.5;

/**
 * Nelder-Mead simplex minimisation.
 *
 * A derivative-free method: it maintains `n + 1` points in `n` dimensions and
 * repeatedly reflects the worst of them through the centroid of the rest,
 * expanding when that helps and contracting when it does not.
 *
 * Derivative-free is the right choice for the linking objectives. Haebara and
 * Stocking-Lord both sum squared differences between response functions
 * evaluated over a quadrature grid, and their gradients with respect to the
 * linking coefficients are long chain-rule expressions through every item's
 * response function. Writing those by hand is a large surface for a sign error
 * that would show up only as slow convergence, and finite-differencing them
 * costs about as many evaluations as this does. The objective is smooth and
 * two-dimensional, which is exactly where Nelder-Mead is comfortable.
 *
 * Termination requires the simplex to be small in *both* the objective and the
 * domain, and both halves are load-bearing.
 *
 * Testing only the spread of function values is the tempting simplification,
 * and it is wrong in a way that is easy to miss: vertices straddling the
 * optimum symmetrically have equal objectives while sitting far apart. A
 * one-dimensional search on `(x - 1.5)^2` whose simplex reaches `{1.4, 1.6}`
 * has a value spread of exactly zero and an answer wrong in the first decimal,
 * and it will report convergence the moment it gets there.
 *
 * Testing only the spread of the points is wrong in the other direction: on a
 * flat valley the vertices can stay far apart forever while every one of them
 * sits at essentially the same objective, and no further movement would improve
 * anything.
 */
export function minimiseSimplex(
  objective: (point: readonly number[]) => number,
  start: readonly number[],
  options: SimplexOptions = {},
): SimplexResult {
  const dimension = start.length;
  if (dimension === 0) throw new RangeError('minimiseSimplex: start must not be empty');
  start.forEach((value, index) => requireFinite(value, `start[${index}]`));

  const tolerance = options.tolerance ?? 1e-12;
  const pointTolerance = options.pointTolerance ?? 1e-10;
  const maxEvaluations = options.maxEvaluations ?? 4000;
  const step = options.step ?? 0.1;
  if (step <= 0) {
    throw new RangeError(`minimiseSimplex: step must be positive, received ${step}`);
  }
  if (maxEvaluations < dimension + 1) {
    throw new RangeError(
      `minimiseSimplex: maxEvaluations must allow at least ${dimension + 1} evaluations`,
    );
  }

  let evaluations = 0;
  const evaluate = (point: readonly number[]): number => {
    evaluations += 1;
    const value = objective(point);
    // A non-finite objective is treated as maximally bad rather than allowed to
    // poison the ordering. The simplex will simply move away from it, which is
    // the behaviour wanted when a trial point strays outside the region where
    // the objective is defined at all.
    return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
  };

  // Build the initial simplex: the start, plus one displacement along each axis.
  const points: number[][] = [[...start]];
  for (let axis = 0; axis < dimension; axis += 1) {
    const vertex = [...start];
    vertex[axis] = (vertex[axis] as number) + step;
    points.push(vertex);
  }
  let values = points.map((point) => evaluate(point));

  const order = (): void => {
    const indices = points.map((_, index) => index);
    indices.sort((left, right) => (values[left] as number) - (values[right] as number));
    const sortedPoints = indices.map((index) => points[index] as number[]);
    const sortedValues = indices.map((index) => values[index] as number);
    for (let index = 0; index < points.length; index += 1) {
      points[index] = sortedPoints[index] as number[];
      values[index] = sortedValues[index] as number;
    }
  };

  order();

  let converged = false;
  while (evaluations < maxEvaluations) {
    const best = values[0] as number;
    const worst = values[values.length - 1] as number;
    if (Number.isFinite(best) && Number.isFinite(worst) && worst - best <= tolerance) {
      // Widest displacement of any vertex from the best one, along any axis.
      let width = 0;
      const bestVertex = points[0] as number[];
      for (let index = 1; index < points.length; index += 1) {
        const vertex = points[index] as number[];
        for (let axis = 0; axis < dimension; axis += 1) {
          const gap = Math.abs((vertex[axis] as number) - (bestVertex[axis] as number));
          if (gap > width) width = gap;
        }
      }
      if (width <= pointTolerance) {
        converged = true;
        break;
      }
    }

    // Centroid of every vertex but the worst.
    const centroid = new Array<number>(dimension).fill(0);
    for (let index = 0; index < points.length - 1; index += 1) {
      const vertex = points[index] as number[];
      for (let axis = 0; axis < dimension; axis += 1) {
        centroid[axis] = (centroid[axis] as number) + (vertex[axis] as number);
      }
    }
    for (let axis = 0; axis < dimension; axis += 1) {
      centroid[axis] = (centroid[axis] as number) / (points.length - 1);
    }

    const worstPoint = points[points.length - 1] as number[];
    const along = (coefficient: number): number[] =>
      centroid.map(
        (value, axis) => value + coefficient * (value - (worstPoint[axis] as number)),
      );

    const reflected = along(ALPHA);
    const reflectedValue = evaluate(reflected);

    if (reflectedValue < (values[0] as number)) {
      // Better than the best: try going further in the same direction.
      const expanded = along(GAMMA);
      const expandedValue = evaluate(expanded);
      if (expandedValue < reflectedValue) {
        points[points.length - 1] = expanded;
        values[values.length - 1] = expandedValue;
      } else {
        points[points.length - 1] = reflected;
        values[values.length - 1] = reflectedValue;
      }
      order();
      continue;
    }

    if (reflectedValue < (values[values.length - 2] as number)) {
      // Better than the second worst: accept and move on.
      points[points.length - 1] = reflected;
      values[values.length - 1] = reflectedValue;
      order();
      continue;
    }

    // Contract, on whichever side of the centroid is currently better.
    const useReflected = reflectedValue < (values[values.length - 1] as number);
    const contracted = useReflected
      ? centroid.map((value, axis) => value + RHO * ((reflected[axis] as number) - value))
      : centroid.map((value, axis) => value + RHO * ((worstPoint[axis] as number) - value));
    const contractedValue = evaluate(contracted);
    const comparison = useReflected ? reflectedValue : (values[values.length - 1] as number);

    if (contractedValue < comparison) {
      points[points.length - 1] = contracted;
      values[values.length - 1] = contractedValue;
      order();
      continue;
    }

    // Nothing worked: shrink the whole simplex towards the best vertex.
    const bestPoint = points[0] as number[];
    for (let index = 1; index < points.length; index += 1) {
      const vertex = points[index] as number[];
      points[index] = vertex.map(
        (value, axis) => (bestPoint[axis] as number) + SIGMA * (value - (bestPoint[axis] as number)),
      );
    }
    values = points.map((point) => evaluate(point));
    order();
  }

  return {
    point: [...(points[0] as number[])],
    value: values[0] as number,
    converged,
    evaluations,
  };
}
