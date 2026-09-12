/**
 * Dense linear algebra, at the scale the engine actually needs it.
 *
 * Nothing here is larger than a handful of rows: a logistic regression with four
 * terms, a covariance matrix of the same size. That is what makes a direct
 * method the right choice — Gaussian elimination with partial pivoting is exact
 * up to rounding on a system this small, and the iterative methods that beat it
 * on large sparse problems would be slower and less accurate here.
 *
 * The pivoting is not optional. Elimination without it divides by whatever
 * happens to sit on the diagonal, and a weighted least squares matrix routinely
 * has a near-zero there when one predictor is nearly redundant — which is
 * exactly the situation a DIF analysis produces when one group answers an item
 * almost uniformly.
 */

import { requireFinite } from './numeric.js';

/** A rectangular matrix as an array of rows. */
export type Matrix = readonly (readonly number[])[];

function checkSquare(matrix: Matrix, caller: string): number {
  const order = matrix.length;
  if (order === 0) throw new RangeError(`${caller}: the matrix is empty`);
  for (const [index, row] of matrix.entries()) {
    if (row.length !== order) {
      throw new RangeError(
        `${caller}: row ${index} has ${row.length} entries in a ${order}x${order} matrix`,
      );
    }
    for (const [column, value] of row.entries()) {
      requireFinite(value, `${caller}: entry (${index}, ${column})`);
    }
  }
  return order;
}

/**
 * Solve `A x = B` for one or more right-hand sides, by elimination.
 *
 * `B` is given as columns, so a single system passes one column and an inverse
 * passes the identity. Both go through the same elimination, which is the point:
 * factoring once and substituting many times is what makes the inverse cost no
 * more than a handful of solves.
 */
function eliminate(matrix: Matrix, columns: Matrix, caller: string): number[][] {
  const order = checkSquare(matrix, caller);
  for (const [index, column] of columns.entries()) {
    if (column.length !== order) {
      throw new RangeError(
        `${caller}: right-hand side ${index} has ${column.length} entries, expected ${order}`,
      );
    }
  }

  // Work on copies; the caller's matrix is readonly and stays that way.
  const a = matrix.map((row) => [...row]);
  const b = columns.map((column) => [...column]);

  for (let pivot = 0; pivot < order; pivot += 1) {
    let best = pivot;
    let magnitude = Math.abs(a[pivot]?.[pivot] as number);
    for (let row = pivot + 1; row < order; row += 1) {
      const candidate = Math.abs(a[row]?.[pivot] as number);
      if (candidate > magnitude) {
        best = row;
        magnitude = candidate;
      }
    }
    if (magnitude === 0) {
      throw new RangeError(
        `${caller}: the matrix is singular — column ${pivot} is a combination of the others`,
      );
    }
    if (best !== pivot) {
      const swap = a[pivot] as number[];
      a[pivot] = a[best] as number[];
      a[best] = swap;
      for (const column of b) {
        const held = column[pivot] as number;
        column[pivot] = column[best] as number;
        column[best] = held;
      }
    }

    const pivotRow = a[pivot] as number[];
    const pivotValue = pivotRow[pivot] as number;
    for (let row = 0; row < order; row += 1) {
      if (row === pivot) continue;
      const target = a[row] as number[];
      const factor = (target[pivot] as number) / pivotValue;
      if (factor === 0) continue;
      for (let column = pivot; column < order; column += 1) {
        target[column] = (target[column] as number) - factor * (pivotRow[column] as number);
      }
      for (const column of b) {
        column[row] = (column[row] as number) - factor * (column[pivot] as number);
      }
    }
  }

  for (let row = 0; row < order; row += 1) {
    const divisor = (a[row] as number[])[row] as number;
    for (const column of b) column[row] = (column[row] as number) / divisor;
  }
  return b;
}

/** Solve `A x = b` for a single right-hand side. */
export function solveLinear(matrix: Matrix, rhs: readonly number[]): number[] {
  const [solution] = eliminate(matrix, [rhs], 'solveLinear');
  return solution as number[];
}

/**
 * Invert a square matrix.
 *
 * Returned as rows, so `inverse[i][j]` is the usual entry. Used for the
 * covariance matrix of a set of regression coefficients, where the whole inverse
 * really is wanted rather than one solve: the diagonal gives the standard errors
 * and the off-diagonal entries say how far the estimates lean on each other.
 */
export function invertMatrix(matrix: Matrix): number[][] {
  const order = checkSquare(matrix, 'invertMatrix');
  const identity = Array.from({ length: order }, (_, column) =>
    Array.from({ length: order }, (_, row) => (row === column ? 1 : 0)),
  );
  const columns = eliminate(matrix, identity, 'invertMatrix');
  // `columns[j][i]` is entry (i, j) of the inverse; transpose into rows.
  return Array.from({ length: order }, (_, row) =>
    Array.from({ length: order }, (_, column) => (columns[column] as number[])[row] as number),
  );
}

/**
 * `X' W X` for a diagonal weight matrix, formed without building `W`.
 *
 * The design matrix of a DIF regression has a few columns and as many rows as
 * there are candidates, so forming the weighted cross-product directly is the
 * difference between a few hundred thousand multiplications and a matrix
 * product whose intermediate is the size of the sample squared.
 */
export function weightedCrossProduct(design: Matrix, weights: readonly number[]): number[][] {
  const rows = design.length;
  if (rows === 0) throw new RangeError('weightedCrossProduct: the design matrix is empty');
  if (weights.length !== rows) {
    throw new RangeError(
      `weightedCrossProduct: ${weights.length} weights for ${rows} rows`,
    );
  }
  const width = (design[0] as readonly number[]).length;
  const result = Array.from({ length: width }, () => new Array<number>(width).fill(0));

  for (let row = 0; row < rows; row += 1) {
    const values = design[row] as readonly number[];
    if (values.length !== width) {
      throw new RangeError(`weightedCrossProduct: row ${row} has ${values.length} of ${width} terms`);
    }
    const weight = weights[row] as number;
    if (weight === 0) continue;
    for (let i = 0; i < width; i += 1) {
      const scaled = weight * (values[i] as number);
      const target = result[i] as number[];
      // Symmetric, so only the upper triangle is accumulated and mirrored.
      for (let j = i; j < width; j += 1) {
        target[j] = (target[j] as number) + scaled * (values[j] as number);
      }
    }
  }

  for (let i = 0; i < width; i += 1) {
    for (let j = 0; j < i; j += 1) {
      (result[i] as number[])[j] = (result[j] as number[])[i] as number;
    }
  }
  return result;
}

/** `X' v` for a column `v`, the other half of a normal-equations step. */
export function transposeProduct(design: Matrix, vector: readonly number[]): number[] {
  const rows = design.length;
  if (rows === 0) throw new RangeError('transposeProduct: the design matrix is empty');
  if (vector.length !== rows) {
    throw new RangeError(`transposeProduct: ${vector.length} entries for ${rows} rows`);
  }
  const width = (design[0] as readonly number[]).length;
  const result = new Array<number>(width).fill(0);
  for (let row = 0; row < rows; row += 1) {
    const values = design[row] as readonly number[];
    const value = vector[row] as number;
    if (value === 0) continue;
    for (let i = 0; i < width; i += 1) {
      result[i] = (result[i] as number) + value * (values[i] as number);
    }
  }
  return result;
}
