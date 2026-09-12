import { describe, expect, it } from 'vitest';
import {
  invertMatrix,
  solveLinear,
  transposeProduct,
  weightedCrossProduct,
} from '../src/core/linear.js';

/** Multiply two matrices given as rows, for checking an inverse. */
function multiply(a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number[][] {
  const inner = b.length;
  const width = (b[0] as readonly number[]).length;
  return a.map((row) =>
    Array.from({ length: width }, (_, column) => {
      let total = 0;
      for (let k = 0; k < inner; k += 1) {
        total += (row[k] as number) * ((b[k] as readonly number[])[column] as number);
      }
      return total;
    }),
  );
}

function apply(a: readonly (readonly number[])[], x: readonly number[]): number[] {
  return a.map((row) => row.reduce((total, value, index) => total + value * (x[index] as number), 0));
}

describe('solveLinear', () => {
  it('solves a system with an exact integer answer', () => {
    const a = [
      [2, 1, -1],
      [-3, -1, 2],
      [-2, 1, 2],
    ];
    const solution = solveLinear(a, [8, -11, -3]);
    expect(solution[0]).toBeCloseTo(2, 12);
    expect(solution[1]).toBeCloseTo(3, 12);
    expect(solution[2]).toBeCloseTo(-1, 12);
  });

  it('reproduces the right-hand side when the solution is substituted back', () => {
    const a = [
      [4, -2, 1, 0.5],
      [-2, 6, -1, 1],
      [1, -1, 5, -2],
      [0.5, 1, -2, 3],
    ];
    const b = [1, -3, 7, 0.25];
    expect(apply(a, solveLinear(a, b))).toEqual(b.map((value) => expect.closeTo(value, 10)));
  });

  it('pivots rather than dividing by a zero on the diagonal', () => {
    // Without partial pivoting this divides by zero on the very first step.
    const a = [
      [0, 2],
      [1, 1],
    ];
    const solution = solveLinear(a, [4, 3]);
    expect(solution[0]).toBeCloseTo(1, 12);
    expect(solution[1]).toBeCloseTo(2, 12);
  });

  it('stays accurate when the diagonal is tiny beside the rest', () => {
    const epsilon = 1e-14;
    const a = [
      [epsilon, 1],
      [1, 1],
    ];
    const solution = solveLinear(a, [1, 2]);
    expect(apply(a, solution)[0]).toBeCloseTo(1, 10);
    expect(apply(a, solution)[1]).toBeCloseTo(2, 10);
  });

  it('refuses a singular system rather than returning noise', () => {
    expect(() =>
      solveLinear(
        [
          [1, 2],
          [2, 4],
        ],
        [1, 2],
      ),
    ).toThrow(/singular/);
  });

  it('rejects a ragged, empty or non-finite matrix and a mismatched vector', () => {
    expect(() => solveLinear([], [])).toThrow(/the matrix is empty/);
    expect(() => solveLinear([[1, 2]], [1])).toThrow(/row 0 has 2 entries/);
    expect(() => solveLinear([[Number.NaN]], [1])).toThrow(/entry \(0, 0\)/);
    expect(() => solveLinear([[1]], [1, 2])).toThrow(/2 entries, expected 1/);
  });

  it('leaves the matrix it was given untouched', () => {
    const a = [
      [2, 1],
      [1, 3],
    ];
    const copy = a.map((row) => [...row]);
    solveLinear(a, [1, 1]);
    expect(a).toEqual(copy);
  });
});

describe('invertMatrix', () => {
  it('inverts the identity to itself', () => {
    expect(invertMatrix([
      [1, 0],
      [0, 1],
    ])).toEqual([
      [1, 0],
      [0, 1],
    ]);
  });

  it('matches the closed form for a two-by-two', () => {
    // [a b; c d]^-1 = 1/(ad - bc) [d -b; -c a]
    const inverse = invertMatrix([
      [4, 7],
      [2, 6],
    ]);
    const determinant = 4 * 6 - 7 * 2;
    expect(inverse[0]?.[0]).toBeCloseTo(6 / determinant, 12);
    expect(inverse[0]?.[1]).toBeCloseTo(-7 / determinant, 12);
    expect(inverse[1]?.[0]).toBeCloseTo(-2 / determinant, 12);
    expect(inverse[1]?.[1]).toBeCloseTo(4 / determinant, 12);
  });

  it('multiplies back to the identity', () => {
    const a = [
      [3, 1, -2, 0],
      [1, 5, 0, 2],
      [-2, 0, 4, 1],
      [0, 2, 1, 6],
    ];
    const product = multiply(a, invertMatrix(a));
    for (let row = 0; row < 4; row += 1) {
      for (let column = 0; column < 4; column += 1) {
        expect(product[row]?.[column]).toBeCloseTo(row === column ? 1 : 0, 10);
      }
    }
  });

  it('keeps a symmetric matrix symmetric', () => {
    const a = [
      [5, 2, 1],
      [2, 4, -1],
      [1, -1, 3],
    ];
    const inverse = invertMatrix(a);
    expect(inverse[0]?.[1]).toBeCloseTo(inverse[1]?.[0] as number, 12);
    expect(inverse[0]?.[2]).toBeCloseTo(inverse[2]?.[0] as number, 12);
    expect(inverse[1]?.[2]).toBeCloseTo(inverse[2]?.[1] as number, 12);
  });

  it('is its own inverse when applied twice', () => {
    const a = [
      [2, -1],
      [3, 4],
    ];
    const back = invertMatrix(invertMatrix(a));
    expect(back[0]?.[0]).toBeCloseTo(2, 10);
    expect(back[1]?.[1]).toBeCloseTo(4, 10);
  });

  it('refuses a singular matrix', () => {
    expect(() =>
      invertMatrix([
        [1, 1],
        [1, 1],
      ]),
    ).toThrow(/singular/);
  });
});

describe('weightedCrossProduct', () => {
  it('reduces to the plain cross-product at unit weights', () => {
    const design = [
      [1, 2],
      [1, 3],
      [1, 5],
    ];
    const result = weightedCrossProduct(design, [1, 1, 1]);
    expect(result).toEqual([
      [3, 10],
      [10, 38],
    ]);
  });

  it('agrees with forming the weight matrix explicitly', () => {
    const design = [
      [1, 0.5, -2],
      [1, 1.5, 0],
      [1, -0.5, 3],
      [1, 2.5, 1],
    ];
    const weights = [0.2, 0.9, 0.05, 0.4];
    const result = weightedCrossProduct(design, weights);
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        let expected = 0;
        for (let row = 0; row < design.length; row += 1) {
          expected +=
            (weights[row] as number) *
            ((design[row] as number[])[i] as number) *
            ((design[row] as number[])[j] as number);
        }
        expect(result[i]?.[j]).toBeCloseTo(expected, 12);
      }
    }
  });

  it('is symmetric by construction', () => {
    const design = [
      [1, 3, 9],
      [1, -2, 4],
      [1, 0.5, 0.25],
    ];
    const result = weightedCrossProduct(design, [0.3, 0.7, 0.1]);
    expect(result[0]?.[2]).toBeCloseTo(result[2]?.[0] as number, 14);
    expect(result[1]?.[2]).toBeCloseTo(result[2]?.[1] as number, 14);
  });

  it('drops a row of zero weight entirely', () => {
    const design = [
      [1, 2],
      [1, 1000],
      [1, 3],
    ];
    const withRow = weightedCrossProduct(design, [1, 0, 1]);
    const without = weightedCrossProduct([design[0] as number[], design[2] as number[]], [1, 1]);
    expect(withRow).toEqual(without);
  });

  it('rejects an empty design, a ragged row or mismatched weights', () => {
    expect(() => weightedCrossProduct([], [])).toThrow(/design matrix is empty/);
    expect(() => weightedCrossProduct([[1, 2]], [1, 1])).toThrow(/2 weights for 1 rows/);
    expect(() => weightedCrossProduct([[1, 2], [1]], [1, 1])).toThrow(/row 1 has 1 of 2 terms/);
  });
});

describe('transposeProduct', () => {
  it('sums the design rows weighted by the vector', () => {
    const design = [
      [1, 2],
      [1, 3],
      [1, 5],
    ];
    expect(transposeProduct(design, [2, -1, 4])).toEqual([5, 21]);
  });

  it('is zero for a zero vector', () => {
    expect(transposeProduct([[1, 2], [3, 4]], [0, 0])).toEqual([0, 0]);
  });

  it('rejects an empty design or a mismatched vector', () => {
    expect(() => transposeProduct([], [])).toThrow(/design matrix is empty/);
    expect(() => transposeProduct([[1]], [1, 2])).toThrow(/2 entries for 1 rows/);
  });
});
