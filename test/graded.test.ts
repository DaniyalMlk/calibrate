import { describe, expect, it } from 'vitest';
import {
  classifyGraded,
  generalizedMantel,
  LARGE_STANDARDIZED,
  NEGLIGIBLE_STANDARDIZED,
  standardizedMeanDifference,
} from '../src/dif/graded.js';
import { mantelHaenszel, standardizedDifference } from '../src/dif/mantel.js';
import { stratify, type DifObservation, type Group } from '../src/dif/strata.js';

type Table = readonly [number, number, number, number];

/** A dichotomous stratification, as counts of right and wrong in each group. */
function binary(tables: readonly Table[]) {
  const observations: DifObservation[] = [];
  for (const [index, [refRight, refWrong, focRight, focWrong]] of tables.entries()) {
    const push = (group: Group, score: number, times: number): void => {
      for (let i = 0; i < times; i += 1) observations.push({ group, match: index + 1, score });
    };
    push('reference', 1, refRight);
    push('reference', 0, refWrong);
    push('focal', 1, focRight);
    push('focal', 0, focWrong);
  }
  return stratify(observations);
}

/** A graded stratification, as a count vector per group per stratum. */
function graded(
  tables: readonly (readonly [readonly number[], readonly number[]])[],
  maxScore: number,
) {
  const observations: DifObservation[] = [];
  for (const [index, [reference, focal]] of tables.entries()) {
    for (const [score, count] of reference.entries()) {
      for (let i = 0; i < count; i += 1) {
        observations.push({ group: 'reference', match: index + 1, score });
      }
    }
    for (const [score, count] of focal.entries()) {
      for (let i = 0; i < count; i += 1) {
        observations.push({ group: 'focal', match: index + 1, score });
      }
    }
  }
  return stratify(observations, { maxScore });
}

/**
 * The Mantel-Haenszel chi-square without a continuity correction, computed the
 * classical way: over the reference group's count of correct answers.
 *
 * Deliberately a different route from the one under test, which sums the focal
 * group's total score. The two deviations are equal and opposite, so squaring
 * has to give the same number — and if the general form has an error in its
 * hypergeometric variance, this is what will catch it.
 */
function uncorrectedTwoByTwo(tables: readonly Table[]): number {
  let observed = 0;
  let expected = 0;
  let variance = 0;
  for (const [refRight, refWrong, focRight, focWrong] of tables) {
    const reference = refRight + refWrong;
    const focal = focRight + focWrong;
    const right = refRight + focRight;
    const wrong = refWrong + focWrong;
    const total = reference + focal;
    observed += refRight;
    expected += (reference * right) / total;
    if (total > 1) variance += (reference * focal * right * wrong) / (total * total * (total - 1));
  }
  return variance > 0 ? (observed - expected) ** 2 / variance : 0;
}

const REFERENCE: Table[] = [
  [12, 8, 6, 14],
  [25, 10, 18, 17],
  [30, 5, 22, 13],
  [9, 11, 4, 16],
];

describe('generalizedMantel', () => {
  it('reproduces the uncorrected Mantel-Haenszel chi-square on a binary item', () => {
    // The two statistics are computed by different routes — one over the focal
    // group's total score, the other over the reference group's correct count —
    // and on a two-category item they are the same number exactly.
    const result = generalizedMantel(binary(REFERENCE));
    expect(result.chiSquare).toBeCloseTo(13.644120426623465, 10);
  });

  it('agrees with the uncorrected two-by-two form on many random tables', () => {
    let seed = 20260912;
    const next = (limit: number): number => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed % limit;
    };
    for (let trial = 0; trial < 40; trial += 1) {
      const tables: Table[] = Array.from(
        { length: 3 + next(4) },
        () => [1 + next(30), 1 + next(30), 1 + next(30), 1 + next(30)] as Table,
      );
      expect(generalizedMantel(binary(tables)).chiSquare).toBeCloseTo(
        uncorrectedTwoByTwo(tables),
        8,
      );
    }
  });

  it('always sits above the continuity-corrected statistic', () => {
    const corrected = mantelHaenszel(binary(REFERENCE));
    const general = generalizedMantel(binary(REFERENCE));
    expect(general.chiSquare).toBeGreaterThan(corrected.chiSquare);
  });

  it('is zero when each stratum splits the scores evenly between the groups', () => {
    const even = graded(
      [
        [
          [4, 4, 4, 4],
          [4, 4, 4, 4],
        ],
        [
          [2, 6, 6, 2],
          [1, 3, 3, 1],
        ],
      ],
      3,
    );
    const result = generalizedMantel(even);
    expect(result.observed).toBeCloseTo(result.expected, 12);
    expect(result.chiSquare).toBeCloseTo(0, 12);
    expect(result.pValue).toBeCloseTo(1, 12);
    expect(result.favours).toBe('neither');
  });

  it('points at the group whose total falls short of expectation', () => {
    const againstFocal = graded(
      [
        [
          [2, 4, 6, 8],
          [8, 6, 4, 2],
        ],
        [
          [1, 3, 7, 9],
          [9, 7, 3, 1],
        ],
      ],
      3,
    );
    const result = generalizedMantel(againstFocal);
    expect(result.observed).toBeLessThan(result.expected);
    expect(result.favours).toBe('reference');
    expect(result.pValue).toBeLessThan(0.001);
  });

  it('respects unevenly spaced rubric levels', () => {
    // The reference group sits mostly at level 1 and the focal group mostly at
    // levels 2 and 3. Squeezing levels 1 and 2 together is not an affine change
    // of the metric — three levels are occupied — so it moves the statistic.
    const tables = [
      [
        [0, 16, 4, 0],
        [0, 4, 12, 4],
      ],
    ] as const;
    const evenly = generalizedMantel(graded(tables, 3));
    const compressed = generalizedMantel(graded(tables, 3), { categoryScores: [0, 1.4, 1.6, 3] });
    expect(evenly.chiSquare).toBeCloseTo(14.181818181818196, 8);
    expect(compressed.chiSquare).toBeCloseTo(7.2761194029850875, 8);
    expect(compressed.chiSquare).toBeLessThan(evenly.chiSquare);
  });

  it('is unchanged by a shift of the category scores and scales with a stretch', () => {
    const strata = graded(
      [
        [
          [3, 5, 7],
          [7, 5, 3],
        ],
        [
          [4, 4, 8],
          [8, 4, 4],
        ],
      ],
      2,
    );
    const plain = generalizedMantel(strata);
    const shifted = generalizedMantel(strata, { categoryScores: [10, 11, 12] });
    const stretched = generalizedMantel(strata, { categoryScores: [0, 2, 4] });
    // A chi-square is invariant to an affine change of the score metric: both
    // the deviation and its standard error move by the same factor.
    expect(shifted.chiSquare).toBeCloseTo(plain.chiSquare, 10);
    expect(stretched.chiSquare).toBeCloseTo(plain.chiSquare, 10);
  });

  it('rejects a score vector of the wrong length or a one-category item', () => {
    const strata = graded([[[1, 2, 3], [3, 2, 1]]], 2);
    expect(() => generalizedMantel(strata, { categoryScores: [0, 1] })).toThrow(
      /2 category scores for 3 categories/,
    );
    expect(() => generalizedMantel(strata, { categoryScores: [0, 1, Number.NaN] })).toThrow(
      /category score 2/,
    );
  });

  it('refuses a sample with nothing to compare', () => {
    expect(() => generalizedMantel(graded([[[1, 2, 3], [0, 0, 0]]], 2))).toThrow(
      /no stratum holds both groups/,
    );
  });
});

describe('standardizedMeanDifference', () => {
  it('reproduces the standardized proportion difference on a binary item', () => {
    // With scores of 0 and 1 the mean of the item score is the proportion
    // correct, so the two statistics are the same quantity.
    const strata = binary(REFERENCE);
    const smd = standardizedMeanDifference(strata);
    const proportion = standardizedDifference(strata);
    expect(smd.value).toBeCloseTo(proportion.value, 13);
    expect(smd.focalMean).toBeCloseTo(proportion.focalProportion, 13);
    expect(smd.referenceMean).toBeCloseTo(proportion.referenceProportion, 13);
  });

  it('is zero when the groups score alike at every level', () => {
    const result = standardizedMeanDifference(
      graded(
        [
          [
            [2, 2, 2],
            [4, 4, 4],
          ],
          [
            [6, 3, 3],
            [2, 1, 1],
          ],
        ],
        2,
      ),
    );
    expect(result.value).toBeCloseTo(0, 13);
    expect(result.standardized).toBeCloseTo(0, 13);
    expect(result.classification).toBe('A');
    expect(result.favours).toBe('neither');
  });

  it('negates when the groups are exchanged', () => {
    const tables = [
      [
        [2, 4, 6, 8],
        [8, 6, 4, 2],
      ],
      [
        [1, 3, 7, 9],
        [9, 7, 3, 1],
      ],
    ] as const;
    const forward = standardizedMeanDifference(graded(tables, 3));
    const reversed = standardizedMeanDifference(
      graded(
        tables.map(([reference, focal]) => [focal, reference] as const),
        3,
      ),
    );
    expect(reversed.value).toBeCloseTo(-forward.value, 12);
    expect(reversed.scoreDeviation).toBeCloseTo(forward.scoreDeviation, 12);
    expect(reversed.favours).toBe('focal');
    expect(forward.favours).toBe('reference');
  });

  it('reports the raw gap alongside the standardized one', () => {
    const result = standardizedMeanDifference(
      graded(
        [
          [
            [0, 0, 20],
            [20, 0, 0],
          ],
          [
            [0, 10, 10],
            [10, 10, 0],
          ],
        ],
        2,
      ),
    );
    // Every reference candidate outscores every focal candidate by a clear
    // margin, so the gap is large in both raw and standardized terms.
    expect(result.value).toBeLessThan(-1);
    expect(result.scoreDeviation).toBeGreaterThan(0);
    expect(result.standardized).toBeCloseTo(result.value / result.scoreDeviation, 13);
    expect(result.classification).toBe('C');
  });

  it('is unchanged by an affine change of the score metric, once standardized', () => {
    const strata = graded(
      [
        [
          [3, 5, 7],
          [7, 5, 3],
        ],
        [
          [4, 4, 8],
          [8, 4, 4],
        ],
      ],
      2,
    );
    const plain = standardizedMeanDifference(strata);
    const stretched = standardizedMeanDifference(strata, { categoryScores: [4, 7, 10] });
    expect(stretched.value).toBeCloseTo(3 * plain.value, 12);
    expect(stretched.standardized).toBeCloseTo(plain.standardized, 12);
  });

  it('reports a zero standardized difference when the item has no spread', () => {
    // Everyone scored the same, so there is nothing to divide by. The stratum is
    // uninformative for that reason, which is why a second one is needed.
    const strata = graded(
      [
        [
          [5, 0, 0],
          [5, 0, 0],
        ],
        [
          [2, 3, 0],
          [3, 2, 0],
        ],
      ],
      2,
    );
    const result = standardizedMeanDifference(strata);
    expect(result.scoreDeviation).toBeGreaterThan(0);
    expect(Number.isFinite(result.standardized)).toBe(true);
  });

  it('refuses a sample with nothing to compare', () => {
    expect(() => standardizedMeanDifference(graded([[[1, 2, 3], [0, 0, 0]]], 2))).toThrow(
      /no stratum holds both groups/,
    );
  });
});

describe('the polytomous classification', () => {
  it('calls an effect negligible when the null survives or the effect is small', () => {
    expect(classifyGraded(0.2, 0.9)).toBe('A');
    expect(classifyGraded(1e-9, NEGLIGIBLE_STANDARDIZED)).toBe('A');
    expect(classifyGraded(1e-9, -0.12)).toBe('A');
  });

  it('calls an effect large only when it is both significant and big', () => {
    expect(classifyGraded(1e-9, LARGE_STANDARDIZED)).toBe('C');
    expect(classifyGraded(1e-9, -0.4)).toBe('C');
    expect(classifyGraded(0.06, 0.4)).toBe('A');
  });

  it('puts the band between the boundaries at moderate', () => {
    expect(classifyGraded(0.01, 0.2)).toBe('B');
    expect(classifyGraded(0.01, -0.2)).toBe('B');
  });
});
