import { describe, expect, it } from 'vitest';
import {
  classifyDelta,
  DELTA_SCALE,
  mantelHaenszel,
  standardizedDifference,
} from '../src/dif/mantel.js';
import { stratify, type DifObservation, type Group } from '../src/dif/strata.js';

type Table = readonly [number, number, number, number];

function expand(tables: readonly Table[], firstMatch = 1): DifObservation[] {
  const observations: DifObservation[] = [];
  for (const [index, [refRight, refWrong, focRight, focWrong]] of tables.entries()) {
    const match = firstMatch + index;
    const push = (group: Group, score: number, times: number): void => {
      for (let i = 0; i < times; i += 1) observations.push({ group, match, score });
    };
    push('reference', 1, refRight);
    push('reference', 0, refWrong);
    push('focal', 1, focRight);
    push('focal', 0, focWrong);
  }
  return observations;
}

const strataOf = (tables: readonly Table[]) => stratify(expand(tables));
const swap = (tables: readonly Table[]): Table[] =>
  tables.map(([a, b, c, d]) => [c, d, a, b] as Table);

/**
 * The reference stratification. Its pooled odds ratio, Robins-Breslow-Greenland
 * standard error and continuity-corrected chi-square were computed
 * independently, outside this codebase, from the same counts.
 */
const REFERENCE: Table[] = [
  [12, 8, 6, 14],
  [25, 10, 18, 17],
  [30, 5, 22, 13],
  [9, 11, 4, 16],
];

describe('mantelHaenszel', () => {
  it('matches an independent implementation on the reference tables', () => {
    const result = mantelHaenszel(strataOf(REFERENCE));
    expect(result.oddsRatio).toBeCloseTo(3.0177383592017737, 12);
    expect(result.logOddsRatio).toBeCloseTo(1.1045076631487876, 12);
    expect(result.logOddsStandardError).toBeCloseTo(0.30173725307602567, 12);
    expect(result.chiSquare).toBeCloseTo(12.614756311597139, 10);
    expect(result.pValue).toBeCloseTo(0.00038271342840090483, 12);
  });

  it('puts the log odds ratio onto the delta metric', () => {
    const result = mantelHaenszel(strataOf(REFERENCE));
    expect(result.delta).toBeCloseTo(-DELTA_SCALE * result.logOddsRatio, 14);
    expect(result.delta).toBeCloseTo(-2.595593008399651, 10);
    expect(result.deltaStandardError).toBeCloseTo(0.7090825447286604, 10);
    expect(result.favours).toBe('reference');
  });

  it('reduces to the ordinary odds ratio on a single stratum', () => {
    // One table, so the pooled estimate has nothing to pool: AD/BC exactly.
    const result = mantelHaenszel(strataOf([[20, 10, 15, 30]]));
    expect(result.oddsRatio).toBeCloseTo((20 * 30) / (10 * 15), 13);
    expect(result.oddsRatio).toBeCloseTo(4, 13);
  });

  it('is exactly one when neither group has an advantage anywhere', () => {
    const balanced: Table[] = [
      [10, 10, 10, 10],
      [30, 10, 15, 5],
      [8, 24, 4, 12],
    ];
    const result = mantelHaenszel(strataOf(balanced));
    expect(result.oddsRatio).toBeCloseTo(1, 13);
    expect(result.delta).toBeCloseTo(0, 13);
    expect(result.chiSquare).toBeCloseTo(0, 13);
    expect(result.pValue).toBeCloseTo(1, 13);
    expect(result.classification).toBe('A');
    expect(result.favours).toBe('neither');
  });

  it('recovers a common odds ratio the strata all share exactly', () => {
    // The estimator is a weighted average of the per-stratum odds ratios, so
    // when every stratum carries the same one the weights cannot matter and the
    // answer is that ratio to machine precision, whatever the stratum sizes.
    const sameRatio: Table[] = [
      [20, 10, 10, 10], // (20 * 10) / (10 * 10) = 2
      [30, 30, 10, 20], // (30 * 20) / (30 * 10) = 2
      [8, 4, 12, 12], //   (8 * 12) / (4 * 12)  = 2
      [40, 20, 10, 10], // (40 * 10) / (20 * 10) = 2
    ];
    expect(mantelHaenszel(strataOf(sameRatio)).oddsRatio).toBeCloseTo(2, 12);
    // And the same with the stratum sizes wildly unequal.
    expect(mantelHaenszel(strataOf([sameRatio[0] as Table, sameRatio[2] as Table])).oddsRatio)
      .toBeCloseTo(2, 12);
  });

  it('inverts exactly when the two groups are exchanged', () => {
    const forward = mantelHaenszel(strataOf(REFERENCE));
    const reversed = mantelHaenszel(strataOf(swap(REFERENCE)));
    expect(reversed.oddsRatio).toBeCloseTo(1 / forward.oddsRatio, 12);
    expect(reversed.logOddsRatio).toBeCloseTo(-forward.logOddsRatio, 12);
    expect(reversed.delta).toBeCloseTo(-forward.delta, 12);
    // The chi-square is a two-sided statement and does not care which group is
    // which; the standard error likewise.
    expect(reversed.chiSquare).toBeCloseTo(forward.chiSquare, 10);
    expect(reversed.logOddsStandardError).toBeCloseTo(forward.logOddsStandardError, 12);
    expect(reversed.favours).toBe('focal');
  });

  it('is unchanged by strata that carry no comparison', () => {
    const padded: Table[] = [
      [0, 0, 9, 3], // reference absent
      ...REFERENCE,
      [7, 0, 5, 0], // everyone right
      [4, 4, 0, 0], // focal absent
    ];
    const padding = mantelHaenszel(stratify(expand(padded, 0)));
    const plain = mantelHaenszel(strataOf(REFERENCE));
    expect(padding.oddsRatio).toBeCloseTo(plain.oddsRatio, 13);
    expect(padding.chiSquare).toBeCloseTo(plain.chiSquare, 10);
    expect(padding.strata.strata).toBe(7);
    expect(padding.strata.informative).toBe(4);
    expect(padding.strata.discarded).toBe(12 + 12 + 8);
  });

  it('reports an unbounded odds ratio rather than dividing by zero', () => {
    // Every reference candidate right, every focal candidate wrong: there is no
    // matched pair pointing the other way, so the odds ratio has no upper bound.
    const separated: Table[] = [
      [15, 0, 0, 15],
      [20, 0, 0, 20],
    ];
    const result = mantelHaenszel(strataOf(separated));
    expect(result.estimable).toBe(false);
    expect(result.oddsRatio).toBe(Number.POSITIVE_INFINITY);
    expect(result.delta).toBe(Number.NEGATIVE_INFINITY);
    expect(result.logOddsStandardError).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isFinite(result.chiSquare)).toBe(true);
    expect(result.pValue).toBeLessThan(0.001);
    expect(result.classification).toBe('C');
    expect(result.favours).toBe('reference');
  });

  it('reports a zero odds ratio in the mirrored degenerate case', () => {
    const result = mantelHaenszel(strataOf([[0, 15, 15, 0], [0, 20, 20, 0]]));
    expect(result.estimable).toBe(false);
    expect(result.oddsRatio).toBe(0);
    expect(result.delta).toBe(Number.POSITIVE_INFINITY);
    expect(result.favours).toBe('focal');
  });

  it('refuses a sample with nothing to compare', () => {
    expect(() => mantelHaenszel(strataOf([[5, 5, 0, 0]]))).toThrow(/no stratum holds both groups/);
    expect(() => mantelHaenszel(strataOf([[6, 0, 4, 0]]))).toThrow(/no comparison is possible/);
  });

  it('refuses a rubric-scored item and says what to use instead', () => {
    const graded = stratify(
      [
        { group: 'reference', match: 2, score: 2 },
        { group: 'focal', match: 2, score: 0 },
      ],
      { maxScore: 2 },
    );
    expect(() => mantelHaenszel(graded)).toThrow(/use generalizedMantel/);
  });

  it('floors the continuity correction rather than letting it overshoot', () => {
    // A deviation under half a count would square back into a positive
    // statistic if the correction were subtracted without a floor.
    const tiny: Table[] = [[10, 10, 10, 10], [11, 9, 10, 10]];
    const result = mantelHaenszel(strataOf(tiny));
    expect(result.chiSquare).toBe(0);
    expect(result.pValue).toBe(1);
  });
});

describe('the ETS classification', () => {
  it('calls an effect negligible when the null survives', () => {
    expect(classifyDelta(3, 0.2, 0.09)).toBe('A');
    expect(classifyDelta(-3, 0.2, 0.51)).toBe('A');
  });

  it('calls an effect negligible when it is under a delta point', () => {
    expect(classifyDelta(0.9, 0.05, 1e-9)).toBe('A');
    expect(classifyDelta(-0.99, 0.01, 1e-30)).toBe('A');
  });

  it('calls an effect large only when it clears 1.5 and rules out 1', () => {
    expect(classifyDelta(2, 0.2, 1e-6)).toBe('C');
    expect(classifyDelta(-2, 0.2, 1e-6)).toBe('C');
    // Clears 1.5, but the interval still reaches down past 1.
    expect(classifyDelta(1.6, 0.5, 1e-6)).toBe('B');
    // Rules out 1 comfortably, but does not clear 1.5.
    expect(classifyDelta(1.4, 0.01, 1e-6)).toBe('B');
  });

  it('falls back to the chi-square when the effect size is unbounded', () => {
    expect(classifyDelta(Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, 1e-8, false)).toBe(
      'C',
    );
    expect(classifyDelta(Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, 0.4, false)).toBe('A');
  });

  it('flags the reference tables as a large effect against the focal group', () => {
    const result = mantelHaenszel(strataOf(REFERENCE));
    expect(result.classification).toBe('C');
    expect(result.favours).toBe('reference');
  });
});

describe('standardizedDifference', () => {
  it('matches a hand computation on the reference tables', () => {
    const result = standardizedDifference(strataOf(REFERENCE));
    expect(result.focalProportion).toBeCloseTo(0.45454545454545453, 12);
    expect(result.referenceProportion).toBeCloseTo(0.6909090909090909, 12);
    expect(result.value).toBeCloseTo(-0.2363636363636364, 12);
    expect(result.classification).toBe('C');
    expect(result.favours).toBe('reference');
  });

  it('is zero when the groups match stratum by stratum', () => {
    const result = standardizedDifference(strataOf([[10, 10, 5, 5], [30, 10, 9, 3]]));
    expect(result.value).toBeCloseTo(0, 14);
    expect(result.classification).toBe('A');
    expect(result.favours).toBe('neither');
  });

  it('negates when the groups are exchanged', () => {
    const forward = standardizedDifference(strataOf(REFERENCE));
    const reversed = standardizedDifference(strataOf(swap(REFERENCE)));
    expect(reversed.value).toBeCloseTo(-forward.value, 12);
    expect(reversed.favours).toBe('focal');
  });

  it('weights by the focal distribution, not the reference one', () => {
    // The item favours the reference group only in the lower stratum. Moving
    // the focal group's mass into that stratum must make the statistic more
    // negative, which it cannot do if the weights come from anywhere else.
    const focalLow: Table[] = [[18, 2, 40, 60], [10, 10, 5, 5]];
    const focalHigh: Table[] = [[18, 2, 4, 6], [10, 10, 50, 50]];
    expect(standardizedDifference(strataOf(focalLow)).value).toBeLessThan(
      standardizedDifference(strataOf(focalHigh)).value,
    );
  });

  it('stays inside the unit interval and honours the effect boundaries', () => {
    const extreme = standardizedDifference(strataOf([[20, 0, 0, 20], [20, 0, 0, 20]]));
    expect(extreme.value).toBeCloseTo(-1, 13);
    expect(extreme.classification).toBe('C');
    expect(standardizedDifference(strataOf([[23, 17, 20, 20]])).value).toBeCloseTo(-0.075, 13);
    expect(standardizedDifference(strataOf([[23, 17, 20, 20]])).classification).toBe('B');
    expect(standardizedDifference(strataOf([[102, 98, 99, 101]])).classification).toBe('A');
  });

  it('treats each effect boundary as belonging to the band above it', () => {
    // Exactly 0.05 is moderate, exactly 0.10 is large. Worth pinning: an item
    // landing on a boundary is not rare, and a rule that reported it either way
    // depending on floating-point noise would be worse than either choice.
    const atFive = standardizedDifference(strataOf([[11, 9, 10, 10]]));
    expect(atFive.value).toBeCloseTo(-0.05, 14);
    expect(atFive.classification).toBe('B');

    const atTen = standardizedDifference(strataOf([[11, 9, 9, 11]]));
    expect(atTen.value).toBeCloseTo(-0.1, 14);
    expect(atTen.classification).toBe('C');
  });

  it('refuses a sample with nothing to compare and a graded item', () => {
    expect(() => standardizedDifference(strataOf([[5, 5, 0, 0]]))).toThrow(
      /no stratum holds both groups/,
    );
    const graded = stratify([{ group: 'focal', match: 1, score: 2 }], { maxScore: 2 });
    expect(() => standardizedDifference(graded)).toThrow(/use generalizedMantel/);
  });
});
