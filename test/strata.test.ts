import { describe, expect, it } from 'vitest';
import { MISSING, ResponseMatrix } from '../src/calibration/matrix.js';
import {
  informativeStrata,
  isInformative,
  matchedSample,
  mergeThinStrata,
  stratify,
  summariseStrata,
  type DifObservation,
  type Group,
} from '../src/dif/strata.js';

/** Expand per-stratum counts into the observations that would produce them. */
export function expand(
  tables: readonly (readonly [number, number, number, number])[],
  firstMatch = 1,
): DifObservation[] {
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

describe('stratify', () => {
  it('collects counts into one stratum per observed matching score', () => {
    const strata = stratify(expand([[12, 8, 6, 14], [25, 10, 18, 17]]));
    expect(strata).toHaveLength(2);
    const [first, second] = strata;
    expect(first?.matchLow).toBe(1);
    expect(first?.matchHigh).toBe(1);
    expect(first?.reference).toEqual([8, 12]);
    expect(first?.focal).toEqual([14, 6]);
    expect(first?.referenceTotal).toBe(20);
    expect(first?.focalTotal).toBe(20);
    expect(first?.total).toBe(40);
    expect(second?.reference).toEqual([10, 25]);
  });

  it('orders strata ascending however the observations arrive', () => {
    const shuffled = [
      { group: 'focal' as const, match: 5, score: 1 },
      { group: 'reference' as const, match: 0, score: 0 },
      { group: 'reference' as const, match: 3, score: 1 },
      { group: 'focal' as const, match: 0, score: 1 },
    ];
    expect(stratify(shuffled).map((s) => s.matchLow)).toEqual([0, 3, 5]);
  });

  it('leaves out levels nobody reached', () => {
    const strata = stratify([
      { group: 'reference', match: 0, score: 1 },
      { group: 'focal', match: 7, score: 0 },
    ]);
    expect(strata.map((s) => s.matchLow)).toEqual([0, 7]);
  });

  it('widens the category axis for a rubric-scored item', () => {
    const strata = stratify(
      [
        { group: 'reference', match: 4, score: 3 },
        { group: 'reference', match: 4, score: 0 },
        { group: 'focal', match: 4, score: 2 },
      ],
      { maxScore: 3 },
    );
    expect(strata[0]?.reference).toEqual([1, 0, 0, 1]);
    expect(strata[0]?.focal).toEqual([0, 0, 1, 0]);
  });

  it('rejects a score outside the category range', () => {
    expect(() => stratify([{ group: 'focal', match: 1, score: 2 }])).toThrow(/outside 0\.\.1/);
    expect(() =>
      stratify([{ group: 'focal', match: 1, score: -1 }], { maxScore: 3 }),
    ).toThrow(/outside 0\.\.3/);
    expect(() =>
      stratify([{ group: 'focal', match: 1, score: 1.5 }], { maxScore: 3 }),
    ).toThrow(/outside 0\.\.3/);
  });

  it('rejects an empty sample, a bad maximum or a non-finite match', () => {
    expect(() => stratify([])).toThrow(/no observations/);
    expect(() => stratify([{ group: 'focal', match: 1, score: 0 }], { maxScore: 0 })).toThrow(
      /maxScore/,
    );
    expect(() =>
      stratify([{ group: 'focal', match: 1, score: 0 }], { minimumStratum: 0 }),
    ).toThrow(/minimumStratum/);
    expect(() => stratify([{ group: 'focal', match: Number.NaN, score: 0 }])).toThrow(
      /non-finite matching score/,
    );
  });
});

describe('merging thin strata', () => {
  it('carries a shortfall upward until the floor is met', () => {
    const strata = stratify(expand([[1, 0, 0, 1], [3, 3, 1, 1], [10, 10, 10, 10]]));
    const merged = mergeThinStrata(strata, 8);
    expect(merged).toHaveLength(2);
    expect(merged[0]?.matchLow).toBe(1);
    expect(merged[0]?.matchHigh).toBe(2);
    expect(merged[0]?.total).toBe(10);
    expect(merged[1]?.total).toBe(40);
  });

  it('folds a leftover top stratum back down rather than keeping it', () => {
    const strata = stratify(expand([[10, 10, 10, 10], [1, 0, 0, 0]]));
    const merged = mergeThinStrata(strata, 5);
    expect(merged).toHaveLength(1);
    expect(merged[0]?.total).toBe(41);
    expect(merged[0]?.matchLow).toBe(1);
    expect(merged[0]?.matchHigh).toBe(2);
  });

  it('conserves every person and every category count', () => {
    const tables = [[3, 1, 2, 2], [1, 1, 0, 1], [8, 4, 7, 5], [2, 0, 1, 1]] as const;
    const strata = stratify(expand(tables.map((t) => [...t] as [number, number, number, number])));
    for (const minimum of [1, 5, 12, 40, 500]) {
      const merged = mergeThinStrata(strata, minimum);
      const before = summariseStrata(strata);
      const after = summariseStrata(merged);
      expect(after.reference).toBe(before.reference);
      expect(after.focal).toBe(before.focal);
    }
  });

  it('leaves a stratification alone when the floor is already met', () => {
    const strata = stratify(expand([[10, 10, 10, 10], [10, 10, 10, 10]]));
    expect(mergeThinStrata(strata, 3)).toEqual(strata);
    expect(stratify(expand([[10, 10, 10, 10]]), { minimumStratum: 3 })).toEqual(
      stratify(expand([[10, 10, 10, 10]])),
    );
  });

  it('returns nothing for nothing', () => {
    expect(mergeThinStrata([], 10)).toEqual([]);
  });
});

describe('informativeness', () => {
  it('rejects a stratum missing either group', () => {
    const [onlyReference] = stratify(expand([[5, 5, 0, 0]]));
    const [onlyFocal] = stratify(expand([[0, 0, 5, 5]]));
    expect(isInformative(onlyReference as never)).toBe(false);
    expect(isInformative(onlyFocal as never)).toBe(false);
  });

  it('rejects a stratum where everyone landed in the same category', () => {
    const [allRight] = stratify(expand([[7, 0, 9, 0]]));
    const [allWrong] = stratify(expand([[0, 7, 0, 9]]));
    expect(isInformative(allRight as never)).toBe(false);
    expect(isInformative(allWrong as never)).toBe(false);
  });

  it('accepts a stratum with both groups and a mix of outcomes', () => {
    const [mixed] = stratify(expand([[7, 1, 2, 9]]));
    expect(isInformative(mixed as never)).toBe(true);
  });

  it('counts the discarded people in the summary', () => {
    const strata = stratify(expand([[5, 0, 4, 0], [6, 4, 3, 7], [2, 2, 0, 0]]));
    const summary = summariseStrata(strata);
    expect(summary.strata).toBe(3);
    expect(summary.informative).toBe(1);
    expect(summary.reference).toBe(19);
    expect(summary.focal).toBe(14);
    expect(summary.discarded).toBe(9 + 4);
    expect(informativeStrata(strata)).toHaveLength(1);
  });
});

describe('matchedSample', () => {
  const groups: Group[] = ['reference', 'reference', 'focal', 'focal'];
  const rows = [
    [1, 1, 1, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0],
  ];

  it('scores the matching criterion over every item by default', () => {
    const matrix = new ResponseMatrix({ rows });
    const { observations, dropped, maxMatch } = matchedSample(matrix, groups, 0);
    expect(dropped).toBe(0);
    expect(maxMatch).toBe(4);
    expect(observations).toEqual([
      { group: 'reference', match: 3, score: 1 },
      { group: 'reference', match: 1, score: 1 },
      { group: 'focal', match: 3, score: 1 },
      { group: 'focal', match: 1, score: 0 },
    ]);
  });

  it('drops the studied item from its own criterion when asked', () => {
    const matrix = new ResponseMatrix({ rows });
    const { observations, maxMatch } = matchedSample(matrix, groups, 0, {
      includeStudied: false,
    });
    expect(maxMatch).toBe(3);
    expect(observations.map((o) => o.match)).toEqual([2, 0, 2, 1]);
  });

  it('restricts the criterion to an anchor', () => {
    const matrix = new ResponseMatrix({ rows });
    const { observations, maxMatch } = matchedSample(matrix, groups, 0, { anchor: [1, 2] });
    expect(maxMatch).toBe(2);
    expect(observations.map((o) => o.match)).toEqual([2, 0, 1, 1]);
  });

  it('drops a candidate missing the studied item or any anchor item', () => {
    const matrix = new ResponseMatrix({
      rows: [
        [1, 1, 1, 0],
        [MISSING, 1, 1, 1],
        [1, MISSING, 1, 1],
        [0, 0, 1, 0],
      ],
    });
    const { observations, dropped } = matchedSample(matrix, groups, 0);
    expect(dropped).toBe(2);
    expect(observations).toHaveLength(2);
  });

  it('keeps a candidate missing an item outside the anchor', () => {
    const matrix = new ResponseMatrix({
      rows: [
        [1, 1, 1, MISSING],
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
      ],
    });
    const { observations, dropped } = matchedSample(matrix, groups, 0, { anchor: [0, 1, 2] });
    expect(dropped).toBe(0);
    expect(observations[0]?.match).toBe(3);
  });

  it('rejects a group vector of the wrong length or an item out of range', () => {
    const matrix = new ResponseMatrix({ rows });
    expect(() => matchedSample(matrix, ['focal'], 0)).toThrow(/group labels for 4 people/);
    expect(() => matchedSample(matrix, groups, 9)).toThrow(/no item at index 9/);
    expect(() => matchedSample(matrix, groups, 0, { anchor: [9] })).toThrow(/no item at index 9/);
  });

  it('rejects a criterion that excluding the studied item would empty', () => {
    const matrix = new ResponseMatrix({ rows });
    expect(() =>
      matchedSample(matrix, groups, 0, { anchor: [0], includeStudied: false }),
    ).toThrow(/matching criterion is empty/);
  });

  it('rejects a sample where nobody survived the completeness rule', () => {
    const matrix = new ResponseMatrix({
      rows: [
        [MISSING, 1, 1, 0],
        [MISSING, 0, 0, 0],
        [MISSING, 1, 0, 1],
        [MISSING, 0, 1, 0],
      ],
    });
    expect(() => matchedSample(matrix, groups, 0)).toThrow(/every candidate was dropped/);
  });
});
