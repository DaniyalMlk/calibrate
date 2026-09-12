import { describe, expect, it } from 'vitest';
import { areaScan, integrateArea, rajuArea, rankByArea } from '../src/dif/area.js';
import { fourPL, makeItem, threePL, twoPL, type Item } from '../src/models/item.js';

const item = (id: string, a: number, b: number, c = 0, d = 1): Item =>
  makeItem(id, fourPL(a, b, c, d));

describe('rajuArea', () => {
  it('agrees with numerical integration of the curves it describes', () => {
    const cases: readonly (readonly [number, number, number, number])[] = [
      [1, -0.5, 1, 0.5],
      [1.2, 0, 0.8, 0.6],
      [0.7, -1, 1.9, 0.4],
      [2, -1.2, 0.6, 1.1],
      [1, 0, 0.35, 0],
      [0.5, 1.5, 2.5, -0.8],
    ];
    for (const [aReference, bReference, aFocal, bFocal] of cases) {
      const reference = item('x', aReference, bReference);
      const focal = item('x', aFocal, bFocal);
      const closed = rajuArea(reference, focal);
      const numeric = integrateArea(reference, focal);
      expect(closed.signed).toBeCloseTo(numeric.signed, 9);
      expect(closed.unsigned).toBeCloseTo(numeric.unsigned, 9);
    }
  });

  it('reduces the signed area to the difference in difficulty', () => {
    // Independent of both discriminations: only the locations matter.
    for (const [aReference, aFocal] of [
      [1, 1],
      [0.4, 2.2],
      [3, 0.6],
    ]) {
      const area = rajuArea(item('x', aReference as number, -0.3), item('x', aFocal as number, 0.9));
      expect(area.signed).toBeCloseTo(-1.2, 12);
    }
  });

  it('makes the unsigned area the plain shift when the curves are parallel', () => {
    const area = rajuArea(item('x', 1.4, -0.25), item('x', 1.4, 0.75));
    expect(area.unsigned).toBeCloseTo(1, 12);
    expect(area.signed).toBeCloseTo(-1, 12);
    expect(area.nonUniformShare).toBeCloseTo(0, 12);
    expect(area.crossing).toBeNull();
  });

  it('cancels the signed area on a crossing item and keeps the unsigned one', () => {
    // Same location, different slope: the curves cross exactly at the common
    // difficulty, the two advantages are mirror images, and the signed area is
    // zero while the departure is large. This is the pair that matters.
    const area = rajuArea(item('x', 1, 0), item('x', 0.35, 0));
    expect(area.signed).toBeCloseTo(0, 12);
    expect(area.unsigned).toBeCloseTo(2.574547, 5);
    expect(area.nonUniformShare).toBeCloseTo(1, 12);
    expect(area.crossing).toBeCloseTo(0, 12);
    expect(area.favours).toBe('neither');
  });

  it('locates the crossing point where the two curves actually meet', () => {
    const reference = item('x', 1.6, -0.4);
    const focal = item('x', 0.5, 0.8);
    const area = rajuArea(reference, focal);
    const theta = area.crossing as number;
    expect(theta).not.toBeNull();
    const at = (i: Item, t: number): number =>
      1 / (1 + Math.exp(-i.parameters.a * (t - i.parameters.b)));
    expect(at(reference, theta)).toBeCloseTo(at(focal, theta), 12);
  });

  it('keeps the unsigned area at least as large as the signed one', () => {
    for (let i = 0; i < 60; i += 1) {
      const aReference = 0.3 + (i % 7) * 0.4;
      const aFocal = 0.3 + ((i * 3) % 9) * 0.35;
      const area = rajuArea(item('x', aReference, -1 + i * 0.03), item('x', aFocal, 1 - i * 0.02));
      expect(area.unsigned).toBeGreaterThanOrEqual(Math.abs(area.signed) - 1e-12);
      expect(area.nonUniformShare).toBeGreaterThanOrEqual(0);
      expect(area.nonUniformShare).toBeLessThanOrEqual(1);
    }
  });

  it('stays stable as the discriminations converge on each other', () => {
    // The closed form divides by the difference of the discriminations and
    // exponentiates the reciprocal, so this is where it would break. It has to
    // approach the parallel-curve answer rather than overflowing to infinity.
    for (const epsilon of [1e-2, 1e-4, 1e-8, 1e-12, 1e-15]) {
      const area = rajuArea(item('x', 1, 0), item('x', 1 + epsilon, 1.5));
      expect(Number.isFinite(area.unsigned)).toBe(true);
      expect(area.unsigned).toBeCloseTo(1.5, 4);
      expect(area.signed).toBeCloseTo(-1.5, 12);
    }
  });

  it('negates the signed area and preserves the unsigned one when the roles swap', () => {
    const reference = item('x', 1.3, -0.2);
    const focal = item('x', 0.7, 0.9);
    const forward = rajuArea(reference, focal);
    const reversed = rajuArea(focal, reference);
    expect(reversed.signed).toBeCloseTo(-forward.signed, 12);
    expect(reversed.unsigned).toBeCloseTo(forward.unsigned, 12);
    expect(reversed.crossing).toBeCloseTo(forward.crossing as number, 10);
    expect(forward.favours).toBe('reference');
    expect(reversed.favours).toBe('focal');
  });

  it('is zero for an item calibrated identically twice', () => {
    const area = rajuArea(item('x', 1.1, 0.4), item('x', 1.1, 0.4));
    expect(area.signed).toBe(0);
    expect(area.unsigned).toBe(0);
    expect(area.nonUniformShare).toBe(0);
    expect(area.favours).toBe('neither');
  });

  it('scales both areas by the span between the asymptotes', () => {
    // A guessing floor compresses the whole curve towards the top of the range,
    // and the area between two compressed curves shrinks in proportion.
    const plain = rajuArea(item('x', 1.2, -0.3), item('x', 0.8, 0.6));
    const guessing = rajuArea(
      makeItem('x', threePL(1.2, -0.3, 0.2)),
      makeItem('x', threePL(0.8, 0.6, 0.2)),
    );
    expect(guessing.signed).toBeCloseTo(0.8 * plain.signed, 12);
    expect(guessing.unsigned).toBeCloseTo(0.8 * plain.unsigned, 12);
    expect(guessing.nonUniformShare).toBeCloseTo(plain.nonUniformShare, 12);

    const slipping = rajuArea(item('x', 1.2, -0.3, 0.15, 0.95), item('x', 0.8, 0.6, 0.15, 0.95));
    expect(slipping.unsigned).toBeCloseTo(0.8 * plain.unsigned, 12);
  });

  it('refuses calibrations that never converge at the tails', () => {
    expect(() =>
      rajuArea(makeItem('x', threePL(1, 0, 0.2)), makeItem('x', threePL(1, 0.5, 0.25))),
    ).toThrow(/different asymptotes/);
    expect(() => rajuArea(item('x', 1, 0, 0, 1), item('x', 1, 0.5, 0, 0.95))).toThrow(
      /unbounded/,
    );
  });

  it('refuses calibrations on different metrics', () => {
    expect(() =>
      rajuArea(makeItem('x', twoPL(1, 0, 'logistic')), makeItem('x', twoPL(1, 0.5, 'normal'))),
    ).toThrow(/different metrics/);
  });
});

describe('integrateArea', () => {
  it('rejects a window or a subdivision it cannot integrate over', () => {
    const reference = item('x', 1, 0);
    const focal = item('x', 1, 0.5);
    expect(() => integrateArea(reference, focal, { limit: 0 })).toThrow(/limit must be positive/);
    expect(() => integrateArea(reference, focal, { intervals: 3 })).toThrow(/even integer/);
    expect(() => integrateArea(reference, focal, { intervals: 0 })).toThrow(/at least 2/);
  });

  it('converges on the closed form as the grid is refined', () => {
    const reference = item('x', 1.5, -0.6);
    const focal = item('x', 0.6, 0.7);
    const exact = rajuArea(reference, focal).unsigned;
    const coarse = Math.abs(integrateArea(reference, focal, { intervals: 200 }).unsigned - exact);
    const fine = Math.abs(integrateArea(reference, focal, { intervals: 20000 }).unsigned - exact);
    expect(fine).toBeLessThan(coarse);
    expect(fine).toBeLessThan(1e-9);
  });
});

describe('areaScan', () => {
  const referenceBank = [
    item('a', 1.1, -0.8),
    item('b', 0.9, 0),
    item('c', 1.4, 0.7),
    item('d', 1, 1.2),
  ];

  it('matches items by identifier, not by position', () => {
    const focalBank = [
      item('c', 1.4, 0.7),
      item('a', 1.1, 0.3), // a full logit harder for the focal group
      item('b', 0.9, 0),
    ];
    const rows = areaScan(referenceBank, focalBank);
    expect(rows.map((row) => row.itemId)).toEqual(['a', 'b', 'c']);
    expect(rows[0]?.area.signed).toBeCloseTo(-1.1, 12);
    expect(rows[1]?.area.unsigned).toBeCloseTo(0, 12);
  });

  it('leaves out an item only one calibration has', () => {
    const rows = areaScan(referenceBank, [item('a', 1.1, -0.8), item('z', 1, 0)]);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.itemId).toBe('a');
  });

  it('refuses two calibrations with nothing in common', () => {
    expect(() => areaScan(referenceBank, [item('z', 1, 0)])).toThrow(/share no item identifiers/);
  });

  it('ranks by total departure, so a crossing item is not hidden', () => {
    const focalBank = [
      item('a', 1.1, -0.5), // a modest uniform shift
      item('b', 0.25, 0), // curves cross: no uniform component at all
      item('c', 1.4, 0.7),
      item('d', 1, 1.2),
    ];
    const ranked = rankByArea(areaScan(referenceBank, focalBank));
    expect(ranked[0]?.itemId).toBe('b');
    expect(ranked[0]?.area.signed).toBeCloseTo(0, 12);
    expect(ranked[0]?.area.nonUniformShare).toBeCloseTo(1, 10);
    expect(ranked[1]?.itemId).toBe('a');
    // Ranking on the signed area alone would have put the crossing item last.
    expect(Math.abs(ranked[0]?.area.signed as number)).toBeLessThan(
      Math.abs(ranked[1]?.area.signed as number),
    );
  });
});
