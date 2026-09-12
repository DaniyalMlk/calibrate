import { describe, expect, it } from 'vitest';
import { MISSING, ResponseMatrix, type Cell } from '../src/calibration/matrix.js';
import { flaggedAt, purifiedScan, rankByEffect, scanBank } from '../src/dif/scan.js';
import type { Group } from '../src/dif/strata.js';
import { makeItem, twoPL, type Item } from '../src/models/item.js';
import { simulateDif } from '../src/simulation/dif.js';

/** A small, well-spread 2PL bank: easy items at the bottom, hard at the top. */
function ladder(size = 20): Item[] {
  return Array.from({ length: size }, (_, index) =>
    makeItem(`i-${index}`, twoPL(1 + 0.03 * index, -1.8 + (3.6 * index) / (size - 1))),
  );
}

const BANK = ladder();

describe('scanBank', () => {
  it('flags nothing when the groups differ in neither ability nor item behaviour', () => {
    const sim = simulateDif({ bank: BANK, seed: 11 });
    const scan = scanBank(sim.matrix, sim.groups);
    expect(scan.analysed).toBe(20);
    expect(scan.rows).toHaveLength(20);
    expect(flaggedAt(scan, 'C')).toEqual([]);
    expect(scan.flagged).toEqual([]);
  });

  it('flags nothing when the groups differ in ability alone', () => {
    // The whole purpose of matching. The focal group is eight tenths of a logit
    // less able, so it answers fewer items correctly on every item in the form —
    // and none of that is the items' fault. An unmatched comparison of
    // proportions would flag the entire bank.
    const sim = simulateDif({ bank: BANK, focalMean: -0.8, seed: 12 });
    const scan = scanBank(sim.matrix, sim.groups);
    expect(flaggedAt(scan, 'C')).toEqual([]);
    expect(scan.flagged).toEqual([]);
  });

  it('recovers a planted uniform effect at the top of the ranking', () => {
    const sim = simulateDif({
      bank: BANK,
      shifts: [{ item: 7, difficulty: 1 }],
      seed: 13,
    });
    const scan = scanBank(sim.matrix, sim.groups);
    expect(flaggedAt(scan, 'C')).toEqual([7]);

    const ranked = rankByEffect(scan);
    expect(ranked[0]?.item).toBe(7);
    expect(ranked[0]?.favours).toBe('reference');
    expect(ranked[0]?.mantel?.delta).toBeLessThan(-2);
    expect(ranked[0]?.mantel?.pValue).toBeLessThan(1e-20);
    // And the runner-up is an ordinary item, an order of magnitude behind.
    expect(Math.abs(ranked[1]?.mantel?.delta ?? 0)).toBeLessThan(1);
  });

  it('separates two planted effects pointing in opposite directions, through impact', () => {
    const sim = simulateDif({
      bank: BANK,
      focalMean: -0.7,
      shifts: [
        { item: 7, difficulty: 1 },
        { item: 14, difficulty: -0.9 },
      ],
      seed: 14,
    });
    const scan = scanBank(sim.matrix, sim.groups);
    expect(flaggedAt(scan, 'C')).toEqual([7, 14]);

    const rows = scan.rows;
    expect(rows[7]?.favours).toBe('reference');
    expect(rows[14]?.favours).toBe('focal');
    expect(rows[7]?.standardized?.value).toBeLessThan(-0.1);
    expect(rows[14]?.standardized?.value).toBeGreaterThan(0.1);
  });

  it('under-reports a non-uniform effect, which is the known weakness', () => {
    // The focal group's response function is much flatter on item 5, so the item
    // favours them below the crossing point and the reference group above it.
    // Pooling across strata cancels most of that, and the odds ratio comes back
    // far smaller than the true departure. Detected, but as moderate rather than
    // large — a result worth pinning, because it is the reason a model-based
    // method is needed alongside this one.
    const sim = simulateDif({
      bank: BANK,
      shifts: [{ item: 5, discrimination: 0.35 }],
      seed: 15,
    });
    const scan = scanBank(sim.matrix, sim.groups);
    expect(scan.flagged).toEqual([5]);
    expect(flaggedAt(scan, 'C')).toEqual([]);
    expect(scan.rows[5]?.classification).toBe('B');
    expect(Math.abs(scan.rows[5]?.mantel?.delta ?? 0)).toBeLessThan(1.5);
  });

  it('carries the standardized difference alongside every odds ratio', () => {
    const sim = simulateDif({ bank: BANK, shifts: [{ item: 7, difficulty: 1 }], seed: 13 });
    const scan = scanBank(sim.matrix, sim.groups);
    for (const row of scan.rows) {
      expect(row.mantel).not.toBeNull();
      expect(row.standardized).not.toBeNull();
      // The two effect sizes always agree about which group is favoured, even
      // where they disagree about how much.
      expect(row.standardized?.favours).toBe(row.mantel?.favours);
    }
  });

  it('records why an item could not be analysed instead of failing the scan', () => {
    const groups: Group[] = ['reference', 'reference', 'focal', 'focal'];
    const rows: Cell[][] = [
      [1, 1, 1, 0],
      [1, 0, 0, 1],
      [1, 1, 0, 1],
      [1, 0, 1, 0],
    ];
    const scan = scanBank(new ResponseMatrix({ rows }), groups);
    // Item 0 is answered correctly by everyone, so no stratum has a mix of
    // outcomes on it and there is nothing to compare.
    expect(scan.rows[0]?.classification).toBeNull();
    expect(scan.rows[0]?.note).toMatch(/no stratum holds both groups|no comparison is possible/);
    expect(scan.analysed).toBeLessThan(4);
    expect(scan.rows).toHaveLength(4);
  });

  it('counts candidates dropped for an incomplete matching score', () => {
    const groups: Group[] = ['reference', 'reference', 'focal', 'focal'];
    const rows: Cell[][] = [
      [1, 1, 0, 0],
      [MISSING, 0, 1, 1],
      [0, 1, 1, 0],
      [1, 0, 0, 1],
    ];
    const scan = scanBank(new ResponseMatrix({ rows }), groups);
    for (const row of scan.rows) expect(row.dropped).toBe(1);
  });

  it('restricts the criterion to an anchor, keeping the studied item in it', () => {
    const sim = simulateDif({ bank: BANK, shifts: [{ item: 7, difficulty: 1 }], seed: 13 });
    const anchor = [0, 1, 2, 3, 4, 5];
    const scan = scanBank(sim.matrix, sim.groups, { anchor });
    expect(scan.anchor).toEqual(anchor);
    expect(scan.analysed).toBe(20);
    // Item 7 is outside the anchor and still analysed, because it is added back
    // into its own matching score.
    expect(scan.rows[7]?.classification).not.toBeNull();
  });

  it('rejects an anchor naming an item that does not exist', () => {
    const sim = simulateDif({ bank: ladder(4), referenceCount: 40, focalCount: 40, seed: 3 });
    expect(() => scanBank(sim.matrix, sim.groups, { anchor: [0, 99] })).toThrow(
      /no item at index 99/,
    );
  });

  it('reports an empty criterion on the row rather than throwing', () => {
    const sim = simulateDif({ bank: ladder(4), referenceCount: 60, focalCount: 60, seed: 4 });
    const scan = scanBank(sim.matrix, sim.groups, { anchor: [2], includeStudied: false });
    expect(scan.rows[2]?.note).toMatch(/matching criterion is empty/);
    expect(scan.rows[2]?.classification).toBeNull();
    expect(scan.rows[0]?.classification).not.toBeNull();
  });
});

describe('purifiedScan', () => {
  /** Four items biased the same way, plus a real ability difference. */
  const contaminated = simulateDif({
    bank: BANK,
    focalMean: -0.4,
    shifts: [2, 7, 12, 17].map((item) => ({ item, difficulty: 1.2 })),
    seed: 16,
  });

  it('removes the false flags an unpurified scan produces', () => {
    // Four biased items drag the total score down for the focal group, so the
    // matching criterion itself is biased, and clean items measured against it
    // look biased the other way. The unpurified scan flags six items as large
    // and nine in total; purification converges on exactly the four planted.
    const plain = scanBank(contaminated.matrix, contaminated.groups);
    expect(flaggedAt(plain, 'C')).toEqual([2, 7, 12, 16, 17, 18]);

    const pure = purifiedScan(contaminated.matrix, contaminated.groups);
    expect(flaggedAt(pure.scan, 'C')).toEqual([2, 7, 12, 17]);
    expect(pure.scan.flagged).toEqual([2, 7, 12, 17]);
    expect(pure.converged).toBe(true);
    expect(pure.note).toBeNull();
  });

  it('reaches a fixed point and reports the rounds it took', () => {
    const pure = purifiedScan(contaminated.matrix, contaminated.groups);
    expect(pure.iterations).toBe(3);
    expect(pure.history).toEqual([
      [2, 7, 12, 16, 17, 18],
      [2, 7, 12, 17],
      [2, 7, 12, 17],
    ]);
    // The last two rounds agreed, which is what convergence means here.
    expect(pure.history.at(-1)).toEqual(pure.history.at(-2));
  });

  it('returns the purified criterion, which excludes the flagged items', () => {
    const pure = purifiedScan(contaminated.matrix, contaminated.groups);
    expect(pure.anchor).not.toContain(2);
    expect(pure.anchor).not.toContain(17);
    expect(pure.anchor).toContain(0);
    expect(pure.anchor).toHaveLength(16);
  });

  it('is a fixed point immediately on a bank with nothing to remove', () => {
    // Nothing is flagged, so the criterion for the next round would be the one
    // just used. Re-running it would burn a full scan to learn nothing.
    const clean = simulateDif({ bank: BANK, focalMean: -0.8, seed: 12 });
    const pure = purifiedScan(clean.matrix, clean.groups);
    expect(pure.iterations).toBe(1);
    expect(pure.converged).toBe(true);
    expect(pure.history).toEqual([[]]);
    expect(pure.anchor).toHaveLength(20);
  });

  it('stops rather than shrinking the criterion past its floor', () => {
    const pure = purifiedScan(contaminated.matrix, contaminated.groups, {
      flagAt: 'B',
      minimumAnchor: 19,
    });
    expect(pure.converged).toBe(false);
    expect(pure.note).toMatch(/below the floor of 19/);
    // The scan handed back is the last one that was actually run.
    expect(pure.scan.rows).toHaveLength(20);
  });

  it('gives up honestly when the flagged set will not settle', () => {
    const pure = purifiedScan(contaminated.matrix, contaminated.groups, {
      flagAt: 'B',
      maxIterations: 1,
    });
    expect(pure.converged).toBe(false);
    expect(pure.note).toMatch(/still moving after 1 rounds/);
  });

  it('rejects a non-positive iteration budget', () => {
    expect(() =>
      purifiedScan(contaminated.matrix, contaminated.groups, { maxIterations: 0 }),
    ).toThrow(/maxIterations/);
  });
});

describe('simulateDif', () => {
  it('leaves the reference group answering the unshifted bank', () => {
    const sim = simulateDif({
      bank: BANK,
      shifts: [{ item: 3, difficulty: 0.8, discrimination: 2 }],
      referenceCount: 5,
      focalCount: 5,
      seed: 7,
    });
    expect(sim.focalBank[3]?.parameters.b).toBeCloseTo((BANK[3]?.parameters.b ?? 0) + 0.8, 12);
    expect(sim.focalBank[3]?.parameters.a).toBeCloseTo((BANK[3]?.parameters.a ?? 0) * 2, 12);
    expect(sim.focalBank[4]).toEqual(BANK[4]);
    expect(sim.shifted).toEqual([3]);
    expect(sim.groups.filter((g) => g === 'reference')).toHaveLength(5);
    expect(sim.abilities).toHaveLength(10);
    expect(sim.matrix.personCount).toBe(10);
    expect(sim.matrix.itemCount).toBe(20);
  });

  it('repeats exactly for a given seed and differs for another', () => {
    const options = { bank: BANK, referenceCount: 30, focalCount: 30, seed: 99 } as const;
    expect(simulateDif(options).matrix.toArray()).toEqual(simulateDif(options).matrix.toArray());
    expect(simulateDif({ ...options, seed: 100 }).matrix.toArray()).not.toEqual(
      simulateDif(options).matrix.toArray(),
    );
  });

  it('moves the focal group down the scale when given impact', () => {
    const sim = simulateDif({
      bank: BANK,
      referenceCount: 400,
      focalCount: 400,
      focalMean: -1,
      seed: 8,
    });
    const reference = sim.abilities.slice(0, 400).reduce((a, b) => a + b, 0) / 400;
    const focal = sim.abilities.slice(400).reduce((a, b) => a + b, 0) / 400;
    expect(reference - focal).toBeCloseTo(1, 0);
  });

  it('rejects an empty bank, a bad count and a bad shift', () => {
    expect(() => simulateDif({ bank: [] })).toThrow(/the bank is empty/);
    expect(() => simulateDif({ bank: BANK, focalCount: 0 })).toThrow(/focalCount/);
    expect(() => simulateDif({ bank: BANK, abilitySd: 0 })).toThrow(/abilitySd/);
    expect(() => simulateDif({ bank: BANK, shifts: [{ item: 99, difficulty: 1 }] })).toThrow(
      /no item at index 99/,
    );
    expect(() =>
      simulateDif({ bank: BANK, shifts: [{ item: 1, discrimination: 0 }] }),
    ).toThrow(/discrimination multiplier must be positive/);
  });
});
