import { describe, expect, it } from 'vitest';
import { linspace } from '../src/core/numeric.js';
import { makeItem, twoPL, type Item } from '../src/models/item.js';
import { standardError, testInformation } from '../src/models/response.js';
import { bankHealth, healthTable } from '../src/calibration/health.js';
import { syntheticBank } from '../src/simulation/bank.js';

/** A bank spread evenly over `[from, to]`, all equally discriminating. */
function spread(from: number, to: number, count: number, a = 1.2): Item[] {
  return linspace(from, to, count).map((b, index) => makeItem(`i-${index}`, twoPL(a, b)));
}

describe('bankHealth', () => {
  const healthy = syntheticBank({ size: 300, seed: 20260101 });

  it('evaluates the requested grid over the requested range', () => {
    const report = bankHealth(healthy, { range: [-2, 2], points: 9 });
    expect(report.points).toHaveLength(9);
    expect(report.points[0]?.theta).toBeCloseTo(-2, 12);
    expect(report.points[8]?.theta).toBeCloseTo(2, 12);
    expect(report.range).toEqual([-2, 2]);
    expect(report.items).toBe(300);
  });

  it('agrees with the test information function it is built on', () => {
    for (const point of bankHealth(healthy, { points: 5 }).points) {
      expect(point.information).toBeCloseTo(testInformation(healthy, point.theta), 9);
      expect(point.standardError).toBeCloseTo(standardError(point.information), 9);
    }
  });

  it('reports a well-stocked bank as mostly covered', () => {
    const report = bankHealth(healthy, { target: 0.3 });
    expect(report.covered).toBeGreaterThan(0.7);
    expect(report.peak).toBeGreaterThan(-1);
    expect(report.peak).toBeLessThan(1);
  });

  it('names the region a bank with a deliberate hole cannot measure', () => {
    // Every item above a difficulty of 1 removed: the bank should say so, in
    // the region where it now has nothing.
    const holed = healthy.filter((item) => item.parameters.b < 1);
    const report = bankHealth(holed, { target: 0.3 });
    const upperGap = report.gaps.find((gap) => gap.from > 0);
    expect(upperGap).toBeDefined();
    expect(upperGap?.from).toBeGreaterThan(1);
    expect(upperGap?.worstStandardError).toBeGreaterThan(0.3);
    expect(report.covered).toBeLessThan(bankHealth(healthy, { target: 0.3 }).covered);
  });

  it('reports one gap per contiguous region, not one per failing point', () => {
    const narrow = spread(-0.2, 0.2, 12);
    const report = bankHealth(narrow, { range: [-3, 3], points: 25, target: 0.3 });
    // A bank clustered at zero fails everywhere, and that is one gap spanning
    // the whole range rather than twenty-five separate findings.
    expect(report.gaps).toHaveLength(1);
    expect(report.gaps[0]?.from).toBeCloseTo(-3, 12);
    expect(report.gaps[0]?.to).toBeCloseTo(3, 12);
    expect(report.covered).toBe(0);
  });

  it('splits gaps at either end around a covered middle', () => {
    const middle = spread(-0.6, 0.6, 60, 1.6);
    const report = bankHealth(middle, { range: [-3, 3], points: 25, target: 0.3 });
    expect(report.gaps.length).toBe(2);
    expect(report.gaps[0]?.from).toBeCloseTo(-3, 12);
    expect(report.gaps[1]?.to).toBeCloseTo(3, 12);
    expect(report.covered).toBeGreaterThan(0);
    expect(report.covered).toBeLessThan(1);
  });

  it('reports no gaps for a bank that covers the whole range', () => {
    const wide = spread(-3, 3, 400, 1.5);
    const report = bankHealth(wide, { range: [-3, 3], target: 0.3 });
    expect(report.gaps).toEqual([]);
    expect(report.covered).toBe(1);
  });

  it('tightens as the target does', () => {
    const loose = bankHealth(healthy, { target: 1 });
    const strict = bankHealth(healthy, { target: 0.15 });
    expect(strict.covered).toBeLessThanOrEqual(loose.covered);
    expect(loose.covered).toBe(1);
  });

  it('counts the items whose information peaks near each point', () => {
    const report = bankHealth(spread(-2, 2, 41), { range: [-2, 2], points: 5 });
    // Items are one tenth of a logit apart, so about eleven fall within the
    // half-logit window around an interior point.
    expect(report.points[2]?.itemsNearby).toBe(11);
    // The window is one-sided at the edge of the range.
    expect(report.points[0]?.itemsNearby).toBe(6);
  });

  it('summarises the difficulty and discrimination distributions', () => {
    const report = bankHealth(spread(-2, 2, 21, 1.3));
    expect(report.difficulty.min).toBeCloseTo(-2, 12);
    expect(report.difficulty.max).toBeCloseTo(2, 12);
    expect(report.difficulty.mean).toBeCloseTo(0, 12);
    expect(report.meanDiscrimination).toBeCloseTo(1.3, 12);
  });

  it('rejects an empty bank and a degenerate configuration', () => {
    expect(() => bankHealth([])).toThrow(RangeError);
    expect(() => bankHealth(healthy, { range: [1, 1] })).toThrow(RangeError);
    expect(() => bankHealth(healthy, { points: 1 })).toThrow(RangeError);
    expect(() => bankHealth(healthy, { target: 0 })).toThrow(RangeError);
  });
});

describe('healthTable', () => {
  it('renders one row per grid point and marks the misses', () => {
    const report = bankHealth(spread(-0.2, 0.2, 8), { range: [-3, 3], points: 7, target: 0.3 });
    const lines = healthTable(report).trimEnd().split('\n');
    expect(lines).toHaveLength(9);
    expect(lines[0]).toContain('theta');
    expect(healthTable(report)).toContain('MISSED');
  });

  it('marks a covered point as met', () => {
    const report = bankHealth(spread(-3, 3, 400, 1.5), { points: 5, target: 0.3 });
    expect(healthTable(report)).toContain('met');
    expect(healthTable(report)).not.toContain('MISSED');
  });
});
