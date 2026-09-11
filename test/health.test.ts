import { describe, expect, it } from 'vitest';
import { linspace } from '../src/core/numeric.js';
import { makeItem, twoPL, type Item } from '../src/models/item.js';
import { standardError, testInformation } from '../src/models/response.js';
import {
  makePolytomousItem,
  testInformationOf,
  type PolytomousItem,
} from '../src/models/mixed.js';
import { graded, polytomousInformationPeak } from '../src/models/polytomous.js';
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

  it('reports a dichotomous bank as entirely dichotomous', () => {
    const report = bankHealth(spread(-2, 2, 21));
    expect(report.formats).toEqual({ dichotomous: 21, polytomous: 0, maximumScore: 21 });
  });

  it('rejects an empty bank and a degenerate configuration', () => {
    expect(() => bankHealth([])).toThrow(RangeError);
    expect(() => bankHealth(healthy, { range: [1, 1] })).toThrow(RangeError);
    expect(() => bankHealth(healthy, { points: 1 })).toThrow(RangeError);
    expect(() => bankHealth(healthy, { target: 0 })).toThrow(RangeError);
  });
});

describe('bankHealth over a mixed-format bank', () => {
  /** Four-category graded items at the given locations, one logit per step. */
  function rubrics(locations: readonly number[], a = 1.2): PolytomousItem[] {
    return locations.map((location, index) =>
      makePolytomousItem(
        `cr-${index}`,
        graded(a, [location - 1, location, location + 1]),
      ),
    );
  }

  it('agrees with the mixed-format test information function', () => {
    const bank = [...spread(-1, 1, 6), ...rubrics([-0.5, 0.5, 1.5])];
    for (const point of bankHealth(bank, { points: 7 }).points) {
      expect(point.information).toBeCloseTo(testInformationOf(bank, point.theta), 9);
      expect(point.standardError).toBeCloseTo(standardError(point.information), 9);
    }
  });

  it('counts the two formats and the total score the bank could award', () => {
    // Six dichotomous items score one point each; three four-category rubrics
    // score three each, for a maximum of 6 + 9.
    const report = bankHealth([...spread(-1, 1, 6), ...rubrics([-0.5, 0.5, 1.5])]);
    expect(report.formats).toEqual({ dichotomous: 6, polytomous: 3, maximumScore: 15 });
  });

  it('credits the rubric items with the coverage they actually provide', () => {
    // A bank with nothing above zero, then the same bank with three rubrics
    // placed in the upper range. Dropping the polytomous items — which is what
    // the dichotomous-only report did — would leave the upper gap in place.
    const lower = spread(-2.5, 0, 30);
    const withRubrics = [...lower, ...rubrics([1, 1.6, 2.2], 1.8)];

    const before = bankHealth(lower, { range: [-2, 2.5], points: 41, target: 0.5 });
    const after = bankHealth(withRubrics, { range: [-2, 2.5], points: 41, target: 0.5 });

    expect(before.gaps.some((gap) => gap.from > 0.5)).toBe(true);
    expect(after.covered).toBeGreaterThan(before.covered);
    for (const point of after.points) {
      expect(point.information).toBeGreaterThanOrEqual(
        (before.points.find((p) => p.theta === point.theta)?.information ?? 0) - 1e-12,
      );
    }
  });

  it('locates a rubric item by its information peak, not by its threshold mean', () => {
    // One rubric item whose thresholds straddle +1.5. The peak of a graded
    // item's information is near — but not exactly at — the mean threshold, so
    // a report that used the location would put `itemsNearby` in a different
    // cell than the information actually justifies.
    const item = rubrics([1.5], 1.6);
    const report = bankHealth(item, { range: [0, 3], points: 7, target: 5 });
    const near = report.points.filter((point) => point.itemsNearby === 1).map((p) => p.theta);
    expect(near.length).toBeGreaterThan(0);
    for (const theta of near) {
      expect(Math.abs(theta - polytomousInformationPeak(item[0]!.parameters))).toBeLessThanOrEqual(
        0.5 + 1e-9,
      );
    }
  });

  it('summarises a rubric item at the mean of its thresholds', () => {
    // The location summary is the one place a threshold mean is the right
    // answer: it is what puts both formats in one column of a report.
    const report = bankHealth(rubrics([0, 2], 1.4));
    expect(report.difficulty.min).toBeCloseTo(0, 12);
    expect(report.difficulty.max).toBeCloseTo(2, 12);
    expect(report.difficulty.mean).toBeCloseTo(1, 12);
    expect(report.meanDiscrimination).toBeCloseTo(1.4, 12);
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
