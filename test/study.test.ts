import { describe, expect, it } from 'vitest';
import { mean } from '../src/core/numeric.js';
import { blueprint, contentBalanced } from '../src/selection/content.js';
import { randomesque } from '../src/selection/exposure.js';
import { maximumInformationSelector } from '../src/selection/information.js';
import { fixedLength, precisionTarget } from '../src/session/stopping.js';
import { syntheticPool } from '../src/simulation/bank.js';
import { conditionalReport } from '../src/simulation/conditional.js';
import {
  drawPopulation,
  evenGridPopulation,
  normalPopulation,
} from '../src/simulation/population.js';
import {
  comparisonTable,
  conditionalTable,
  exposureSummary,
  summarise,
} from '../src/simulation/report.js';
import { compareStudies, runStudy, simulateSession, type Policy } from '../src/simulation/study.js';

const pool = syntheticPool({ size: 120, seed: 4242 });

const maxInfo: Policy = {
  name: 'max-info',
  selector: maximumInformationSelector(),
  stopping: precisionTarget(0.35, { minimum: 5, maximum: 24 }),
};

const balanced: Policy = {
  name: 'balanced',
  selector: contentBalanced(
    randomesque(maximumInformationSelector(), 5),
    blueprint({ arrays: 0.4, graphs: 0.3, 'dynamic-programming': 0.3 }),
  ),
  stopping: precisionTarget(0.35, { minimum: 5, maximum: 24 }),
};

const twenty: Policy = {
  name: 'fixed-20',
  selector: maximumInformationSelector(),
  stopping: fixedLength(20),
};

const abilities = drawPopulation(normalPopulation(), 40, 17);

describe('simulateSession', () => {
  it('returns the same outcome for the same seed', () => {
    const first = simulateSession(maxInfo, pool, 0.7, 5);
    const second = simulateSession(maxInfo, pool, 0.7, 5);
    expect(second).toEqual(first);
  });

  it('records a transcript consistent with its own summary', () => {
    const outcome = simulateSession(maxInfo, pool, -0.4, 9);
    expect(outcome.itemIds).toHaveLength(outcome.length);
    expect(new Set(outcome.itemIds).size).toBe(outcome.length);
    expect(outcome.trueTheta).toBe(-0.4);
    expect(Number.isFinite(outcome.estimate)).toBe(true);
    expect(outcome.standardError).toBeGreaterThan(0);
  });

  it('honours the stopping rule it was given', () => {
    const outcome = simulateSession(twenty, pool, 0.1, 3);
    expect(outcome.length).toBe(20);
    expect(outcome.stopRule).toContain('fixed-length');
  });
});

describe('runStudy', () => {
  it('produces one outcome per examinee, in order', () => {
    const result = runStudy(maxInfo, { pool, abilities });
    expect(result.outcomes).toHaveLength(abilities.length);
    expect(result.outcomes.map((outcome) => outcome.trueTheta)).toEqual([...abilities]);
    expect(result.poolSize).toBe(pool.size);
    expect(result.policy).toBe('max-info');
  });

  it('gives an examinee the same session however many others are in the study', () => {
    // The property that makes a comparison a comparison: examinee i must not
    // depend on how much randomness examinees before them consumed.
    const small = runStudy(maxInfo, { pool, abilities: abilities.slice(0, 4) });
    const large = runStudy(maxInfo, { pool, abilities });
    expect(large.outcomes.slice(0, 4)).toEqual(small.outcomes);
  });

  it('replays identically for a given seed and differs across seeds', () => {
    const a = runStudy(maxInfo, { pool, abilities, seed: 1 });
    const b = runStudy(maxInfo, { pool, abilities, seed: 1 });
    const c = runStudy(balanced, { pool, abilities, seed: 2 });
    expect(b.outcomes).toEqual(a.outcomes);
    expect(c.outcomes).not.toEqual(a.outcomes);
  });

  it('carries the population name into the result', () => {
    const result = runStudy(maxInfo, { pool, abilities, population: 'normal(0,1)' });
    expect(result.population).toBe('normal(0,1)');
    expect(runStudy(maxInfo, { pool, abilities }).population).toBe('unnamed');
  });

  it('rejects an empty population', () => {
    expect(() => runStudy(maxInfo, { pool, abilities: [] })).toThrow(RangeError);
  });
});

describe('compareStudies', () => {
  it('runs every policy against the same examinees', () => {
    const results = compareStudies([maxInfo, balanced, twenty], { pool, abilities });
    expect(results.map((result) => result.policy)).toEqual(['max-info', 'balanced', 'fixed-20']);
    for (const result of results) {
      expect(result.outcomes.map((outcome) => outcome.trueTheta)).toEqual([...abilities]);
    }
  });

  it('rejects duplicate policy names, which would make the report ambiguous', () => {
    expect(() => compareStudies([maxInfo, maxInfo], { pool, abilities })).toThrow(RangeError);
  });

  it('rejects an empty policy list', () => {
    expect(() => compareStudies([], { pool, abilities })).toThrow(RangeError);
  });
});

describe('summarise', () => {
  const result = runStudy(maxInfo, { pool, abilities, population: 'normal(0,1)' });
  const summary = summarise(result);

  it('recovers ability well enough to correlate strongly with the truth', () => {
    expect(summary.correlation).toBeGreaterThan(0.85);
    expect(Math.abs(summary.bias)).toBeLessThan(0.25);
    expect(summary.rmse).toBeLessThan(0.7);
  });

  it('reports interval coverage in the region a 95% interval should reach', () => {
    expect(summary.coverage).toBeGreaterThan(0.8);
    expect(summary.coverage).toBeLessThanOrEqual(1);
  });

  it('reports lengths consistent with the stopping rule', () => {
    expect(summary.meanLength).toBeGreaterThanOrEqual(5);
    expect(summary.meanLength).toBeLessThanOrEqual(24);
    expect(summary.lengthRange[0]).toBeGreaterThanOrEqual(5);
    expect(summary.lengthRange[1]).toBeLessThanOrEqual(24);
  });

  it('ties the mean exposure rate to the mean test length by identity', () => {
    // Every administered item is one unit of exposure, so the mean rate over the
    // whole bank is exactly the mean test length divided by the bank size.
    expect(summary.exposure.meanRate).toBeCloseTo(summary.meanLength / pool.size, 12);
  });

  it('accounts for every item in the bank as used or unused', () => {
    const { itemsUsed, unused } = summary.exposure;
    expect(itemsUsed + unused * pool.size).toBeCloseTo(pool.size, 9);
    expect(summary.exposure.maxRate).toBeGreaterThan(0);
    expect(summary.exposure.maxRate).toBeLessThanOrEqual(1);
  });

  it('places the overlap rate between its floor and its ceiling', () => {
    expect(summary.exposure.overlap).toBeGreaterThan(summary.meanLength / pool.size);
    expect(summary.exposure.overlap).toBeLessThanOrEqual(1);
  });

  it('counts over-exposed items against the target it was given', () => {
    const strict = exposureSummary(result, 0.05);
    const loose = exposureSummary(result, 0.9);
    expect(strict.aboveTarget).toBeGreaterThanOrEqual(loose.aboveTarget);
    expect(strict.target).toBe(0.05);
  });

  it('rejects a study with no outcomes', () => {
    expect(() => summarise({ ...result, outcomes: [] })).toThrow(RangeError);
  });
});

describe('exposure control against unrestricted maximum information', () => {
  it('spreads the bank further, at a cost in nothing much', () => {
    const [greedy, spread] = compareStudies([maxInfo, balanced], { pool, abilities });
    const greedySummary = summarise(greedy!);
    const spreadSummary = summarise(spread!);

    expect(spreadSummary.exposure.maxRate).toBeLessThan(greedySummary.exposure.maxRate);
    expect(spreadSummary.exposure.unused).toBeLessThan(greedySummary.exposure.unused);
    // The precision given up for that is small — a fraction of a standard error.
    expect(spreadSummary.rmse - greedySummary.rmse).toBeLessThan(0.1);
  });
});

describe('conditionalReport', () => {
  const gridAbilities = drawPopulation(evenGridPopulation(-2.5, 2.5, 11), 44, 1);
  const result = runStudy(twenty, { pool, abilities: gridAbilities });

  it('returns the requested number of bins covering the requested range', () => {
    const bins = conditionalReport(result, { lower: -3, upper: 3, bins: 6 });
    expect(bins).toHaveLength(6);
    expect(bins[0]?.lower).toBeCloseTo(-3, 12);
    expect(bins[5]?.upper).toBeCloseTo(3, 12);
    expect(bins[2]?.center).toBeCloseTo(-0.5, 12);
  });

  it('accounts for every examinee, folding the range edges inward', () => {
    const bins = conditionalReport(result, { lower: -1, upper: 1, bins: 2 });
    expect(bins.reduce((total, bin) => total + bin.count, 0)).toBe(gridAbilities.length);
  });

  it('bins on generating ability, not on the estimate', () => {
    const bins = conditionalReport(result, { lower: -2.5, upper: 2.5, bins: 5 });
    for (const bin of bins) {
      if (bin.count === 0) continue;
      expect(bin.meanTrueTheta).toBeGreaterThanOrEqual(bin.lower - 1e-9);
      expect(bin.meanTrueTheta).toBeLessThanOrEqual(bin.upper + 1e-9);
    }
  });

  it('reports an empty bin as empty rather than as perfect measurement', () => {
    const narrow = runStudy(twenty, { pool, abilities: [0.1, 0.2, 0.3] });
    const bins = conditionalReport(narrow, { lower: -3, upper: 3, bins: 6 });
    const empty = bins.filter((bin) => bin.count === 0);
    expect(empty.length).toBeGreaterThan(0);
    for (const bin of empty) {
      expect(Number.isNaN(bin.rmse)).toBe(true);
      expect(Number.isNaN(bin.coverage)).toBe(true);
    }
  });

  it('decomposes RMSE into bias and error spread', () => {
    for (const bin of conditionalReport(result, { bins: 4 })) {
      if (bin.count === 0) continue;
      expect(bin.rmse * bin.rmse).toBeCloseTo(bin.bias * bin.bias + bin.errorSd * bin.errorSd, 9);
    }
  });

  it('measures the middle of the scale better than the edges', () => {
    // The bank is normal around zero, so the tails are thin and the conditional
    // standard error there is necessarily larger.
    const bins = conditionalReport(result, { lower: -2.5, upper: 2.5, bins: 5 });
    const middle = bins[2] as { meanStandardError: number };
    const edges = [bins[0], bins[4]] as { meanStandardError: number }[];
    expect(middle.meanStandardError).toBeLessThan(mean(edges.map((bin) => bin.meanStandardError)));
  });

  it('rejects a degenerate range or bin count', () => {
    expect(() => conditionalReport(result, { lower: 1, upper: 1 })).toThrow(RangeError);
    expect(() => conditionalReport(result, { bins: 0 })).toThrow(RangeError);
    expect(() => conditionalReport(result, { bins: 1.5 })).toThrow(RangeError);
  });
});

describe('report tables', () => {
  it('renders one row per policy under a header', () => {
    const summaries = compareStudies([maxInfo, balanced], { pool, abilities }).map((result) =>
      summarise(result),
    );
    const table = comparisonTable(summaries);
    const lines = table.trimEnd().split('\n');
    expect(lines).toHaveLength(4);
    expect(lines[0]).toContain('policy');
    expect(lines[0]).toContain('overlap');
    expect(table).toContain('max-info');
    expect(table).toContain('balanced');
  });

  it('renders one row per conditional bin', () => {
    const summary = summarise(runStudy(maxInfo, { pool, abilities }), { bins: 4 });
    const lines = conditionalTable(summary).trimEnd().split('\n');
    expect(lines).toHaveLength(6);
    expect(lines[0]).toContain('ability');
  });

  it('says so when there is nothing to compare', () => {
    expect(comparisonTable([])).toContain('no policies');
  });

  it('prints a dash rather than NaN for an empty bin', () => {
    const summary = summarise(runStudy(twenty, { pool, abilities: [0.1, 0.2] }), { bins: 6 });
    expect(conditionalTable(summary)).not.toContain('NaN');
  });
});
