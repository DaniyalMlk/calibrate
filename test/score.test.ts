import { describe, expect, it } from 'vitest';
import { linspace } from '../src/core/numeric.js';
import { createRng } from '../src/core/random.js';
import { normalGaussHermiteRule, normalGridRule } from '../src/core/quadrature.js';
import { makeItem, onePL, twoPL, type Item } from '../src/models/item.js';
import { graded, partialCredit } from '../src/models/polytomous.js';
import {
  makePolytomousItem,
  maximumTestScore,
  type AnyItem,
  type AnyResponse,
} from '../src/models/mixed.js';
import { probabilityCorrect } from '../src/models/response.js';
import { estimateEap } from '../src/estimation/bayes.js';
import { STANDARD_NORMAL_PRIOR } from '../src/estimation/prior.js';
import { scoreLikelihoods } from '../src/scoring/summed.js';
import {
  conversionTable,
  conversionText,
  marginalReliability,
  scoreFromTotal,
  scorePattern,
} from '../src/scoring/score.js';

const rule = normalGaussHermiteRule(0, 1, 61);

/** A Rasch form: equal discriminations, so the total score is sufficient. */
const raschForm: Item[] = linspace(-1.8, 1.8, 8).map((b, index) =>
  makeItem(`r-${index}`, onePL(b)),
);

/** A 2PL form: discriminations deliberately spread, so the total is not. */
const twoPlForm: Item[] = linspace(-1.8, 1.8, 8).map((b, index) =>
  makeItem(`t-${index}`, twoPL(0.5 + (index % 4) * 0.55, b)),
);

const mixedForm: AnyItem[] = [
  makeItem('b0', onePL(-0.8)),
  makePolytomousItem('r0', graded(1, [-1, 0.1, 1.2])),
  makeItem('b1', onePL(0.4)),
  makePolytomousItem('r1', partialCredit([-0.3, 0.9])),
];

function respond(items: readonly Item[], pattern: readonly number[]): AnyResponse[] {
  return items.map((item, index) => ({ item, response: pattern[index] as 0 | 1 }));
}

/** Every binary pattern of a given length, as an array of 0/1 arrays. */
function allPatterns(length: number): number[][] {
  const out: number[][] = [];
  for (let mask = 0; mask < 2 ** length; mask += 1) {
    out.push(Array.from({ length }, (_, bit) => (mask >> bit) & 1));
  }
  return out;
}

describe('scorePattern', () => {
  it('agrees with the normal-prior estimator when the rule is a standard normal', () => {
    const responses = respond(raschForm, [1, 1, 0, 1, 0, 0, 1, 0]);
    const here = scorePattern(responses, rule);
    const there = estimateEap(responses, { prior: STANDARD_NORMAL_PRIOR, points: 61 });
    expect(here.ability).toBeCloseTo(there.theta, 8);
    expect(here.standardError).toBeCloseTo(there.standardError, 8);
  });

  it('returns the population itself when there are no responses', () => {
    const empty = scorePattern([], rule);
    expect(empty.ability).toBeCloseTo(0, 10);
    expect(empty.standardError).toBeCloseTo(1, 6);
  });

  it('follows a population that is not centred where the default prior is', () => {
    const responses = respond(raschForm, [1, 1, 1, 1, 0, 0, 0, 0]);
    const standard = scorePattern(responses, rule);
    const shifted = scorePattern(responses, normalGaussHermiteRule(1.5, 1, 61));
    expect(shifted.ability).toBeGreaterThan(standard.ability);
  });

  it('is finite for a perfect and for a zero score', () => {
    for (const pattern of [
      [1, 1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0, 0, 0, 0],
    ]) {
      const score = scorePattern(respond(raschForm, pattern), rule);
      expect(Number.isFinite(score.ability)).toBe(true);
      expect(score.standardError).toBeGreaterThan(0);
    }
  });

  it('is monotone in the number of correct answers, on a Rasch form', () => {
    let previous = Number.NEGATIVE_INFINITY;
    for (let correct = 0; correct <= raschForm.length; correct += 1) {
      const pattern = raschForm.map((_, index) => (index < correct ? 1 : 0));
      const score = scorePattern(respond(raschForm, pattern), rule);
      expect(score.ability).toBeGreaterThan(previous);
      previous = score.ability;
    }
  });

  it('rejects a malformed rule', () => {
    expect(() => scorePattern([], { nodes: [0, 1], weights: [1] })).toThrow(RangeError);
  });
});

describe('conversionTable', () => {
  it('covers every attainable total, with proportions that sum to one', () => {
    for (const form of [raschForm, twoPlForm, mixedForm]) {
      const rows = conversionTable(form, rule);
      expect(rows).toHaveLength(maximumTestScore(form) + 1);
      let mass = 0;
      for (const [index, row] of rows.entries()) {
        expect(row.score).toBe(index);
        expect(row.proportion).toBeGreaterThanOrEqual(0);
        mass += row.proportion;
      }
      expect(mass).toBeCloseTo(1, 10);
    }
  });

  it('is strictly increasing in the total score', () => {
    for (const form of [raschForm, twoPlForm, mixedForm]) {
      const rows = conversionTable(form, rule);
      for (let index = 1; index < rows.length; index += 1) {
        expect(rows[index]?.ability).toBeGreaterThan(rows[index - 1]?.ability as number);
      }
    }
  });

  it('measures least precisely at the extremes', () => {
    const rows = conversionTable(raschForm, rule);
    const middle = rows[Math.floor(rows.length / 2)]?.standardError as number;
    expect(rows[0]?.standardError).toBeGreaterThan(middle);
    expect(rows[rows.length - 1]?.standardError).toBeGreaterThan(middle);
  });

  it('reproduces the pattern estimate exactly under the Rasch model', () => {
    // The total score is a sufficient statistic for ability under Rasch: two
    // candidates with the same total have proportional likelihood functions and
    // therefore identical posteriors. Reporting from the total is then not an
    // approximation at all, and every one of the 256 patterns on this form has
    // to land on its own row of the table.
    const rows = conversionTable(raschForm, rule);
    for (const pattern of allPatterns(raschForm.length)) {
      const total = pattern.reduce((sum, value) => sum + value, 0);
      const fromPattern = scorePattern(respond(raschForm, pattern), rule);
      expect(fromPattern.ability).toBeCloseTo(rows[total]?.ability as number, 9);
      expect(fromPattern.standardError).toBeCloseTo(rows[total]?.standardError as number, 9);
    }
  });

  it('does not reproduce it under the 2PL, and the gap is real', () => {
    // With discriminations that differ, which items were answered correctly
    // matters and the total cannot say. The table is then a genuine summary
    // rather than an identity, and the spread within a total is what summed
    // score reporting gives up.
    const rows = conversionTable(twoPlForm, rule);
    let worst = 0;
    for (const pattern of allPatterns(twoPlForm.length)) {
      const total = pattern.reduce((sum, value) => sum + value, 0);
      const fromPattern = scorePattern(respond(twoPlForm, pattern), rule);
      worst = Math.max(worst, Math.abs(fromPattern.ability - (rows[total]?.ability as number)));
    }
    expect(worst).toBeGreaterThan(0.15);
  });

  it('has the table ability as the average of the patterns that produce it', () => {
    // What the row actually is: the posterior given the total, which is the
    // mixture of the pattern posteriors weighted by how likely each pattern is.
    // Checking it at one total pins down both the likelihood table and the
    // moments taken from it.
    const target = 5;
    const rows = conversionTable(twoPlForm, rule);
    let mass = 0;
    let weighted = 0;
    for (const pattern of allPatterns(twoPlForm.length)) {
      if (pattern.reduce((sum, value) => sum + value, 0) !== target) continue;
      // The unconditional probability of the pattern, averaged over the population.
      let probability = 0;
      for (const [k, node] of rule.nodes.entries()) {
        let conditional = 1;
        for (const [index, item] of twoPlForm.entries()) {
          const p = probabilityCorrect(item.parameters, node);
          conditional *= pattern[index] === 1 ? p : 1 - p;
        }
        probability += (rule.weights[k] as number) * conditional;
      }
      mass += probability;
      weighted += probability * scorePattern(respond(twoPlForm, pattern), rule).ability;
    }
    expect(weighted / mass).toBeCloseTo(rows[target]?.ability as number, 8);
  });

  it('recovers the abilities that generated a simulated cohort', () => {
    const rng = createRng(151515);
    const rows = conversionTable(raschForm, rule);
    let bias = 0;
    let absolute = 0;
    const people = 4000;
    for (let person = 0; person < people; person += 1) {
      const theta = rng.nextNormal();
      let total = 0;
      for (const item of raschForm) {
        if (rng.next() < probabilityCorrect(item.parameters, theta)) total += 1;
      }
      const estimate = rows[total]?.ability as number;
      bias += estimate - theta;
      absolute += Math.abs(estimate - theta);
    }
    // A posterior mean shrinks towards the population, so it is biased inward
    // by construction; what it should not be is biased in one direction overall.
    expect(Math.abs(bias / people)).toBeLessThan(0.05);
    expect(absolute / people).toBeLessThan(0.6);
  });

  it('agrees between a Gauss-Hermite rule and a dense grid', () => {
    const gauss = conversionTable(raschForm, rule);
    const grid = conversionTable(raschForm, normalGridRule(0, 1, 121, 5));
    for (const [index, row] of gauss.entries()) {
      expect(grid[index]?.ability).toBeCloseTo(row.ability, 3);
    }
  });

  it('rejects a total the form cannot produce', () => {
    const likelihoods = scoreLikelihoods(raschForm, rule);
    expect(() => scoreFromTotal(likelihoods, raschForm.length + 1, rule)).toThrow(RangeError);
  });
});

describe('marginalReliability', () => {
  it('lies in the unit interval and matches its own definition', () => {
    const reliability = marginalReliability(raschForm, rule);
    expect(reliability.marginal).toBeGreaterThan(0);
    expect(reliability.marginal).toBeLessThan(1);
    expect(reliability.marginal).toBeCloseTo(
      1 - reliability.meanSquaredError / reliability.populationVariance,
      12,
    );
    expect(reliability.populationVariance).toBeCloseTo(1, 6);
  });

  it('rises as the form gets longer', () => {
    const short = marginalReliability(raschForm.slice(0, 4), rule);
    const medium = marginalReliability(raschForm, rule);
    const long = marginalReliability([...raschForm, ...raschForm, ...raschForm], rule);
    expect(medium.marginal).toBeGreaterThan(short.marginal);
    expect(long.marginal).toBeGreaterThan(medium.marginal);
  });

  it('rises as the items get more discriminating', () => {
    const flat = linspace(-1.8, 1.8, 8).map((b, index) => makeItem(`f-${index}`, twoPL(0.5, b)));
    const sharp = linspace(-1.8, 1.8, 8).map((b, index) => makeItem(`s-${index}`, twoPL(2.0, b)));
    expect(marginalReliability(sharp, rule).marginal).toBeGreaterThan(
      marginalReliability(flat, rule).marginal,
    );
  });

  it('agrees with the variance the estimates actually recover', () => {
    // The claim marginal reliability makes, checked against a cohort: the
    // squared correlation between the reported ability and the generating one
    // should be near the reliability quoted for the form.
    const rng = createRng(626262);
    const rows = conversionTable(raschForm, rule);
    const truth: number[] = [];
    const estimates: number[] = [];
    for (let person = 0; person < 6000; person += 1) {
      const theta = rng.nextNormal();
      let total = 0;
      for (const item of raschForm) {
        if (rng.next() < probabilityCorrect(item.parameters, theta)) total += 1;
      }
      truth.push(theta);
      estimates.push(rows[total]?.ability as number);
    }

    const meanOf = (values: readonly number[]): number =>
      values.reduce((sum, value) => sum + value, 0) / values.length;
    const mt = meanOf(truth);
    const me = meanOf(estimates);
    let covariance = 0;
    let varT = 0;
    let varE = 0;
    for (const [index, value] of truth.entries()) {
      const dt = value - mt;
      const de = (estimates[index] as number) - me;
      covariance += dt * de;
      varT += dt * dt;
      varE += de * de;
    }
    const squared = (covariance * covariance) / (varT * varE);
    const quoted = marginalReliability(raschForm, rule).marginal;
    expect(squared).toBeGreaterThan(quoted - 0.08);
    expect(squared).toBeLessThan(quoted + 0.08);
  });

  it('works on a form mixing both item formats', () => {
    const reliability = marginalReliability(mixedForm, rule);
    expect(reliability.marginal).toBeGreaterThan(0);
    expect(reliability.marginal).toBeLessThan(1);
    expect(reliability.meanStandardError).toBeGreaterThan(0);
  });
});

describe('conversionText', () => {
  it('renders one line per total, under a header', () => {
    const rows = conversionTable(raschForm, rule);
    const text = conversionText(rows);
    const lines = text.trimEnd().split('\n');
    expect(lines).toHaveLength(rows.length + 2);
    expect(lines[0]).toContain('theta');
    expect(lines[2]).toMatch(/^\s+0\s+-?\d+\.\d{3}\s+\d+\.\d{3}\s+\d+\.\d%$/);
  });
});
