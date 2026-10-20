import { describe, expect, it } from 'vitest';
import { linspace, mean, variance } from '../src/core/numeric.js';
import { createRng } from '../src/core/random.js';
import { makeItem, onePL, twoPL, type Item } from '../src/models/item.js';
import { graded, partialCredit } from '../src/models/polytomous.js';
import { makePolytomousItem, type AnyResponse } from '../src/models/mixed.js';
import { probabilityCorrect } from '../src/models/response.js';
import { estimateMle } from '../src/estimation/mle.js';
import { personFit, scanPersons } from '../src/estimation/person.js';
import { simulateResponses } from '../src/simulation/respondent.js';

/** Forty items spread over the range, with discriminations that vary. */
const form: Item[] = linspace(-2, 2, 40).map((b, index) =>
  makeItem(`i${index}`, twoPL(0.7 + ((index * 7) % 40) / 40, b)),
);

/** The one-sided 5% point of the standard normal. */
const CRITICAL = -1.6448536269514722;

interface Sampled {
  readonly atTruth: number[];
  readonly atEstimate: number[];
  readonly corrected: number[];
  readonly infit: number[];
  readonly outfit: number[];
}

/** Model-conforming candidates, scored three ways. */
function sample(count: number, seed: number): Sampled {
  const rng = createRng(seed);
  const out: Sampled = { atTruth: [], atEstimate: [], corrected: [], infit: [], outfit: [] };
  for (let person = 0; person < count; person += 1) {
    const theta = rng.nextNormal();
    const responses = simulateResponses(form, theta, rng);
    const estimate = estimateMle(responses);
    if (!estimate.converged) continue;
    out.atTruth.push(personFit(responses, theta).lz);
    const fit = personFit(responses, estimate.theta);
    out.atEstimate.push(fit.lz);
    out.corrected.push(fit.lzStar as number);
    out.infit.push(fit.infit);
    out.outfit.push(fit.outfit);
  }
  return out;
}

const sampled = sample(3000, 20260101);
const rejectionRate = (values: readonly number[]): number =>
  values.filter((value) => value < CRITICAL).length / values.length;

describe('personFit: the null distribution', () => {
  it('is standard normal when the ability is known', () => {
    expect(Math.abs(mean(sampled.atTruth))).toBeLessThan(0.06);
    expect(Math.sqrt(variance(sampled.atTruth))).toBeGreaterThan(0.95);
    expect(Math.sqrt(variance(sampled.atTruth))).toBeLessThan(1.05);
  });

  it('rejects at about the nominal rate when the ability is known', () => {
    expect(rejectionRate(sampled.atTruth)).toBeGreaterThan(0.035);
    expect(rejectionRate(sampled.atTruth)).toBeLessThan(0.07);
  });

  it('is too narrow once the ability is estimated from the same responses', () => {
    // This is the whole reason the correction exists, so the test asserts the
    // problem rather than only the fix. A suite that checked lzStar alone
    // would pass just as well against a correction applied to a statistic that
    // never needed one.
    const spread = Math.sqrt(variance(sampled.atEstimate));
    expect(spread).toBeLessThan(0.95);
    expect(spread).toBeGreaterThan(0.85);
  });

  it('rejects less often than it claims, once the ability is estimated', () => {
    // Conservative: the direction that quietly does nothing.
    expect(rejectionRate(sampled.atEstimate)).toBeLessThan(0.045);
  });

  it('is brought back to about unit variance by the correction', () => {
    const spread = Math.sqrt(variance(sampled.corrected));
    expect(spread).toBeGreaterThan(0.96);
    expect(spread).toBeLessThan(1.06);
    // And it is closer to one than the uncorrected statistic was.
    expect(Math.abs(spread - 1)).toBeLessThan(
      Math.abs(Math.sqrt(variance(sampled.atEstimate)) - 1),
    );
  });

  it('puts the rejection rate back near nominal', () => {
    expect(rejectionRate(sampled.corrected)).toBeGreaterThan(0.04);
    expect(rejectionRate(sampled.corrected)).toBeLessThan(0.075);
  });

  it('centres both mean squares near one', () => {
    expect(mean(sampled.infit)).toBeGreaterThan(0.93);
    expect(mean(sampled.infit)).toBeLessThan(1.07);
    expect(mean(sampled.outfit)).toBeGreaterThan(0.9);
    expect(mean(sampled.outfit)).toBeLessThan(1.1);
  });
});

describe('personFit: the correction is a projection', () => {
  it('never moves the statistic towards zero', () => {
    // The corrected variance is the plain one minus a non-negative term, so
    // the corrected statistic has the same sign and at least the magnitude.
    const rng = createRng(4242);
    for (let person = 0; person < 200; person += 1) {
      const theta = rng.nextNormal();
      const responses = simulateResponses(form, theta, rng);
      const estimate = estimateMle(responses);
      if (!estimate.converged) continue;
      const fit = personFit(responses, estimate.theta);
      const corrected = fit.lzStar as number;
      expect(Math.sign(corrected)).toBe(Math.sign(fit.lz));
      expect(Math.abs(corrected)).toBeGreaterThanOrEqual(Math.abs(fit.lz) - 1e-12);
    }
  });
});

describe('personFit: the moments over categories', () => {
  it('reduces to the dichotomous formula exactly', () => {
    // The moments are summed over categories so a rubric item needs no second
    // code path. For a two-category item that has to come out at the familiar
    // `P lnP + Q lnQ` and `PQ ln(P/Q)^2` — not nearly, since it is the same
    // arithmetic rearranged.
    for (const item of form.slice(0, 12)) {
      for (const theta of [-1.5, -0.4, 0, 0.9, 2.2]) {
        const p = probabilityCorrect(item.parameters, theta);
        const q = 1 - p;
        const expectedLogLikelihood = p * Math.log(p) + q * Math.log(q);
        const expectedVariance = p * q * Math.log(p / q) ** 2;

        const fit = personFit([{ item, response: 1 }], theta);
        expect(fit.expectedLogLikelihood).toBeCloseTo(expectedLogLikelihood, 14);
        expect(fit.logLikelihood).toBeCloseTo(Math.log(p), 14);
        // Recovered from the statistic, which is the only way the variance is
        // observable from outside.
        const recovered = ((fit.logLikelihood - fit.expectedLogLikelihood) / fit.lz) ** 2;
        expect(recovered).toBeCloseTo(expectedVariance, 12);
      }
    }
  });

  it('scores a rubric item and refuses to correct it', () => {
    const responses: AnyResponse[] = [
      { item: makeItem('b', twoPL(1.2, 0)), response: 1 },
      { item: makePolytomousItem('r', graded(1.1, [-1, 0, 1])), category: 2 },
    ];
    const fit = personFit(responses, 0.3);
    expect(Number.isFinite(fit.lz)).toBe(true);
    // The correction is stated for the weights of the dichotomous score
    // equation. A number here would look right and mean nothing.
    expect(fit.lzStar).toBeNull();
  });

  it('handles a partial credit item at every category', () => {
    const item = makePolytomousItem('pc', partialCredit([-0.5, 0.8]));
    for (const category of [0, 1, 2]) {
      const fit = personFit([{ item, category }], 0.2);
      expect(Number.isFinite(fit.lz)).toBe(true);
      expect(Number.isFinite(fit.outfit)).toBe(true);
    }
  });

  it('refuses a category the item does not have', () => {
    const item = makePolytomousItem('pc', partialCredit([-0.5, 0.8]));
    expect(() => personFit([{ item, category: 7 }], 0)).toThrow(/category 7 is not one of them/);
  });

  it('refuses an empty pattern and a non-finite ability', () => {
    expect(() => personFit([], 0)).toThrow(/at least one response/);
    expect(() => personFit([{ item: form[0] as Item, response: 1 }], Number.NaN)).toThrow(
      /must be finite/,
    );
  });

  it('does not produce an infinite log-likelihood for a certainty', () => {
    // A 3PL item far below a candidate's ability has Q underflowing to zero,
    // and ln(0) would take the whole statistic to NaN.
    const certain = makeItem('easy', twoPL(2.5, -12));
    const fit = personFit([certain].map((item) => ({ item, response: 0 as const })), 4);
    expect(Number.isFinite(fit.logLikelihood)).toBe(true);
    expect(Number.isFinite(fit.expectedLogLikelihood)).toBe(true);
  });
});

describe('personFit: what it flags', () => {
  /** A strong candidate who missed the easiest items on the form. */
  function spuriouslyLow(): AnyResponse[] {
    const sorted = [...form].sort(
      (left, right) =>
        probabilityCorrect(right.parameters, 1.5) - probabilityCorrect(left.parameters, 1.5),
    );
    const slipped = new Set(sorted.slice(0, 8).map((item) => item.id));
    const rng = createRng(31337);
    return form.map((item) => ({
      item,
      response: slipped.has(item.id)
        ? (0 as const)
        : rng.next() < probabilityCorrect(item.parameters, 1.5)
          ? (1 as const)
          : (0 as const),
    }));
  }

  it('flags a pattern that misses what it should not', () => {
    const responses = spuriouslyLow();
    const fit = personFit(responses, estimateMle(responses).theta);
    expect(fit.lz).toBeLessThan(CRITICAL);
    expect(fit.outfit).toBeGreaterThan(1.3);
  });

  it('leaves a pattern generated by the model alone', () => {
    const rng = createRng(999);
    const responses = simulateResponses(form, 0.4, rng);
    const fit = personFit(responses, estimateMle(responses).theta);
    expect(fit.lz).toBeGreaterThan(CRITICAL);
  });
});

describe('scanPersons', () => {
  const rng = createRng(606);
  const sitting = Array.from({ length: 60 }, (_, index) => {
    const theta = rng.nextNormal();
    const responses = simulateResponses(form, theta, rng);
    return { id: `p-${index}`, responses, theta: estimateMle(responses).theta };
  });

  it('returns the worst patterns first', () => {
    const flagged = scanPersons(sitting, { threshold: -0.5 });
    expect(flagged.length).toBeGreaterThan(0);
    for (let index = 1; index < flagged.length; index += 1) {
      expect(flagged[index]?.statistic).toBeGreaterThanOrEqual(
        flagged[index - 1]?.statistic as number,
      );
    }
  });

  it('flags more at a looser threshold, and the sets nest', () => {
    const strict = new Set(scanPersons(sitting, { threshold: -2 }).map((row) => row.id));
    const loose = new Set(scanPersons(sitting, { threshold: -1 }).map((row) => row.id));
    expect(loose.size).toBeGreaterThanOrEqual(strict.size);
    for (const id of strict) expect(loose.has(id)).toBe(true);
  });

  it('uses the corrected statistic by default, and the plain one when asked', () => {
    const corrected = scanPersons(sitting, { threshold: -1 });
    const plain = scanPersons(sitting, { threshold: -1, corrected: false });
    // lzStar is never nearer zero, so it can only flag at least as many.
    expect(corrected.length).toBeGreaterThanOrEqual(plain.length);
    for (const row of corrected) expect(row.statistic).toBe(row.fit.lzStar);
    for (const row of plain) expect(row.statistic).toBe(row.fit.lz);
  });

  it('refuses a threshold on the wrong side of zero', () => {
    expect(() => scanPersons(sitting, { threshold: 1.5 })).toThrow(/must be negative/);
  });

  it('falls back to the plain statistic for a pattern it cannot correct', () => {
    const item = makePolytomousItem('r', graded(1.1, [-1, 0, 1]));
    const responses: AnyResponse[] = [
      { item, category: 0 },
      { item: makeItem('b', onePL(0)), response: 1 },
    ];
    const flagged = scanPersons([{ id: 'mixed', responses, theta: 2.5 }], { threshold: -0.1 });
    if (flagged.length > 0) {
      expect(flagged[0]?.statistic).toBe(flagged[0]?.fit.lz);
    }
  });
});
