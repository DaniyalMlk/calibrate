import { describe, expect, it } from 'vitest';
import {
  classifyLogistic,
  EFFECT_RULES,
  fitLogistic,
  logisticDif,
  nagelkerke,
} from '../src/dif/logistic.js';
import { mantelHaenszel } from '../src/dif/mantel.js';
import { matchedSample, stratify, type DifObservation } from '../src/dif/strata.js';
import { makeItem, twoPL, type Item } from '../src/models/item.js';
import { simulateDif } from '../src/simulation/dif.js';

/**
 * A Lehmer generator. Every product stays under 2^53, so the same recurrence
 * reproduces this dataset exactly in any language with double arithmetic —
 * which is what let the reference fit below be computed elsewhere.
 */
function lehmer(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 16807) % 2147483647;
    return state / 2147483647;
  };
}

/** The fixed dataset the reference coefficients were computed from. */
function referenceData(n = 600, seed = 12345): DifObservation[] {
  const rng = lehmer(seed);
  const out: DifObservation[] = [];
  for (let i = 0; i < n; i += 1) {
    const match = Math.floor(rng() * 21);
    const group = i % 2 === 0 ? ('reference' as const) : ('focal' as const);
    const indicator = group === 'focal' ? 1 : 0;
    const centred = (match - 10) / 6;
    const linear = -0.4 + 1.1 * centred + 0.5 * indicator - 0.45 * centred * indicator;
    out.push({ group, match, score: rng() < 1 / (1 + Math.exp(-linear)) ? 1 : 0 });
  }
  return out;
}

function rawDesign(data: readonly DifObservation[]): number[][] {
  return data.map((d) => {
    const group = d.group === 'focal' ? 1 : 0;
    return [1, d.match, group, d.match * group];
  });
}

const BANK: Item[] = Array.from({ length: 20 }, (_, index) =>
  makeItem(`i-${index}`, twoPL(1 + 0.03 * index, -1.8 + (3.6 * index) / 19)),
);

describe('fitLogistic', () => {
  it('matches an independent implementation on a fixed dataset', () => {
    // Coefficients, standard errors and log-likelihood computed outside this
    // codebase from the same 600 rows.
    const data = referenceData();
    const fit = fitLogistic(rawDesign(data), data.map((d) => d.score));

    const coefficients = [
      -1.9909143835458076, 0.16596656253739406, 1.150264531361964, -0.08284242594368336,
    ];
    const errors = [
      0.2871773263166371, 0.023140262946949947, 0.3686409925605704, 0.03073353828695147,
    ];
    for (const [index, expected] of coefficients.entries()) {
      expect(fit.coefficients[index]).toBeCloseTo(expected, 10);
    }
    for (const [index, expected] of errors.entries()) {
      expect(fit.standardErrors[index]).toBeCloseTo(expected, 10);
    }
    expect(fit.logLikelihood).toBeCloseTo(-374.02830321529797, 8);
    expect(fit.converged).toBe(true);
    expect(fit.iterations).toBeLessThan(12);
  });

  it('reproduces the log-likelihoods of every nested model', () => {
    const data = referenceData();
    const outcomes = data.map((d) => d.score);
    const design = rawDesign(data);
    const column = (indices: readonly number[]): number[][] =>
      design.map((row) => indices.map((index) => row[index] as number));

    expect(fitLogistic(column([0]), outcomes).logLikelihood).toBeCloseTo(-414.9244587775536, 8);
    expect(fitLogistic(column([0, 1]), outcomes).logLikelihood).toBeCloseTo(-379.06611931797636, 8);
    expect(fitLogistic(column([0, 1, 2]), outcomes).logLikelihood).toBeCloseTo(
      -377.7298806051733,
      8,
    );
  });

  it('recovers the log-odds of the base rate from an intercept alone', () => {
    // With one column the maximum likelihood estimate is in closed form.
    const outcomes = Array.from({ length: 200 }, (_, index) => (index < 60 ? 1 : 0));
    const fit = fitLogistic(outcomes.map(() => [1]), outcomes);
    expect(fit.coefficients[0]).toBeCloseTo(Math.log(60 / 140), 8);
    expect(fit.converged).toBe(true);
  });

  it('improves the fit monotonically as terms are nested inside one another', () => {
    const data = referenceData();
    const outcomes = data.map((d) => d.score);
    const design = rawDesign(data);
    let previous = Number.NEGATIVE_INFINITY;
    for (const width of [1, 2, 3, 4]) {
      const fit = fitLogistic(
        design.map((row) => row.slice(0, width)),
        outcomes,
      );
      expect(fit.logLikelihood).toBeGreaterThanOrEqual(previous - 1e-9);
      previous = fit.logLikelihood;
    }
  });

  it('stays finite on perfectly separated data instead of returning NaN', () => {
    // Every low score wrong and every high score right: the likelihood has no
    // maximum, the coefficient runs off to infinity, and the honest report is
    // that it did not converge.
    const outcomes: number[] = [];
    const design: number[][] = [];
    for (let i = 0; i < 80; i += 1) {
      design.push([1, i]);
      outcomes.push(i < 40 ? 0 : 1);
    }
    const fit = fitLogistic(design, outcomes, { maxIterations: 30 });
    expect(fit.coefficients.every((value) => Number.isFinite(value))).toBe(true);
    expect(Number.isFinite(fit.logLikelihood)).toBe(true);
    expect(Math.abs(fit.coefficients[1] as number)).toBeGreaterThan(1);
  });

  it('rejects a malformed problem', () => {
    expect(() => fitLogistic([], [])).toThrow(/no observations/);
    expect(() => fitLogistic([[1], [1]], [1])).toThrow(/1 outcomes for 2 rows/);
    expect(() => fitLogistic([[], [], []], [0, 1, 0])).toThrow(/no columns/);
    expect(() => fitLogistic([[1, 2]], [1])).toThrow(/1 observations cannot determine 2/);
    expect(() => fitLogistic([[1], [1], [1]], [0, 2, 1])).toThrow(/outcome 1 is 2/);
  });
});

describe('nagelkerke', () => {
  it('is zero when the model fits no better than the intercept', () => {
    expect(nagelkerke(-100, -100, 200)).toBeCloseTo(0, 14);
  });

  it('approaches one as the model approaches a perfect fit', () => {
    expect(nagelkerke(-100, -1e-12, 200)).toBeCloseTo(1, 8);
  });

  it('rises with the fit and stays inside the unit interval', () => {
    let previous = -1;
    for (const logLikelihood of [-100, -80, -60, -40, -20, -5]) {
      const value = nagelkerke(-100, logLikelihood, 200);
      expect(value).toBeGreaterThan(previous);
      expect(value).toBeLessThanOrEqual(1);
      previous = value;
    }
  });

  it('rejects a non-positive sample size', () => {
    expect(() => nagelkerke(-10, -5, 0)).toThrow(/positive integer/);
  });
});

describe('logisticDif', () => {
  const sampleFor = (options: Parameters<typeof simulateDif>[0], item: number) => {
    const simulation = simulateDif(options);
    return matchedSample(simulation.matrix, simulation.groups, item);
  };

  it('finds nothing on an item nobody touched', () => {
    const result = logisticDif(sampleFor({ bank: BANK, seed: 11 }, 3).observations);
    expect(result.converged).toBe(true);
    expect(result.uniform.pValue).toBeGreaterThan(0.05);
    expect(result.nonUniform.pValue).toBeGreaterThan(0.05);
    expect(result.deltaRSquared).toBeLessThan(0.01);
    expect(result.classification).toBe('A');
  });

  it('finds nothing when the groups differ in ability alone', () => {
    const result = logisticDif(sampleFor({ bank: BANK, focalMean: -0.8, seed: 12 }, 9).observations);
    expect(result.combined.pValue).toBeGreaterThan(0.05);
    expect(result.classification).toBe('A');
  });

  it('attributes a difficulty shift to the group term and not the interaction', () => {
    const result = logisticDif(
      sampleFor({ bank: BANK, shifts: [{ item: 7, difficulty: 1 }], seed: 13 }, 7).observations,
    );
    expect(result.uniform.chiSquare).toBeGreaterThan(100);
    expect(result.uniform.pValue).toBeLessThan(1e-20);
    // The item is uniformly harder, so there is no interaction to find.
    expect(result.nonUniform.chiSquare).toBeLessThan(4);
    expect(result.nonUniform.pValue).toBeGreaterThan(0.05);
    expect(result.coefficients.group).toBeLessThan(0);
    expect(result.classification).toBe('C');
  });

  it('catches a crossing item through the interaction, which the pooled method cannot', () => {
    // This is the whole argument for the method. Mantel-Haenszel calls the item
    // moderate at best, because pooling averages away the advantage each group
    // gets on its own side of the crossing point. The interaction term is
    // measuring the crossing directly and is unambiguous about it.
    const sample = sampleFor({ bank: BANK, shifts: [{ item: 5, discrimination: 0.35 }], seed: 15 }, 5);
    const pooled = mantelHaenszel(stratify(sample.observations));
    const result = logisticDif(sample.observations);

    expect(pooled.classification).toBe('B');
    expect(Math.abs(pooled.delta)).toBeLessThan(1.5);

    expect(result.nonUniform.chiSquare).toBeGreaterThan(30);
    expect(result.nonUniform.pValue).toBeLessThan(1e-7);
    expect(result.classification).not.toBe('A');
  });

  it('grows the interaction statistic with the sample, as a real effect should', () => {
    const small = logisticDif(
      sampleFor(
        {
          bank: BANK,
          referenceCount: 600,
          focalCount: 600,
          shifts: [{ item: 5, discrimination: 0.3 }],
          seed: 21,
        },
        5,
      ).observations,
    );
    const large = logisticDif(
      sampleFor(
        {
          bank: BANK,
          referenceCount: 3000,
          focalCount: 3000,
          shifts: [{ item: 5, discrimination: 0.3 }],
          seed: 21,
        },
        5,
      ).observations,
    );
    expect(large.nonUniform.chiSquare).toBeGreaterThan(small.nonUniform.chiSquare);
    // The effect size does not, which is the point of reporting one.
    expect(Math.abs(large.deltaRSquared - small.deltaRSquared)).toBeLessThan(0.05);
  });

  it('splits the combined test into its two components', () => {
    const result = logisticDif(
      sampleFor({ bank: BANK, shifts: [{ item: 7, difficulty: 1 }], seed: 13 }, 7).observations,
    );
    expect(result.combined.degreesOfFreedom).toBe(2);
    expect(result.uniform.degreesOfFreedom).toBe(1);
    expect(result.nonUniform.degreesOfFreedom).toBe(1);
    // Successive nesting, so the two one-degree tests add to the combined one.
    expect(result.combined.chiSquare).toBeCloseTo(
      result.uniform.chiSquare + result.nonUniform.chiSquare,
      6,
    );
  });

  it('leaves the likelihood-ratio tests unchanged by standardising the score', () => {
    // Centring and scaling a predictor is an affine reparameterisation, so the
    // fitted probabilities — and therefore every likelihood — are identical.
    const sample = sampleFor({ bank: BANK, shifts: [{ item: 7, difficulty: 1 }], seed: 13 }, 7);
    const scaled = logisticDif(sample.observations, { standardize: true });
    const raw = logisticDif(sample.observations, { standardize: false });
    expect(raw.uniform.chiSquare).toBeCloseTo(scaled.uniform.chiSquare, 6);
    expect(raw.nonUniform.chiSquare).toBeCloseTo(scaled.nonUniform.chiSquare, 6);
    expect(raw.deltaRSquared).toBeCloseTo(scaled.deltaRSquared, 8);
    expect(raw.standardized).toBe(false);
    expect(scaled.standardized).toBe(true);
  });

  it('reports the sample it used and both effect sizes', () => {
    const sample = sampleFor({ bank: BANK, shifts: [{ item: 7, difficulty: 1 }], seed: 13 }, 7);
    const result = logisticDif(sample.observations);
    expect(result.people).toBe(sample.observations.length);
    expect(result.deltaRSquared).toBeGreaterThan(result.uniformRSquared - 1e-12);
    expect(result.uniformRSquared).toBeGreaterThan(0);
  });

  it('rejects a graded item, an empty sample and a missing group', () => {
    expect(() => logisticDif([])).toThrow(/no observations/);
    expect(() =>
      logisticDif([{ group: 'focal', match: 3, score: 2 }]),
    ).toThrow(/expects a dichotomous item/);
    expect(() =>
      logisticDif([
        { group: 'focal', match: 3, score: 1 },
        { group: 'focal', match: 1, score: 0 },
      ]),
    ).toThrow(/both groups must be present/);
  });
});

describe('the logistic classification', () => {
  it('needs both significance and an effect to flag', () => {
    expect(classifyLogistic(0.2, 0.5)).toBe('A');
    expect(classifyLogistic(1e-9, 0.01)).toBe('A');
    expect(classifyLogistic(1e-9, 0.05)).toBe('B');
    expect(classifyLogistic(1e-9, 0.2)).toBe('C');
  });

  it('takes the boundaries as belonging to the band above', () => {
    expect(classifyLogistic(1e-9, EFFECT_RULES.jodoinGierl.negligible)).toBe('B');
    expect(classifyLogistic(1e-9, EFFECT_RULES.jodoinGierl.large)).toBe('C');
  });

  it('is far more conservative under the wider published pair', () => {
    // The reason the narrower pair is the default. On an item made a full logit
    // harder for the focal group, with a p-value near 1e-34, the change in
    // pseudo R-squared is about 0.076 — which the wider pair calls negligible.
    const simulation = simulateDif({ bank: BANK, shifts: [{ item: 7, difficulty: 1 }], seed: 13 });
    const sample = matchedSample(simulation.matrix, simulation.groups, 7);
    const narrow = logisticDif(sample.observations);
    const wide = logisticDif(sample.observations, { rule: EFFECT_RULES.zumboThomas });

    expect(narrow.deltaRSquared).toBeCloseTo(wide.deltaRSquared, 12);
    expect(narrow.combined.pValue).toBeLessThan(1e-30);
    expect(narrow.classification).toBe('C');
    expect(wide.classification).toBe('A');
  });
});
