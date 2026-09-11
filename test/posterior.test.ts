import { describe, expect, it } from 'vitest';

import {
  credibleInterval,
  estimateEap,
  estimateMap,
  graded,
  logLikelihood,
  makeItem,
  makePolytomousItem,
  normalPrior,
  posteriorCdf,
  posteriorDensity,
  posteriorMassBetween,
  rasch,
  twoPL,
  type ScoredResponse,
} from '../src/index.js';

/** The 0.975 standard normal quantile, to seven figures. */
const Z_975 = 1.959964;

/** Standard normal density, the closed form the empty posterior must reproduce. */
function standardNormalDensity(theta: number): number {
  return Math.exp(-0.5 * theta * theta) / Math.sqrt(2 * Math.PI);
}

function bank(difficulties: readonly number[], discrimination = 1.2) {
  return difficulties.map((b, i) => makeItem(`i-${i}`, twoPL(discrimination, b)));
}

function pattern(
  difficulties: readonly number[],
  scores: readonly (0 | 1)[],
  discrimination = 1.2,
): ScoredResponse[] {
  const items = bank(difficulties, discrimination);
  return items.map((item, i) => ({ item, response: scores[i] as 0 | 1 }));
}

describe('posteriorDensity', () => {
  it('reproduces the prior when there are no responses', () => {
    // Up to the normalisation, which divides by the trapezoid mass over a grid
    // truncated at six standard deviations rather than by one. That discrepancy
    // is around 2e-9 and is the only thing standing between this and equality.
    const posterior = posteriorDensity([]);

    for (const [i, theta] of posterior.grid.entries()) {
      expect(posterior.density[i] as number).toBeCloseTo(standardNormalDensity(theta), 8);
    }
  });

  it('recovers the moments of a standard normal prior', () => {
    const posterior = posteriorDensity([]);

    expect(posterior.mean).toBeCloseTo(0, 9);
    expect(posterior.sd).toBeCloseTo(1, 5);
    expect(posterior.mode).toBeCloseTo(0, 9);
  });

  it('recovers the moments of a shifted, stretched prior', () => {
    const posterior = posteriorDensity([], { prior: normalPrior(0.75, 1.4) });

    expect(posterior.mean).toBeCloseTo(0.75, 8);
    expect(posterior.sd).toBeCloseTo(1.4, 4);
    expect(posterior.mode).toBeCloseTo(0.75, 8);
  });

  it('integrates to one', () => {
    const responses = pattern([-1.5, -0.5, 0.5, 1.5], [1, 1, 0, 0]);
    const posterior = posteriorDensity(responses);

    let total = ((posterior.density[0] as number) + (posterior.density.at(-1) as number)) / 2;
    for (let i = 1; i < posterior.density.length - 1; i += 1) {
      total += posterior.density[i] as number;
    }

    expect(total * posterior.step).toBeCloseTo(1, 12);
  });

  it('agrees with the Gauss-Hermite EAP estimate on mean and standard deviation', () => {
    // Two independent quadratures over the same posterior: an even grid with the
    // trapezoid rule here, Gauss-Hermite nodes in estimateEap. Agreement is
    // evidence about the posterior, not about either rule.
    const cases: readonly (0 | 1)[][] = [
      [1, 1, 0, 0, 1],
      [0, 0, 0, 1, 0],
      [1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0],
      [1, 0, 1, 0, 1],
    ];
    const difficulties = [-2, -1, 0, 1, 2];

    for (const scores of cases) {
      const responses = pattern(difficulties, scores);
      const grid = posteriorDensity(responses, { points: 601 });
      const eap = estimateEap(responses);

      expect(grid.mean).toBeCloseTo(eap.theta, 5);
      expect(grid.sd).toBeCloseTo(eap.posteriorSd, 5);
    }
  });

  it('agrees with the MAP estimate on the mode', () => {
    const responses = pattern([-2, -1, 0, 1, 2], [1, 1, 0, 1, 0]);
    const grid = posteriorDensity(responses, { points: 1201 });
    const map = estimateMap(responses);

    expect(grid.mode).toBeCloseTo(map.theta, 4);
  });

  it('is defined for all-correct and all-incorrect patterns', () => {
    const allCorrect = posteriorDensity(pattern([-2, -1, 0, 1, 2], [1, 1, 1, 1, 1]));
    const allIncorrect = posteriorDensity(pattern([-2, -1, 0, 1, 2], [0, 0, 0, 0, 0]));

    expect(Number.isFinite(allCorrect.mean)).toBe(true);
    expect(Number.isFinite(allIncorrect.mean)).toBe(true);
    expect(allCorrect.mean).toBeGreaterThan(0);
    expect(allIncorrect.mean).toBeLessThan(0);
  });

  it('shifts the posterior towards the responses and away from the prior mean', () => {
    const correct = posteriorDensity(pattern([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]));
    const incorrect = posteriorDensity(pattern([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]));

    expect(correct.mean).toBeGreaterThan(0.5);
    expect(incorrect.mean).toBeLessThan(-0.5);
    // Symmetric items, mirrored responses, symmetric prior: mirrored posteriors.
    expect(correct.mean).toBeCloseTo(-incorrect.mean, 9);
  });

  it('concentrates as responses accumulate', () => {
    const difficulties = [0, 0.4, -0.4, 0.8, -0.8, 0.2, -0.2, 0.6];
    const scores: readonly (0 | 1)[] = [1, 0, 1, 0, 1, 1, 0, 1];

    let previous = posteriorDensity([]).sd;
    for (let n = 1; n <= difficulties.length; n += 1) {
      const responses = pattern(difficulties.slice(0, n), scores.slice(0, n));
      const sd = posteriorDensity(responses).sd;
      expect(sd).toBeLessThan(previous);
      previous = sd;
    }
  });

  it('does not quantise the mode to the grid', () => {
    // A coarse grid and a fine grid should place the mode in nearly the same
    // spot. Without interpolation the coarse answer would be pinned to a
    // multiple of its own spacing, which here is 0.24 wide.
    const responses = pattern([-2, -1, 0, 1, 2], [1, 1, 0, 1, 0]);
    const coarse = posteriorDensity(responses, { points: 51 });
    const fine = posteriorDensity(responses, { points: 4001 });

    expect(coarse.step).toBeGreaterThan(0.2);
    expect(coarse.mode).toBeCloseTo(fine.mode, 2);
  });

  it('reports an edge ratio near zero when the grid comfortably contains the posterior', () => {
    expect(posteriorDensity([]).edgeRatio).toBeLessThan(1e-7);
  });

  it('reports a large edge ratio when the grid truncates the posterior', () => {
    const truncated = posteriorDensity([], { lower: -0.5, upper: 0.5 });

    expect(truncated.edgeRatio).toBeGreaterThan(0.8);
    // And the moments are visibly biased inward, which is what the diagnostic warns about.
    expect(truncated.sd).toBeLessThan(0.35);
  });

  it('is unaffected by grid resolution once the grid is fine enough', () => {
    const responses = pattern([-1, 0, 1], [1, 0, 1]);
    const a = posteriorDensity(responses, { points: 401 });
    const b = posteriorDensity(responses, { points: 1601 });

    expect(a.mean).toBeCloseTo(b.mean, 6);
    expect(a.sd).toBeCloseTo(b.sd, 6);
  });

  it('produces a strictly increasing, evenly spaced grid that hits both bounds', () => {
    const posterior = posteriorDensity([], { lower: -3, upper: 4, points: 71 });

    expect(posterior.grid[0]).toBeCloseTo(-3, 12);
    expect(posterior.grid.at(-1)).toBeCloseTo(4, 12);
    expect(posterior.step).toBeCloseTo(7 / 70, 12);
    for (let i = 1; i < posterior.grid.length; i += 1) {
      expect((posterior.grid[i] as number) - (posterior.grid[i - 1] as number)).toBeCloseTo(
        posterior.step,
        10,
      );
    }
  });

  it('rejects a grid that is degenerate or too coarse to interpolate', () => {
    expect(() => posteriorDensity([], { lower: 1, upper: 1 })).toThrow(/upper bound must exceed/);
    expect(() => posteriorDensity([], { lower: 2, upper: -2 })).toThrow(/upper bound must exceed/);
    expect(() => posteriorDensity([], { points: 2 })).toThrow(/at least 3/);
    expect(() => posteriorDensity([], { points: 40.5 })).toThrow(/integer/);
    expect(() => posteriorDensity([], { halfWidth: 0 })).toThrow(/halfWidth must be positive/);
  });

  it('rejects a prior the constructor would reject', () => {
    expect(() => posteriorDensity([], { prior: { mean: 0, sd: -1 } })).toThrow(/must be positive/);
    expect(() => posteriorDensity([], { prior: { mean: Number.NaN, sd: 1 } })).toThrow();
  });
});

describe('posteriorCdf', () => {
  it('runs from zero to one and never decreases', () => {
    const responses = pattern([-1.5, -0.5, 0.5, 1.5], [1, 0, 1, 0]);
    const cdf = posteriorCdf(posteriorDensity(responses));

    expect(cdf[0]).toBe(0);
    expect(cdf.at(-1)).toBeCloseTo(1, 12);
    for (let i = 1; i < cdf.length; i += 1) {
      expect(cdf[i] as number).toBeGreaterThanOrEqual(cdf[i - 1] as number);
    }
  });

  it('matches the standard normal distribution function on the empty posterior', () => {
    const posterior = posteriorDensity([], { points: 2401 });
    const cdf = posteriorCdf(posterior);
    expect(cdf.at(-1)).toBeCloseTo(1, 12);

    // Interpolated between grid points, because the grid does not land on
    // 1.959964 and a nearest-point lookup would report the error in the
    // lookup rather than in the distribution function.
    const at = (theta: number): number =>
      posteriorMassBetween(posterior, posterior.grid[0] as number, theta);

    // Closed-form standard normal quantiles.
    expect(at(0)).toBeCloseTo(0.5, 6);
    expect(at(-1)).toBeCloseTo(0.158655, 5);
    expect(at(1)).toBeCloseTo(0.841345, 5);
    expect(at(-Z_975)).toBeCloseTo(0.025, 5);
    expect(at(Z_975)).toBeCloseTo(0.975, 5);
  });
});

describe('credibleInterval', () => {
  it('matches the closed-form normal interval on the empty posterior', () => {
    const posterior = posteriorDensity([], { points: 2401 });
    const interval = credibleInterval(posterior, 0.95);

    expect(interval.lower).toBeCloseTo(-Z_975, 3);
    expect(interval.upper).toBeCloseTo(Z_975, 3);
    expect(interval.width).toBeCloseTo(2 * Z_975, 3);
  });

  it('matches the closed-form normal interval under a shifted, stretched prior', () => {
    const prior = normalPrior(0.5, 1.25);
    const interval = credibleInterval(posteriorDensity([], { prior, points: 2401 }), 0.95);

    expect(interval.lower).toBeCloseTo(0.5 - 1.25 * Z_975, 3);
    expect(interval.upper).toBeCloseTo(0.5 + 1.25 * Z_975, 3);
  });

  it('matches closed-form quantiles at other masses', () => {
    const posterior = posteriorDensity([], { points: 2401 });

    // Standard normal quantiles for the 50%, 80% and 99% central intervals.
    expect(credibleInterval(posterior, 0.5).upper).toBeCloseTo(0.674490, 3);
    expect(credibleInterval(posterior, 0.8).upper).toBeCloseTo(1.281552, 3);
    expect(credibleInterval(posterior, 0.99).upper).toBeCloseTo(2.575829, 3);
  });

  it('holds the mass it claims to hold', () => {
    const responses = pattern([-1, -0.2, 0.6, 1.4, 2], [1, 1, 0, 1, 0]);
    const posterior = posteriorDensity(responses, { points: 1201 });

    for (const mass of [0.5, 0.8, 0.9, 0.95, 0.99]) {
      const interval = credibleInterval(posterior, mass);
      expect(posteriorMassBetween(posterior, interval.lower, interval.upper)).toBeCloseTo(mass, 6);
    }
  });

  it('nests: a wider mass gives a wider interval', () => {
    const responses = pattern([-1, 0, 1], [1, 0, 1]);
    const posterior = posteriorDensity(responses);

    const masses = [0.5, 0.8, 0.9, 0.95, 0.99];
    const intervals = masses.map((mass) => credibleInterval(posterior, mass));
    for (let i = 1; i < intervals.length; i += 1) {
      const inner = intervals[i - 1] as (typeof intervals)[number];
      const outer = intervals[i] as (typeof intervals)[number];
      expect(outer.lower).toBeLessThan(inner.lower);
      expect(outer.upper).toBeGreaterThan(inner.upper);
      expect(outer.width).toBeGreaterThan(inner.width);
    }
  });

  it('leaves equal mass in each tail', () => {
    // The defining property, and the one that separates this from a
    // highest-density interval on a skewed posterior.
    const responses = pattern([1, 1.5, 2, 2.5], [1, 1, 1, 1]);
    const posterior = posteriorDensity(responses, { points: 1601 });
    const interval = credibleInterval(posterior, 0.9);

    const below = posteriorMassBetween(posterior, posterior.grid[0] as number, interval.lower);
    const above = posteriorMassBetween(posterior, interval.upper, posterior.grid.at(-1) as number);

    expect(below).toBeCloseTo(0.05, 5);
    expect(above).toBeCloseTo(0.05, 5);
  });

  it('narrows as a test proceeds', () => {
    const difficulties = [0, 0.5, -0.5, 1, -1, 0.25, -0.25, 0.75];
    const scores: readonly (0 | 1)[] = [1, 0, 1, 0, 1, 1, 0, 1];

    let previous = credibleInterval(posteriorDensity([])).width;
    for (let n = 1; n <= difficulties.length; n += 1) {
      const responses = pattern(difficulties.slice(0, n), scores.slice(0, n));
      const width = credibleInterval(posteriorDensity(responses)).width;
      expect(width).toBeLessThan(previous);
      previous = width;
    }
  });

  it('brackets the posterior mean and mode', () => {
    const responses = pattern([-2, -1, 0, 1, 2], [1, 1, 0, 1, 0]);
    const posterior = posteriorDensity(responses);
    const interval = credibleInterval(posterior, 0.95);

    expect(posterior.mean).toBeGreaterThan(interval.lower);
    expect(posterior.mean).toBeLessThan(interval.upper);
    expect(posterior.mode).toBeGreaterThan(interval.lower);
    expect(posterior.mode).toBeLessThan(interval.upper);
  });

  it('rejects a mass outside the open unit interval', () => {
    const posterior = posteriorDensity([]);

    expect(() => credibleInterval(posterior, 0)).toThrow(/must lie in \(0, 1\)/);
    expect(() => credibleInterval(posterior, 1)).toThrow(/must lie in \(0, 1\)/);
    expect(() => credibleInterval(posterior, 1.5)).toThrow(/must lie in \(0, 1\)/);
    expect(() => credibleInterval(posterior, -0.2)).toThrow(/must lie in \(0, 1\)/);
    expect(() => credibleInterval(posterior, Number.NaN)).toThrow();
  });
});

describe('posteriorMassBetween', () => {
  it('gives zero width zero mass and the whole grid all of it', () => {
    const posterior = posteriorDensity(pattern([-1, 0, 1], [1, 0, 1]));

    expect(posteriorMassBetween(posterior, 0.3, 0.3)).toBeCloseTo(0, 12);
    expect(
      posteriorMassBetween(posterior, posterior.grid[0] as number, posterior.grid.at(-1) as number),
    ).toBeCloseTo(1, 12);
  });

  it('clamps to the grid rather than extrapolating', () => {
    const posterior = posteriorDensity([]);

    expect(posteriorMassBetween(posterior, -50, 50)).toBeCloseTo(1, 12);
    expect(posteriorMassBetween(posterior, 20, 50)).toBeCloseTo(0, 12);
  });

  it('splits the standard normal at its median', () => {
    const posterior = posteriorDensity([], { points: 2401 });
    const lower = posteriorMassBetween(posterior, posterior.grid[0] as number, 0);

    expect(lower).toBeCloseTo(0.5, 6);
  });

  it('answers a cut-score question the way a decision rule would', () => {
    // A candidate whose posterior sits above a cut of 0.5: the mass above the
    // cut is the confidence a pass decision would carry.
    const responses = pattern([-1, -0.5, 0, 0.5, 1, 1.5], [1, 1, 1, 1, 1, 0]);
    const posterior = posteriorDensity(responses, { points: 1601 });
    const above = posteriorMassBetween(posterior, 0.5, posterior.grid.at(-1) as number);
    const below = posteriorMassBetween(posterior, posterior.grid[0] as number, 0.5);

    expect(above + below).toBeCloseTo(1, 6);
    expect(above).toBeGreaterThan(below);
  });

  it('rejects an inverted interval', () => {
    expect(() => posteriorMassBetween(posteriorDensity([]), 1, -1)).toThrow(
      /must not be below lower/,
    );
  });
});

describe('posterior under a Rasch bank', () => {
  it('shrinks a single correct answer inside the prior-free closed form', () => {
    // One Rasch item answered correctly is the case with a closed-form
    // prior-free answer: Warm's estimator returns exactly `b + ln 3`, and
    // maximum likelihood returns infinity. A posterior under a standard normal
    // prior has to land above the prior mean, because a correct answer is
    // evidence upward, and below the prior-free estimate, because that is what
    // shrinkage means.
    const b = 0.4;
    const item = makeItem('single', rasch(b));
    const posterior = posteriorDensity([{ item, response: 1 }], { points: 1601 });

    expect(posterior.mean).toBeGreaterThan(0);
    expect(posterior.mean).toBeLessThan(b + Math.log(3));
    expect(credibleInterval(posterior).width).toBeLessThan(
      credibleInterval(posteriorDensity([])).width,
    );
  });

  it('mirrors a single incorrect answer about the item difficulty', () => {
    // Symmetry of the Rasch model: a correct answer to an item at `b` and an
    // incorrect answer to an item at `-b` are the same evidence reflected, so
    // under a prior centred at zero the two posterior means are negatives.
    const correct = posteriorDensity([{ item: makeItem('a', rasch(0.4)), response: 1 }], {
      points: 1601,
    });
    const incorrect = posteriorDensity([{ item: makeItem('b', rasch(-0.4)), response: 0 }], {
      points: 1601,
    });

    expect(correct.mean).toBeCloseTo(-incorrect.mean, 9);
    expect(correct.sd).toBeCloseTo(incorrect.sd, 9);
  });
});

describe('posteriorDensity over a mixed-format pattern', () => {
  const rubric = makePolytomousItem('cr-1', graded(1.3, [-0.8, 0.1, 0.9]));
  const dichotomous = makeItem('mc-1', twoPL(1.2, 0.2));

  it('is the normalised product of the mixed likelihood and the prior', () => {
    // The whole of the widening is that the likelihood already accepted both
    // formats. This pins that the density really is the function the engine
    // scores with, and not a dichotomous approximation of it.
    const responses = [
      { item: dichotomous, response: 1 as const },
      { item: rubric, category: 2 },
    ];
    const posterior = posteriorDensity(responses, { points: 401 });

    const unnormalised = posterior.grid.map(
      (theta) => Math.exp(logLikelihood(responses, theta)) * standardNormalDensity(theta),
    );
    let mass = 0;
    for (const [i, value] of unnormalised.entries()) {
      const edge = i === 0 || i === unnormalised.length - 1;
      mass += (edge ? 0.5 : 1) * value * posterior.step;
    }
    for (const [i, value] of unnormalised.entries()) {
      expect(posterior.density[i] as number).toBeCloseTo(value / mass, 9);
    }
  });

  it('treats a top-category rubric score as the evidence it is', () => {
    // The highest category on a three-threshold item is stronger evidence than
    // the middle one, which is stronger than the lowest. Nothing subtle — but
    // it is exactly what a posterior built on the dichotomous path got wrong,
    // by reading category 3 as "not 1" and therefore as a wrong answer.
    const means = [0, 1, 2, 3].map(
      (category) => posteriorDensity([{ item: rubric, category }], { points: 801 }).mean,
    );
    for (let k = 1; k < means.length; k += 1) {
      expect(means[k] as number).toBeGreaterThan(means[k - 1] as number);
    }
    expect(means[0] as number).toBeLessThan(0);
    expect(means[3] as number).toBeGreaterThan(0);
  });

  it('matches the dichotomous posterior for a two-category rubric', () => {
    // A graded item with one threshold is a 2PL item written in the other
    // notation, so the two posteriors must agree to numerical precision. If
    // they ever diverge, one of the two response models has drifted.
    const twoCategory = makePolytomousItem('cr-2', graded(1.1, [0.35]));
    const equivalent = makeItem('mc-2', twoPL(1.1, 0.35));

    const viaCategory = posteriorDensity([{ item: twoCategory, category: 1 }], { points: 401 });
    const viaResponse = posteriorDensity([{ item: equivalent, response: 1 }], { points: 401 });

    expect(viaCategory.mean).toBeCloseTo(viaResponse.mean, 10);
    expect(viaCategory.sd).toBeCloseTo(viaResponse.sd, 10);
  });
});
