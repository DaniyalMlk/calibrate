import { describe, expect, it } from 'vitest';
import { correlation, linspace, mean } from '../src/core/numeric.js';
import { createRng } from '../src/core/random.js';
import { integrate, normalGaussHermiteRule, normalGridRule } from '../src/core/quadrature.js';
import { makeItem, onePL, twoPL, type Item } from '../src/models/item.js';
import { probabilityCorrect } from '../src/models/response.js';
import { simulateMatrix } from '../src/simulation/respondent.js';
import { MISSING, ResponseMatrix, type Cell } from '../src/calibration/matrix.js';
import { calibrate } from '../src/calibration/jmle.js';
import { marginalCalibrate, toMarginalItems } from '../src/calibration/mml.js';
import { pointProbability } from '../src/calibration/expected.js';

const rule = normalGaussHermiteRule(0, 1, 41);

const trueDifficulty = linspace(-2, 2, 30);
const raschBank: Item[] = trueDifficulty.map((b, index) => makeItem(`i-${index}`, onePL(b)));
const trueDiscrimination = trueDifficulty.map((_, index) => 0.6 + (1.4 * ((index * 7) % 30)) / 29);
const twoPlBank: Item[] = trueDifficulty.map((b, index) =>
  makeItem(`i-${index}`, twoPL(trueDiscrimination[index] as number, b)),
);

function normalAbilities(count: number, seed: number, centre = 0, spread = 1): number[] {
  const rng = createRng(seed);
  return Array.from({ length: count }, () => centre + spread * rng.nextNormal());
}

function aligned(ids: readonly string[], bank: readonly Item[], truth: readonly number[]): number[] {
  return ids.map((id) => truth[bank.findIndex((item) => item.id === id)] as number);
}

function meanSignedError(estimates: readonly number[], truth: readonly number[]): number {
  let acc = 0;
  for (const [index, value] of estimates.entries()) acc += value - (truth[index] as number);
  return acc / estimates.length;
}

function meanAbsoluteError(estimates: readonly number[], truth: readonly number[]): number {
  let acc = 0;
  for (const [index, value] of estimates.entries()) {
    acc += Math.abs(value - (truth[index] as number));
  }
  return acc / estimates.length;
}

describe('marginalCalibrate: the EM guarantee', () => {
  it('never decreases the marginal log-likelihood, over seeds and both models', () => {
    for (const seed of [11, 202, 4703]) {
      for (const model of ['rasch', '2pl'] as const) {
        const bank = model === 'rasch' ? raschBank : twoPlBank;
        const matrix = simulateMatrix(bank, normalAbilities(400, seed), seed + 1);
        const result = marginalCalibrate(matrix, { model, rule });
        expect(result.history.length).toBeGreaterThan(2);
        for (let i = 1; i < result.history.length; i += 1) {
          // Exactly non-decreasing, not "roughly". A decrease here means the
          // expectation and maximisation steps are optimising different things.
          expect(result.history[i] as number).toBeGreaterThanOrEqual(
            (result.history[i - 1] as number) - 1e-9,
          );
        }
        expect(result.logLikelihood).toBe(
          result.history[result.history.length - 1] as number,
        );
      }
    }
  });

  it('improves on the fit at its own starting values', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(400, 77), 78);
    const result = marginalCalibrate(matrix, { rule });
    expect(result.logLikelihood).toBeGreaterThan(result.history[0] as number);
  });

  it('converges and reports that it did', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(500, 909), 910);
    const result = marginalCalibrate(matrix, { rule });
    expect(result.converged).toBe(true);
    expect(result.iterations).toBeLessThan(500);
    for (const item of result.items) expect(item.stationary).toBe(true);
  });

  it('stops at the iteration ceiling without claiming convergence', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(200, 31), 32);
    const result = marginalCalibrate(matrix, { rule, maxIterations: 2, tolerance: 1e-14 });
    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(2);
  });
});

describe('marginalCalibrate: the closed-form case', () => {
  /**
   * The population-averaged probability of a correct answer at difficulty `b`,
   * on the same rule the estimator uses so the two are exactly comparable.
   */
  function averageProbability(b: number): number {
    return integrate(rule, (theta) => pointProbability({ discrimination: 1, difficulty: b }, theta));
  }

  /** Bisection for the `b` at which the population-average equals `target`. */
  function solve(target: number): number {
    let lo = -6;
    let hi = 6;
    // The average is strictly decreasing in b, so the bracket is unambiguous.
    for (let step = 0; step < 200; step += 1) {
      const mid = (lo + hi) / 2;
      if (averageProbability(mid) > target) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  }

  it('puts a one-item test exactly where the marginal likelihood is stationary', () => {
    // With a single item the marginal likelihood depends on the data only
    // through the proportion correct, and is maximised where the
    // population-averaged probability of a correct answer equals it. That is a
    // scalar equation with one root, solvable independently by bisection, and
    // the estimator has to land on it.
    for (const [correct, total] of [
      [30, 100],
      [50, 100],
      [64, 100],
      [91, 120],
    ]) {
      const rows: Cell[][] = Array.from({ length: total as number }, (_, index) => [
        index < (correct as number) ? 1 : 0,
      ]);
      const result = marginalCalibrate(new ResponseMatrix({ rows }), {
        rule,
        tolerance: 1e-13,
        maxIterations: 3000,
      });
      const expected = solve((correct as number) / (total as number));
      expect(result.items[0]?.difficulty).toBeCloseTo(expected, 5);
    }
  });

  it('puts an item answered correctly by exactly half the population at zero', () => {
    // A corollary worth asserting on its own: the standard normal is symmetric
    // and the logistic is antisymmetric about its own location, so the
    // population-averaged probability at b = 0 is exactly one half.
    expect(averageProbability(0)).toBeCloseTo(0.5, 12);
    const rows: Cell[][] = Array.from({ length: 200 }, (_, index) => [index < 100 ? 1 : 0]);
    const result = marginalCalibrate(new ResponseMatrix({ rows }), {
      rule,
      tolerance: 1e-13,
      maxIterations: 3000,
    });
    expect(result.items[0]?.difficulty).toBeCloseTo(0, 6);
  });

  it('reproduces the observed proportion correct at the estimate', () => {
    const rows: Cell[][] = Array.from({ length: 250 }, (_, index) => [index < 88 ? 1 : 0]);
    const result = marginalCalibrate(new ResponseMatrix({ rows }), {
      rule,
      tolerance: 1e-13,
      maxIterations: 3000,
    });
    expect(averageProbability(result.items[0]?.difficulty as number)).toBeCloseTo(88 / 250, 6);
  });
});

describe('marginalCalibrate: recovery', () => {
  it('recovers Rasch difficulties across the range', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(3000, 5150), 5151);
    const result = marginalCalibrate(matrix, { rule });
    const estimates = result.items.map((item) => item.difficulty);
    const truth = aligned(
      result.items.map((item) => item.id),
      raschBank,
      trueDifficulty,
    );
    expect(correlation(estimates, truth)).toBeGreaterThan(0.99);
    expect(meanAbsoluteError(estimates, truth)).toBeLessThan(0.1);
  });

  it('recovers 2PL discriminations as well as difficulties', () => {
    const matrix = simulateMatrix(twoPlBank, normalAbilities(4000, 6270), 6271);
    const result = marginalCalibrate(matrix, { model: '2pl', rule });
    const ids = result.items.map((item) => item.id);
    expect(
      correlation(
        result.items.map((item) => item.difficulty),
        aligned(ids, twoPlBank, trueDifficulty),
      ),
    ).toBeGreaterThan(0.99);
    expect(
      correlation(
        result.items.map((item) => item.discrimination),
        aligned(ids, twoPlBank, trueDiscrimination),
      ),
    ).toBeGreaterThan(0.9);
  });

  it('does not inflate the difficulties the way uncorrected joint estimation does', () => {
    // The joint bias goes as L / (L - 1), so it is worst on a short test. Ten
    // items should show it clearly; the marginal estimator has no such term.
    const shortBank = raschBank.slice(0, 10);
    const shortTruth = trueDifficulty.slice(0, 10);
    let jointSpread = 0;
    let marginalSpread = 0;
    const seeds = [101, 202, 303, 404, 505];

    for (const seed of seeds) {
      const matrix = simulateMatrix(shortBank, normalAbilities(2000, seed), seed + 7);
      const joint = calibrate(matrix, { model: 'rasch', biasCorrection: false });
      const marginal = marginalCalibrate(matrix, { rule });

      const jointTruth = aligned(
        joint.items.map((item) => item.id),
        shortBank,
        shortTruth,
      );
      const marginalTruth = aligned(
        marginal.items.map((item) => item.id),
        shortBank,
        shortTruth,
      );
      // The bias is a stretch of the scale rather than a shift, so it shows up
      // as inflated spread rather than as a mean offset.
      jointSpread += spreadRatio(joint.items.map((item) => item.difficulty), jointTruth);
      marginalSpread += spreadRatio(marginal.items.map((item) => item.difficulty), marginalTruth);
    }

    jointSpread /= seeds.length;
    marginalSpread /= seeds.length;
    expect(jointSpread).toBeGreaterThan(1.05);
    expect(marginalSpread).toBeLessThan(jointSpread);
    expect(Math.abs(marginalSpread - 1)).toBeLessThan(Math.abs(jointSpread - 1));
  });

  it('is centred, with no systematic offset in the difficulties', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(3000, 8181), 8182);
    const result = marginalCalibrate(matrix, { rule });
    const truth = aligned(
      result.items.map((item) => item.id),
      raschBank,
      trueDifficulty,
    );
    expect(
      Math.abs(meanSignedError(result.items.map((item) => item.difficulty), truth)),
    ).toBeLessThan(0.06);
  });
});

/** Standard deviation of the estimates divided by that of the truth. */
function spreadRatio(estimates: readonly number[], truth: readonly number[]): number {
  const sd = (values: readonly number[]): number => {
    const centre = mean(values);
    let acc = 0;
    for (const value of values) acc += (value - centre) * (value - centre);
    return Math.sqrt(acc / values.length);
  };
  return sd(estimates) / sd(truth);
}

describe('marginalCalibrate: the data it keeps', () => {
  it('keeps perfect and zero scorers, and uses them', () => {
    const bank = raschBank.slice(0, 8);
    const abilities = normalAbilities(600, 4242);
    const matrix = simulateMatrix(bank, abilities, 4243);

    const extremes = matrix
      .toArray()
      .filter((row) => row.every((cell) => cell === 1) || row.every((cell) => cell === 0)).length;
    expect(extremes).toBeGreaterThan(0);

    const result = marginalCalibrate(matrix, { rule });
    expect(result.screening.excludedPersons).toHaveLength(0);

    // Joint estimation has to throw those respondents away; marginal estimation
    // does not, and the two therefore see different data.
    const joint = calibrate(matrix, { model: 'rasch' });
    expect(joint.screening.excludedPersons.length).toBe(extremes);
  });

  it('still removes an item nobody answered correctly', () => {
    const rows: Cell[][] = [
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
      [1, 0, 0],
    ];
    const result = marginalCalibrate(new ResponseMatrix({ rows }), { rule });
    expect(result.items).toHaveLength(2);
    expect(result.screening.excludedItems).toHaveLength(1);
    expect(result.screening.excludedItems[0]?.reason).toBe('answered-by-none');
  });

  it('handles missing responses without distorting the estimates', () => {
    const abilities = normalAbilities(1500, 1212);
    const complete = simulateMatrix(raschBank.slice(0, 12), abilities, 1213);
    const rows = complete.toArray();
    // Blank one response in twelve, at a position that does not depend on the
    // response itself, so the data stay missing at random.
    for (const [person, row] of rows.entries()) {
      row[(person * 5) % row.length] = MISSING;
    }
    const punched = new ResponseMatrix({ rows, itemIds: complete.itemIds });

    const full = marginalCalibrate(complete, { rule });
    const partial = marginalCalibrate(punched, { rule });
    let worst = 0;
    for (const [index, item] of partial.items.entries()) {
      worst = Math.max(worst, Math.abs(item.difficulty - (full.items[index]?.difficulty as number)));
      expect(item.answered).toBeLessThan(full.items[index]?.answered as number);
    }
    // A twelfth of the responses gone moves the hardest item by rather more
    // than the easiest, so the check is on the worst case rather than a mean.
    expect(worst).toBeLessThan(0.15);
    expect(
      correlation(
        partial.items.map((item) => item.difficulty),
        full.items.map((item) => item.difficulty),
      ),
    ).toBeGreaterThan(0.995);
  });

  it('does not depend on the order the respondents are stored in', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(600, 3131), 3132);
    const reversed = new ResponseMatrix({
      rows: matrix.toArray().reverse(),
      itemIds: matrix.itemIds,
    });
    const forward = marginalCalibrate(matrix, { rule });
    const backward = marginalCalibrate(reversed, { rule });
    expect(backward.logLikelihood).toBeCloseTo(forward.logLikelihood, 6);
    for (const [index, item] of forward.items.entries()) {
      expect(backward.items[index]?.difficulty).toBeCloseTo(item.difficulty, 6);
    }
  });
});

describe('marginalCalibrate: the population', () => {
  it('leaves the population alone when told to', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(800, 606), 607);
    const result = marginalCalibrate(matrix, { rule });
    for (const [k, weight] of rule.weights.entries()) {
      expect(result.population.weights[k] as number).toBeCloseTo(weight, 14);
    }
    // A standard normal held fixed: no skew, and tails exactly a normal's.
    expect(result.populationSkewness).toBeCloseTo(0, 8);
    expect(result.populationExcessKurtosis).toBeCloseTo(0, 8);
  });

  it('sees the shape of a skewed population, which a normal cannot have', () => {
    // Three quarters of the respondents low, a quarter well above them: a
    // distribution with a long right tail. Its mean and variance are not
    // recoverable — they define the metric — but its asymmetry is.
    const rng = createRng(7373);
    const abilities = Array.from({ length: 4000 }, () =>
      rng.next() < 0.75 ? -0.5 + 0.45 * rng.nextNormal() : 1.6 + 0.6 * rng.nextNormal(),
    );
    const matrix = simulateMatrix(raschBank, abilities, 7374);
    const fitted = marginalCalibrate(matrix, { rule, latent: 'empirical' });
    const assumed = marginalCalibrate(matrix, { rule });
    expect(fitted.populationSkewness).toBeGreaterThan(0.3);
    expect(assumed.populationSkewness).toBeCloseTo(0, 8);
  });

  it('fits a skewed population better than the normal it is not', () => {
    // A mixture with most of its mass low and a long upper tail. A free
    // distribution has strictly more room than a normal one, so it must reach
    // at least as high a marginal log-likelihood on the same data.
    const rng = createRng(9182);
    const abilities = Array.from({ length: 2500 }, () =>
      rng.next() < 0.75 ? -0.6 + 0.5 * rng.nextNormal() : 1.5 + 0.8 * rng.nextNormal(),
    );
    const matrix = simulateMatrix(raschBank, abilities, 9183);
    const normal = marginalCalibrate(matrix, { rule });
    const empirical = marginalCalibrate(matrix, { rule, latent: 'empirical' });
    expect(empirical.logLikelihood).toBeGreaterThan(normal.logLikelihood);
  });

  it('keeps the fitted population a probability distribution', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(1200, 5555, 0.4, 1.2), 5556);
    const result = marginalCalibrate(matrix, { rule, latent: 'empirical' });
    let total = 0;
    for (const weight of result.population.weights) {
      expect(weight).toBeGreaterThanOrEqual(0);
      total += weight;
    }
    expect(total).toBeCloseTo(1, 10);
    expect(result.population.nodes).toEqual(rule.nodes);
  });

  it('does not lose ground overall when the population is estimated too', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(1500, 2468, 0.3, 1.2), 2469);
    const result = marginalCalibrate(matrix, { rule, latent: 'empirical' });
    expect(result.logLikelihood).toBeGreaterThan(result.history[0] as number);
    // Re-standardising a distribution held on a fixed node set is exact only in
    // the continuum, so a cycle may give back a little of the interpolation
    // error. What it may not do is move the fit materially backwards.
    for (let i = 1; i < result.history.length; i += 1) {
      const drop = (result.history[i - 1] as number) - (result.history[i] as number);
      expect(drop).toBeLessThan(1e-3 * Math.abs(result.logLikelihood));
    }
  });
});

describe('marginalCalibrate: standard errors', () => {
  const matrix = simulateMatrix(raschBank, normalAbilities(2000, 1357), 1358);

  it('reports finite, positive errors for every item', () => {
    const result = marginalCalibrate(matrix, { rule });
    for (const item of result.items) {
      expect(item.difficultyStandardError).toBeGreaterThan(0);
      expect(Number.isFinite(item.difficultyStandardError)).toBe(true);
      expect(item.discriminationStandardError).toBe(0);
    }
  });

  it('shrinks as the sample grows, at roughly the square-root rate', () => {
    const small = marginalCalibrate(simulateMatrix(raschBank, normalAbilities(500, 24), 25), {
      rule,
    });
    const large = marginalCalibrate(simulateMatrix(raschBank, normalAbilities(2000, 24), 25), {
      rule,
    });
    const ratio =
      mean(large.items.map((item) => item.difficultyStandardError)) /
      mean(small.items.map((item) => item.difficultyStandardError));
    expect(ratio).toBeGreaterThan(0.4);
    expect(ratio).toBeLessThan(0.65);
  });

  it('gives wider intervals than the curvature of the expected counts does', () => {
    // The whole reason the cross-product form is the default: the curvature of
    // the expected-count likelihood ignores the uncertainty in the counts
    // themselves, so it is optimistic, and it should be visibly so.
    const crossProduct = marginalCalibrate(matrix, { rule, errors: 'cross-product' });
    const curvature = marginalCalibrate(matrix, { rule, errors: 'expected-counts' });
    const wider = crossProduct.items.filter(
      (item, index) =>
        item.difficultyStandardError > (curvature.items[index]?.difficultyStandardError as number),
    );
    expect(wider.length).toBeGreaterThan(crossProduct.items.length * 0.8);
    // Same parameters either way — only the reported uncertainty differs.
    for (const [index, item] of crossProduct.items.entries()) {
      expect(item.difficulty).toBeCloseTo(curvature.items[index]?.difficulty as number, 12);
    }
  });

  it('estimates errors that track the spread of repeated calibrations', () => {
    // The claim a standard error makes, checked directly: calibrate the same
    // bank on twenty fresh samples and the scatter of one item's estimates
    // should be near the error reported for it.
    const item = 14;
    const estimates: number[] = [];
    let reported = 0;
    for (let replicate = 0; replicate < 20; replicate += 1) {
      const seed = 40000 + replicate * 13;
      const result = marginalCalibrate(simulateMatrix(raschBank, normalAbilities(800, seed), seed + 1), {
        rule,
      });
      estimates.push(result.items[item]?.difficulty as number);
      reported += result.items[item]?.difficultyStandardError as number;
    }
    reported /= 20;
    const centre = mean(estimates);
    let acc = 0;
    for (const value of estimates) acc += (value - centre) * (value - centre);
    const observed = Math.sqrt(acc / (estimates.length - 1));
    expect(observed).toBeGreaterThan(reported * 0.5);
    expect(observed).toBeLessThan(reported * 2);
  });
});

describe('marginalCalibrate: interoperation and validation', () => {
  it('produces items the rest of the engine can use', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(800, 1919), 1920);
    const result = marginalCalibrate(matrix, { rule });
    const items = toMarginalItems(result);
    expect(items).toHaveLength(result.items.length);
    for (const [index, item] of items.entries()) {
      expect(item.id).toBe(result.items[index]?.id);
      const p = probabilityCorrect(item.parameters, 0);
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThan(1);
    }
  });

  it('agrees between a Gauss-Hermite rule and a dense grid', () => {
    const matrix = simulateMatrix(raschBank, normalAbilities(1500, 8642), 8643);
    const gauss = marginalCalibrate(matrix, { rule });
    const grid = marginalCalibrate(matrix, { rule: normalGridRule(0, 1, 81, 5) });
    for (const [index, item] of gauss.items.entries()) {
      expect(grid.items[index]?.difficulty).toBeCloseTo(item.difficulty, 2);
    }
  });

  it('rejects options that do not describe a problem', () => {
    const matrix = simulateMatrix(raschBank.slice(0, 5), normalAbilities(100, 1), 2);
    expect(() => marginalCalibrate(matrix, { tolerance: 0 })).toThrow(RangeError);
    expect(() => marginalCalibrate(matrix, { maxIterations: 0 })).toThrow(RangeError);
    expect(() => marginalCalibrate(matrix, { discriminationBounds: [2, 1] })).toThrow(RangeError);
    expect(() => marginalCalibrate(matrix, { difficultyBounds: [3, -3] })).toThrow(RangeError);
    expect(() =>
      marginalCalibrate(matrix, { rule: { nodes: [0, 1], weights: [1] } }),
    ).toThrow(RangeError);
  });

  it('refuses data in which no item carries information', () => {
    const rows: Cell[][] = [
      [1, 1],
      [1, 1],
    ];
    expect(() => marginalCalibrate(new ResponseMatrix({ rows }), { rule })).toThrow(RangeError);
  });
});
