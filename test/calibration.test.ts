import { describe, expect, it } from 'vitest';
import { createRng } from '../src/core/random.js';
import { linspace, mean } from '../src/core/numeric.js';
import { makeItem, onePL, twoPL, type Item } from '../src/models/item.js';
import { probabilityCorrect } from '../src/models/response.js';
import { itemFit, standardiseMeanSquare } from '../src/calibration/fit.js';
import { calibrate, toItems } from '../src/calibration/jmle.js';
import { MISSING, ResponseMatrix, type Cell } from '../src/calibration/matrix.js';
import { simulateMatrix } from '../src/simulation/respondent.js';

function correlation(xs: readonly number[], ys: readonly number[]): number {
  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    const dx = (xs[i] as number) - mx;
    const dy = (ys[i] as number) - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  return sxy / Math.sqrt(sxx * syy);
}

function meanAbsoluteError(xs: readonly number[], ys: readonly number[]): number {
  let acc = 0;
  for (let i = 0; i < xs.length; i += 1) acc += Math.abs((xs[i] as number) - (ys[i] as number));
  return acc / xs.length;
}

function abilities(count: number, seed: number): number[] {
  const rng = createRng(seed);
  return Array.from({ length: count }, () => rng.nextNormal());
}

/** True difficulties, evenly spread across the range a real bank would cover. */
const trueDifficulty = linspace(-2, 2, 40);
const raschBank: Item[] = trueDifficulty.map((b, index) => makeItem(`i-${index}`, onePL(b)));

/** Discriminations shuffled by a stride, so they do not correlate with difficulty. */
const trueDiscrimination = trueDifficulty.map((_, index) => 0.6 + (1.4 * ((index * 7) % 40)) / 39);
const twoPlBank: Item[] = trueDifficulty.map((b, index) =>
  makeItem(`i-${index}`, twoPL(trueDiscrimination[index] as number, b)),
);

/** Line up an estimate with the generating value, by item id. */
function aligned(
  ids: readonly string[],
  bank: readonly Item[],
  truth: readonly number[],
): number[] {
  return ids.map((id) => truth[bank.findIndex((item) => item.id === id)] as number);
}

describe('calibrate — Rasch recovery', () => {
  const matrix = simulateMatrix(raschBank, abilities(1000, 99), 4242);
  const result = calibrate(matrix, { model: 'rasch' });
  const ids = result.items.map((item) => item.id);
  const estimated = result.items.map((item) => item.difficulty);
  const truth = aligned(ids, raschBank, trueDifficulty);

  it('converges in a handful of iterations', () => {
    expect(result.converged).toBe(true);
    expect(result.iterations).toBeLessThan(30);
    expect(result.maxChange).toBeLessThan(1e-4);
  });

  it('recovers the generating difficulties', () => {
    expect(correlation(estimated, truth)).toBeGreaterThan(0.99);
    expect(meanAbsoluteError(estimated, truth)).toBeLessThan(0.1);
  });

  it('pins the scale by centring the difficulties at zero', () => {
    // Without this the estimates would drift along the additive indeterminacy
    // and the convergence test would never fire.
    expect(mean(result.items.map((item) => item.difficulty))).toBeCloseTo(0, 6);
  });

  it('recovers abilities that correlate with the truth', () => {
    const truthByPerson = abilities(1000, 99);
    const kept = result.personIds.map(
      (id) => truthByPerson[Number(id.slice(2))] as number,
    );
    expect(correlation(result.abilities, kept)).toBeGreaterThan(0.85);
  });

  it('reports a standard error for every difficulty and none for discrimination', () => {
    for (const item of result.items) {
      expect(item.difficultyStandardError).toBeGreaterThan(0);
      expect(item.difficultyStandardError).toBeLessThan(0.5);
      expect(item.discriminationStandardError).toBe(0);
      expect(item.discrimination).toBe(1);
    }
  });

  it('reports the proportion correct alongside each estimate', () => {
    for (const item of result.items) {
      expect(item.proportionCorrect).toBeGreaterThan(0);
      expect(item.proportionCorrect).toBeLessThan(1);
      expect(item.answered).toBe(result.personIds.length);
    }
  });

  it('orders difficulty estimates the way the generating values were ordered', () => {
    const easiest = result.items.reduce((a, b) => (b.difficulty < a.difficulty ? b : a));
    const hardest = result.items.reduce((a, b) => (b.difficulty > a.difficulty ? b : a));
    expect(easiest.proportionCorrect).toBeGreaterThan(hardest.proportionCorrect);
  });

  it('turns back into items the rest of the engine can use', () => {
    const items = toItems(result);
    expect(items).toHaveLength(result.items.length);
    expect(items[0]?.parameters.a).toBe(1);
    expect(probabilityCorrect(items[0]!.parameters, items[0]!.parameters.b)).toBeCloseTo(0.5, 12);
  });
});

describe('the joint estimation bias correction', () => {
  it('reduces the error of the difficulty estimates at two sample sizes', () => {
    // Joint estimates are inflated away from zero because the person parameters
    // grow with the sample. The (L - 1) / L correction should measurably help,
    // and it is checked rather than assumed.
    for (const [count, seed] of [
      [1000, 99],
      [200, 31],
    ] as const) {
      const matrix = simulateMatrix(raschBank, abilities(count, seed), seed + 1);
      const corrected = calibrate(matrix, { model: 'rasch', biasCorrection: true });
      const raw = calibrate(matrix, { model: 'rasch', biasCorrection: false });
      const ids = corrected.items.map((item) => item.id);
      const truth = aligned(ids, raschBank, trueDifficulty);

      const correctedError = meanAbsoluteError(
        corrected.items.map((item) => item.difficulty),
        truth,
      );
      const rawError = meanAbsoluteError(raw.items.map((item) => item.difficulty), truth);
      expect(correctedError).toBeLessThan(rawError);
    }
  });

  it('shrinks every difficulty towards zero by exactly (L - 1) / L', () => {
    const matrix = simulateMatrix(raschBank, abilities(300, 7), 8);
    const corrected = calibrate(matrix, { model: 'rasch', biasCorrection: true });
    const raw = calibrate(matrix, { model: 'rasch', biasCorrection: false });
    const factor = (raw.items.length - 1) / raw.items.length;
    for (const [index, item] of corrected.items.entries()) {
      expect(item.difficulty).toBeCloseTo((raw.items[index]?.difficulty as number) * factor, 10);
    }
  });

  it('is off by default under the 2PL, where the discriminations absorb it', () => {
    const matrix = simulateMatrix(twoPlBank, abilities(400, 5), 6);
    const byDefault = calibrate(matrix, { model: '2pl' });
    const raw = calibrate(matrix, { model: '2pl', biasCorrection: false });
    expect(byDefault.items.map((item) => item.difficulty)).toEqual(
      raw.items.map((item) => item.difficulty),
    );
  });
});

describe('calibrate — 2PL recovery', () => {
  const matrix = simulateMatrix(twoPlBank, abilities(1000, 99), 777);
  const result = calibrate(matrix, { model: '2pl', maxIterations: 400 });
  const ids = result.items.map((item) => item.id);
  const difficulties = result.items.map((item) => item.difficulty);
  const discriminations = result.items.map((item) => item.discrimination);

  it('converges', () => {
    expect(result.converged).toBe(true);
    expect(result.model).toBe('2pl');
  });

  it('recovers both parameters', () => {
    expect(correlation(difficulties, aligned(ids, twoPlBank, trueDifficulty))).toBeGreaterThan(0.99);
    expect(meanAbsoluteError(difficulties, aligned(ids, twoPlBank, trueDifficulty))).toBeLessThan(0.15);
    expect(correlation(discriminations, aligned(ids, twoPlBank, trueDiscrimination))).toBeGreaterThan(0.9);
    expect(meanAbsoluteError(discriminations, aligned(ids, twoPlBank, trueDiscrimination))).toBeLessThan(0.2);
  });

  it('pins the multiplicative indeterminacy as well as the additive one', () => {
    // Under the 2PL it is the ability distribution that fixes the scale — mean
    // zero, unit variance — rather than the mean difficulty, because both the
    // location and the unit have to be chosen and standardising the abilities
    // chooses both at once. The difficulties then land near zero as a
    // consequence rather than by construction.
    const spread = Math.sqrt(
      mean(result.abilities.map((theta) => (theta - mean(result.abilities)) ** 2)),
    );
    expect(mean(result.abilities)).toBeCloseTo(0, 10);
    expect(spread).toBeCloseTo(1, 10);
    expect(Math.abs(mean(difficulties))).toBeLessThan(0.1);
    expect(mean(discriminations)).toBeGreaterThan(0.7);
    expect(mean(discriminations)).toBeLessThan(1.8);
  });

  it('reports standard errors for both parameters', () => {
    for (const item of result.items) {
      expect(item.difficultyStandardError).toBeGreaterThan(0);
      expect(item.discriminationStandardError).toBeGreaterThan(0);
    }
  });
});

describe('calibrate — screening and options', () => {
  it('drops extreme respondents and reports them', () => {
    const rng = createRng(3);
    const rows: Cell[][] = abilities(300, 12).map((theta) =>
      raschBank.map(
        (item) => (rng.next() < probabilityCorrect(item.parameters, theta) ? 1 : 0) as Cell,
      ),
    );
    // Two respondents who tell the calibration nothing.
    rows.push(raschBank.map(() => 1 as Cell));
    rows.push(raschBank.map(() => 0 as Cell));
    const matrix = new ResponseMatrix({
      rows,
      itemIds: raschBank.map((item) => item.id),
    });

    const result = calibrate(matrix, { model: 'rasch' });
    expect(result.screening.excludedPersons.map((exclusion) => exclusion.reason)).toEqual([
      'perfect-score',
      'zero-score',
    ]);
    expect(result.personIds).toHaveLength(300);
    expect(result.converged).toBe(true);
  });

  it('calibrates from an incomplete matrix', () => {
    const rng = createRng(21);
    const full = simulateMatrix(raschBank, abilities(600, 4), 5).toArray();
    // Blank a fifth of the responses at random — a linked design, not a fault.
    const sparse = full.map((row) => row.map((cell) => (rng.next() < 0.2 ? MISSING : cell)));
    const matrix = new ResponseMatrix({ rows: sparse, itemIds: raschBank.map((i) => i.id) });

    const result = calibrate(matrix, { model: 'rasch' });
    const ids = result.items.map((item) => item.id);
    expect(result.converged).toBe(true);
    expect(
      correlation(result.items.map((item) => item.difficulty), aligned(ids, raschBank, trueDifficulty)),
    ).toBeGreaterThan(0.97);
    for (const item of result.items) expect(item.answered).toBeLessThan(result.personIds.length);
  });

  it('reports a failure to converge rather than pretending otherwise', () => {
    const matrix = simulateMatrix(raschBank, abilities(200, 13), 14);
    const result = calibrate(matrix, { model: 'rasch', maxIterations: 1 });
    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(1);
    expect(result.maxChange).toBeGreaterThan(1e-4);
  });

  it('rejects nonsense options', () => {
    const matrix = simulateMatrix(raschBank, abilities(50, 1), 2);
    expect(() => calibrate(matrix, { tolerance: 0 })).toThrow(RangeError);
    expect(() => calibrate(matrix, { maxIterations: 0 })).toThrow(RangeError);
    expect(() => calibrate(matrix, { discriminationBounds: [2, 1] })).toThrow(RangeError);
    expect(() => calibrate(matrix, { scaleBounds: [3, -3] })).toThrow(RangeError);
  });
});

describe('itemFit', () => {
  const rng = createRng(2024);
  const bank = linspace(-2, 2, 30).map((b, index) => makeItem(`i-${index}`, onePL(b)));
  const rows: Cell[][] = abilities(800, 2024).map((theta) =>
    bank.map((item, column) => {
      // i-7 is answered at random; i-15 is miskeyed, so its scoring is reversed.
      if (column === 7) return (rng.next() < 0.5 ? 1 : 0) as Cell;
      const scored = rng.next() < probabilityCorrect(item.parameters, theta) ? 1 : 0;
      return (column === 15 ? 1 - scored : scored) as Cell;
    }),
  );
  const matrix = new ResponseMatrix({ rows, itemIds: bank.map((item) => item.id) });
  const calibration = calibrate(matrix, { model: 'rasch' });
  const report = itemFit(matrix, calibration);

  function fitOf(id: string) {
    const fit = report.items.find((candidate) => candidate.id === id);
    if (fit === undefined) throw new Error(`no fit for ${id}`);
    return fit;
  }

  it('reports a mean square near one for items that fit the model', () => {
    const fitting = report.items.filter((fit) => fit.id !== 'i-7' && fit.id !== 'i-15');
    for (const fit of fitting) {
      expect(fit.infit).toBeGreaterThan(0.7);
      expect(fit.infit).toBeLessThan(1.3);
      expect(fit.outfit).toBeGreaterThan(0.7);
      expect(fit.outfit).toBeLessThan(1.3);
    }
  });

  it('flags the random item and the miskeyed one, and nothing else', () => {
    expect(report.misfitting.map((fit) => fit.id).sort()).toEqual(['i-15', 'i-7']);
  });

  it('puts the miskeyed item further out than the random one', () => {
    // Reversing the key does not just add noise, it inverts the relationship
    // between ability and success, which is a much larger departure.
    expect(fitOf('i-15').outfit).toBeGreaterThan(fitOf('i-7').outfit);
    expect(fitOf('i-15').infitT).toBeGreaterThan(fitOf('i-7').infitT);
  });

  it('reports standardised deviates that are unremarkable for fitting items', () => {
    for (const fit of report.items) {
      if (fit.id === 'i-7' || fit.id === 'i-15') {
        expect(fit.infitT).toBeGreaterThan(3);
        continue;
      }
      expect(Math.abs(fit.infitT)).toBeLessThan(4);
    }
  });

  it('counts the responses each statistic came from', () => {
    for (const fit of report.items) expect(fit.answered).toBeGreaterThan(0);
  });

  it('carries the acceptance range it used', () => {
    expect(report.range).toEqual([0.7, 1.3]);
    expect(itemFit(matrix, calibration, { range: [0.5, 1.5] }).misfitting.length).toBeLessThanOrEqual(
      report.misfitting.length,
    );
  });

  it('rejects an inverted acceptance range', () => {
    expect(() => itemFit(matrix, calibration, { range: [1.3, 0.7] })).toThrow(RangeError);
  });

  it('refuses a calibration that does not belong to the matrix', () => {
    const other = simulateMatrix(
      [makeItem('elsewhere', onePL(0)), makeItem('elsewhere-2', onePL(1))],
      abilities(20, 9),
      3,
    );
    expect(() => itemFit(other, calibration)).toThrow(/is not in the matrix/);
  });
});

describe('standardiseMeanSquare', () => {
  it('maps a mean square of one onto a deviate near zero', () => {
    expect(standardiseMeanSquare(1, 0.04)).toBeCloseTo(Math.sqrt(0.04) / 3, 12);
  });

  it('is increasing in the mean square', () => {
    const variance = 0.02;
    expect(standardiseMeanSquare(1.5, variance)).toBeGreaterThan(
      standardiseMeanSquare(1.0, variance),
    );
    expect(standardiseMeanSquare(0.5, variance)).toBeLessThan(
      standardiseMeanSquare(1.0, variance),
    );
  });

  it('makes the same mean square more extreme when the variance is smaller', () => {
    // The whole point of the transform: a mean square of 1.3 is unremarkable on
    // thirty responses and damning on three thousand.
    expect(standardiseMeanSquare(1.3, 0.001)).toBeGreaterThan(standardiseMeanSquare(1.3, 0.1));
  });

  it('reports nothing for a degenerate mean square or a zero variance', () => {
    expect(Number.isNaN(standardiseMeanSquare(0, 0.04))).toBe(true);
    expect(standardiseMeanSquare(1.2, 0)).toBe(0);
  });
});
