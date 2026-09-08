import { clamp, logistic, mean } from '../core/numeric.js';
import { makeItem, onePL, twoPL, type Item } from '../models/item.js';
import { MISSING, type Cell } from './matrix.js';
import { screenExtremes, type ScreenResult } from './screen.js';
import type { ResponseMatrix } from './matrix.js';

/** Which model to calibrate. */
export type CalibrationModel = 'rasch' | '2pl';

export interface CalibrationOptions {
  /** Model to fit. Default `rasch`. */
  readonly model?: CalibrationModel;
  /** Largest absolute parameter change accepted as converged. Default 1e-4. */
  readonly tolerance?: number;
  /** Iteration ceiling for the outer loop. Default 200. */
  readonly maxIterations?: number;
  /**
   * Apply the `(L - 1) / L` correction to the estimated difficulties. Defaults
   * to true under the Rasch model and false under the 2PL; see the note on bias
   * below.
   */
  readonly biasCorrection?: boolean;
  /** Bounds on the estimated discrimination. Default 0.05 to 4. */
  readonly discriminationBounds?: readonly [number, number];
  /** Bounds on estimated difficulties and abilities. Default -6 to 6. */
  readonly scaleBounds?: readonly [number, number];
}

export interface CalibratedItem {
  readonly id: string;
  readonly discrimination: number;
  readonly difficulty: number;
  /** Standard error of the difficulty estimate, from the observed information. */
  readonly difficultyStandardError: number;
  /** Standard error of the discrimination estimate; zero under the Rasch model. */
  readonly discriminationStandardError: number;
  /** Persons who answered this item. */
  readonly answered: number;
  /** Proportion correct among those who answered. */
  readonly proportionCorrect: number;
}

export interface CalibrationResult {
  readonly model: CalibrationModel;
  readonly items: readonly CalibratedItem[];
  /** Ability estimates for the persons who survived screening. */
  readonly abilities: readonly number[];
  readonly personIds: readonly string[];
  readonly iterations: number;
  readonly converged: boolean;
  /** Largest parameter change on the final iteration. */
  readonly maxChange: number;
  readonly screening: ScreenResult;
}

/** Turn a calibration into items the rest of the engine can use. */
export function toItems(result: CalibrationResult): Item[] {
  return result.items.map((item) =>
    makeItem(
      item.id,
      result.model === 'rasch'
        ? onePL(item.difficulty)
        : twoPL(item.discrimination, item.difficulty),
    ),
  );
}

/**
 * Joint maximum likelihood calibration of item parameters and abilities.
 *
 * The Birnbaum paradigm: hold the item parameters fixed and estimate every
 * ability, hold the abilities fixed and estimate every item, repeat. Each half
 * is a well-behaved one- or two-parameter problem even though the joint problem
 * over thousands of parameters is not.
 *
 * **The scale has to be pinned on every cycle.** Adding a constant to every
 * ability and every difficulty leaves the likelihood exactly unchanged, and
 * under the 2PL so does multiplying the ability scale by a constant while
 * dividing the discriminations by it. Left alone, the estimates drift along
 * those directions forever and the convergence test never fires — not because
 * the fit is improving but because the parameters are sliding along a ridge. So
 * the difficulties are centred at zero after every item step, and under the 2PL
 * the ability scale is standardised as well.
 *
 * **The estimates are biased, and the bias is known.** Because the number of
 * person parameters grows with the sample, the usual consistency argument does
 * not apply: joint estimates of difficulty are inflated away from zero by a
 * factor of roughly `L / (L - 1)` for a test of `L` items. The classical
 * correction is to multiply the estimated difficulties by `(L - 1) / L`. It is a
 * real correction rather than a cosmetic one — on a simulated 40-item Rasch bank
 * it takes the mean absolute error of the difficulties from 0.068 to 0.060 at
 * 1,000 respondents and from 0.163 to 0.152 at 150, and the test suite asserts
 * that it helps rather than assuming it.
 *
 * It is a Rasch result, though, and it is applied by default only there. Under
 * the 2PL the discriminations absorb part of the same bias and the correction
 * measurably overshoots, so it is off unless asked for.
 */
export function calibrate(
  matrix: ResponseMatrix,
  options: CalibrationOptions = {},
): CalibrationResult {
  const model = options.model ?? 'rasch';
  const tolerance = options.tolerance ?? 1e-4;
  const maxIterations = options.maxIterations ?? 200;
  const biasCorrection = options.biasCorrection ?? model === 'rasch';
  const [minA, maxA] = options.discriminationBounds ?? [0.05, 4];
  const [minScale, maxScale] = options.scaleBounds ?? [-6, 6];

  if (!(tolerance > 0)) {
    throw new RangeError(`calibrate: tolerance must be positive, received ${tolerance}`);
  }
  if (!Number.isInteger(maxIterations) || maxIterations < 1) {
    throw new RangeError(
      `calibrate: maxIterations must be a positive integer, received ${maxIterations}`,
    );
  }
  if (!(minA > 0 && minA < maxA)) {
    throw new RangeError(`calibrate: discrimination bounds must satisfy 0 < min < max`);
  }
  if (!(minScale < maxScale)) {
    throw new RangeError('calibrate: scale bounds must satisfy min < max');
  }

  const screening = screenExtremes(matrix);
  const screened = screening.matrix;
  const personCount = screened.personCount;
  const itemCount = screened.itemCount;

  // Starting values: the logit of each item's proportion correct, negated,
  // which is the exact Rasch difficulty when every ability is zero. Starting
  // from zeros instead costs several iterations and, under the 2PL, sometimes a
  // step in the wrong direction on the first item pass.
  const difficulty: number[] = [];
  const discrimination: number[] = [];
  for (let item = 0; item < itemCount; item += 1) {
    const { correct, answered } = screened.itemScore(item);
    const p = clamp(correct / answered, 0.01, 0.99);
    difficulty.push(clamp(Math.log((1 - p) / p), minScale, maxScale));
    discrimination.push(1);
  }
  const ability: number[] = new Array<number>(personCount).fill(0);

  let iterations = 0;
  let maxChange = Number.POSITIVE_INFINITY;
  let converged = false;

  for (; iterations < maxIterations; iterations += 1) {
    let change = 0;

    for (let person = 0; person < personCount; person += 1) {
      const before = ability[person] as number;
      ability[person] = estimateAbility(
        screened.row(person),
        discrimination,
        difficulty,
        before,
        minScale,
        maxScale,
      );
      change = Math.max(change, Math.abs((ability[person] as number) - before));
    }

    for (let item = 0; item < itemCount; item += 1) {
      const column = screened.column(item);
      const beforeB = difficulty[item] as number;
      const beforeA = discrimination[item] as number;
      if (model === 'rasch') {
        difficulty[item] = estimateDifficulty(column, ability, beforeB, minScale, maxScale);
      } else {
        const step = estimateTwoParameter(column, ability, beforeA, beforeB);
        discrimination[item] = clamp(step.a, minA, maxA);
        difficulty[item] = clamp(step.b, minScale, maxScale);
      }
      change = Math.max(
        change,
        Math.abs((difficulty[item] as number) - beforeB),
        Math.abs((discrimination[item] as number) - beforeA),
      );
    }

    centreScale(ability, difficulty, discrimination, model, minScale, maxScale, minA, maxA);

    maxChange = change;
    if (change < tolerance) {
      converged = true;
      iterations += 1;
      break;
    }
  }

  const factor = biasCorrection && itemCount > 1 ? (itemCount - 1) / itemCount : 1;
  const items: CalibratedItem[] = [];
  for (let item = 0; item < itemCount; item += 1) {
    const column = screened.column(item);
    const { correct, answered } = screened.itemScore(item);
    const b = (difficulty[item] as number) * factor;
    const a = discrimination[item] as number;
    const errors = parameterStandardErrors(column, ability, a, b, model);
    items.push({
      id: screened.itemIds[item] as string,
      discrimination: a,
      difficulty: b,
      difficultyStandardError: errors.difficulty,
      discriminationStandardError: errors.discrimination,
      answered,
      proportionCorrect: correct / answered,
    });
  }

  return {
    model,
    items,
    abilities: ability,
    personIds: screened.personIds,
    iterations,
    converged,
    maxChange,
    screening,
  };
}

/** Newton steps on the ability of one person, with the item parameters fixed. */
function estimateAbility(
  row: readonly Cell[],
  discrimination: readonly number[],
  difficulty: readonly number[],
  start: number,
  min: number,
  max: number,
): number {
  let theta = start;
  for (let step = 0; step < 30; step += 1) {
    let score = 0;
    let information = 0;
    for (const [item, cell] of row.entries()) {
      if (cell === MISSING) continue;
      const a = discrimination[item] as number;
      const p = logistic(a * (theta - (difficulty[item] as number)));
      score += a * (cell - p);
      information += a * a * p * (1 - p);
    }
    // The Rasch and 2PL log-likelihoods are strictly concave in ability, so the
    // information is positive wherever any response is informative and Newton
    // cannot be thrown uphill the way it can under the 3PL.
    if (information < 1e-12) break;
    const delta = clamp(score / information, -1, 1);
    theta = clamp(theta + delta, min, max);
    if (Math.abs(delta) < 1e-8) break;
  }
  return theta;
}

/** Newton steps on the difficulty of one Rasch item, with the abilities fixed. */
function estimateDifficulty(
  column: readonly Cell[],
  ability: readonly number[],
  start: number,
  min: number,
  max: number,
): number {
  let b = start;
  for (let step = 0; step < 30; step += 1) {
    let score = 0;
    let information = 0;
    for (const [person, cell] of column.entries()) {
      if (cell === MISSING) continue;
      const p = logistic((ability[person] as number) - b);
      score -= cell - p;
      information += p * (1 - p);
    }
    if (information < 1e-12) break;
    const delta = clamp(score / information, -1, 1);
    b = clamp(b + delta, min, max);
    if (Math.abs(delta) < 1e-8) break;
  }
  return b;
}

/**
 * One damped Newton step on the discrimination and difficulty of a 2PL item.
 *
 * A single step per outer iteration rather than an inner loop to convergence.
 * Driving each item to its conditional optimum against abilities that are
 * themselves provisional wastes work and, worse, lets a badly scaled early
 * iteration push a discrimination to its bound and strand it there. The step is
 * damped and falls back to the gradient direction whenever the Hessian is not
 * negative definite, which it need not be far from the optimum.
 */
function estimateTwoParameter(
  column: readonly Cell[],
  ability: readonly number[],
  startA: number,
  startB: number,
): { a: number; b: number } {
  const a = startA;
  const b = startB;

  let gradA = 0;
  let gradB = 0;
  let hAA = 0;
  let hBB = 0;
  let hAB = 0;

  for (const [person, cell] of column.entries()) {
    if (cell === MISSING) continue;
    const theta = ability[person] as number;
    const d = theta - b;
    const p = logistic(a * d);
    const w = p * (1 - p);
    const residual = cell - p;

    gradA += residual * d;
    gradB -= a * residual;
    hAA -= w * d * d;
    hBB -= a * a * w;
    hAB += a * w * d - residual;
  }

  const determinant = hAA * hBB - hAB * hAB;
  // A negative-definite Hessian has a positive determinant and a negative
  // leading entry. Anything else means the quadratic model is not a maximum
  // here, and a Newton step would move away from one.
  if (determinant > 1e-12 && hAA < 0) {
    const stepA = (hBB * gradA - hAB * gradB) / determinant;
    const stepB = (hAA * gradB - hAB * gradA) / determinant;
    return { a: a - clamp(stepA, -0.5, 0.5), b: b - clamp(stepB, -0.5, 0.5) };
  }
  return { a: a + clamp(0.1 * gradA, -0.5, 0.5), b: b + clamp(0.1 * gradB, -0.5, 0.5) };
}

/**
 * Pin the scale.
 *
 * Under the Rasch model the mean difficulty is fixed at zero, which removes the
 * additive indeterminacy. Under the 2PL the abilities are standardised to mean
 * zero and unit variance and the item parameters transformed to match, which
 * removes the multiplicative one as well. Both transformations leave every
 * response probability exactly unchanged; they choose a representative from a
 * family of equivalent solutions rather than changing the fit.
 */
function centreScale(
  ability: number[],
  difficulty: number[],
  discrimination: number[],
  model: CalibrationModel,
  minScale: number,
  maxScale: number,
  minA: number,
  maxA: number,
): void {
  const shift = mean(difficulty);
  for (let i = 0; i < difficulty.length; i += 1) {
    difficulty[i] = clamp((difficulty[i] as number) - shift, minScale, maxScale);
  }
  for (let i = 0; i < ability.length; i += 1) {
    ability[i] = clamp((ability[i] as number) - shift, minScale, maxScale);
  }
  if (model === 'rasch') return;

  const centre = mean(ability);
  let acc = 0;
  for (const theta of ability) acc += (theta - centre) * (theta - centre);
  const spread = Math.sqrt(acc / ability.length);
  if (!(spread > 1e-8)) return;

  for (let i = 0; i < ability.length; i += 1) {
    ability[i] = clamp(((ability[i] as number) - centre) / spread, minScale, maxScale);
  }
  for (let i = 0; i < difficulty.length; i += 1) {
    difficulty[i] = clamp(((difficulty[i] as number) - centre) / spread, minScale, maxScale);
    discrimination[i] = clamp((discrimination[i] as number) * spread, minA, maxA);
  }
}

/**
 * Standard errors from the observed information of the item's own likelihood.
 *
 * Conditional on the ability estimates, which understates the true uncertainty:
 * the abilities were estimated from the same data and carry error of their own.
 * The number is still the right one to rank items by — an item answered by
 * thirty people is genuinely less well pinned down than one answered by three
 * thousand — but it should not be read as a calibrated interval.
 */
function parameterStandardErrors(
  column: readonly Cell[],
  ability: readonly number[],
  a: number,
  b: number,
  model: CalibrationModel,
): { difficulty: number; discrimination: number } {
  let hBB = 0;
  let hAA = 0;
  let hAB = 0;
  for (const [person, cell] of column.entries()) {
    if (cell === MISSING) continue;
    const theta = ability[person] as number;
    const d = theta - b;
    const p = logistic(a * d);
    const w = p * (1 - p);
    hBB += a * a * w;
    hAA += w * d * d;
    hAB += (cell - p) - a * w * d;
  }

  if (model === 'rasch') {
    return {
      difficulty: hBB > 0 ? 1 / Math.sqrt(hBB) : Number.POSITIVE_INFINITY,
      discrimination: 0,
    };
  }

  const determinant = hAA * hBB - hAB * hAB;
  if (!(determinant > 1e-12)) {
    return { difficulty: Number.POSITIVE_INFINITY, discrimination: Number.POSITIVE_INFINITY };
  }
  return {
    difficulty: Math.sqrt(hAA / determinant),
    discrimination: Math.sqrt(hBB / determinant),
  };
}
