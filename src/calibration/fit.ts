import { logistic } from '../core/numeric.js';
import { MISSING, type Cell } from './matrix.js';
import type { ResponseMatrix } from './matrix.js';
import type { CalibrationResult } from './jmle.js';

/** How well one item's observed responses match what the model predicted. */
export interface ItemFit {
  readonly id: string;
  /**
   * Outfit mean square: the mean squared standardised residual.
   *
   * Unweighted, so a single surprising response carries as much weight as any
   * other — which makes it sensitive to exactly the responses that are most
   * surprising and least representative. A very able candidate slipping on an
   * easy item can move it noticeably on its own.
   */
  readonly outfit: number;
  /**
   * Infit mean square: residuals weighted by their own variance.
   *
   * Weighting damps the responses of candidates far from the item's difficulty,
   * where a surprise is cheap information, and emphasises those near it, where
   * the item is actually doing its work. This is the statistic to read first.
   */
  readonly infit: number;
  /** Outfit as a standardised deviate — roughly standard normal under fit. */
  readonly outfitT: number;
  /** Infit as a standardised deviate. */
  readonly infitT: number;
  /** Responses the statistics were computed from. */
  readonly answered: number;
}

export interface FitReport {
  readonly items: readonly ItemFit[];
  /** Items whose infit or outfit falls outside the accepted range. */
  readonly misfitting: readonly ItemFit[];
  /** The range treated as acceptable. */
  readonly range: readonly [number, number];
}

export interface FitOptions {
  /**
   * Mean-square range treated as acceptable. Default 0.7 to 1.3.
   *
   * The convention rather than a derivation. Values above 1 mean the responses
   * are noisier than the model expects — the item is measuring something else,
   * or is miskeyed. Values below 1 mean they are *too* predictable, which sounds
   * harmless and is not: an item that tells you only what its neighbours already
   * told you is adding length rather than information.
   */
  readonly range?: readonly [number, number];
}

/**
 * The Wilson–Hilferty cube-root transformation of a mean-square to a
 * standardised deviate.
 *
 * A mean square is a ratio of chi-squares and is strongly right-skewed, so
 * comparing it to 1 does not say how surprising it is; the same value of 1.4
 * means something quite different on thirty responses and on three thousand.
 * The cube root of a chi-square ratio is close to normal, which is what makes
 * `t` comparable across items answered by different numbers of people.
 */
export function standardiseMeanSquare(meanSquare: number, variance: number): number {
  if (!(meanSquare > 0)) return Number.NaN;
  if (!(variance > 0)) return 0;
  const q = Math.sqrt(variance);
  return (Math.cbrt(meanSquare) - 1) * (3 / q) + q / 3;
}

/**
 * Infit and outfit mean squares for every calibrated item.
 *
 * Both are computed from the same standardised residuals; they differ only in
 * how they are weighted. For each response, with `p` the modelled probability
 * and `w = p(1 - p)` its variance:
 *
 * ```
 * outfit = mean of (x - p)^2 / w        infit = sum (x - p)^2 / sum w
 * ```
 *
 * The variances used for the standardised deviates come from the fourth central
 * moment `C = p(1-p)[(1-p)^3 + p^3]`, following Wright and Masters:
 * `Var(outfit) = sum(C/w^2 - 1) / n^2` and
 * `Var(infit) = sum(C - w^2) / (sum w)^2`.
 */
export function itemFit(
  matrix: ResponseMatrix,
  calibration: CalibrationResult,
  options: FitOptions = {},
): FitReport {
  const [low, high] = options.range ?? [0.7, 1.3];
  if (!(low > 0 && low < high)) {
    throw new RangeError(`itemFit: range must satisfy 0 < low < high, received [${low}, ${high}]`);
  }

  const columnOf = new Map<string, number>();
  for (const [index, id] of matrix.itemIds.entries()) columnOf.set(id, index);
  const rowOf = new Map<string, number>();
  for (const [index, id] of matrix.personIds.entries()) rowOf.set(id, index);

  const items: ItemFit[] = [];
  for (const item of calibration.items) {
    const column = columnOf.get(item.id);
    if (column === undefined) {
      throw new RangeError(`itemFit: calibrated item "${item.id}" is not in the matrix`);
    }

    let squared = 0;
    let weight = 0;
    let outfitSum = 0;
    let outfitVariance = 0;
    let infitVariance = 0;
    let answered = 0;

    for (const [index, personId] of calibration.personIds.entries()) {
      const row = rowOf.get(personId);
      if (row === undefined) {
        throw new RangeError(`itemFit: calibrated person "${personId}" is not in the matrix`);
      }
      const cell: Cell = matrix.at(row, column);
      if (cell === MISSING) continue;

      const theta = calibration.abilities[index] as number;
      const p = logistic(item.discrimination * (theta - item.difficulty));
      const w = p * (1 - p);
      // Numerically, w can underflow for a candidate far from the item. Such a
      // response carries no weight in the infit and an unbounded one in the
      // outfit, which is the outfit's known weakness rather than a defect here;
      // skipping it keeps the ratio finite without distorting either statistic.
      if (w < 1e-12) continue;

      const residual = cell - p;
      const c = w * ((1 - p) ** 3 + p ** 3);

      squared += residual * residual;
      weight += w;
      outfitSum += (residual * residual) / w;
      outfitVariance += c / (w * w) - 1;
      infitVariance += c - w * w;
      answered += 1;
    }

    if (answered === 0) {
      items.push({ id: item.id, outfit: Number.NaN, infit: Number.NaN, outfitT: Number.NaN, infitT: Number.NaN, answered: 0 });
      continue;
    }

    const outfit = outfitSum / answered;
    const infit = weight > 0 ? squared / weight : Number.NaN;

    items.push({
      id: item.id,
      outfit,
      infit,
      outfitT: standardiseMeanSquare(outfit, outfitVariance / (answered * answered)),
      infitT: standardiseMeanSquare(infit, infitVariance / (weight * weight)),
      answered,
    });
  }

  return {
    items,
    misfitting: items.filter(
      (fit) =>
        Number.isFinite(fit.infit) &&
        (fit.infit < low || fit.infit > high || fit.outfit < low || fit.outfit > high),
    ),
    range: [low, high],
  };
}
