import { linspace, mean } from '../core/numeric.js';
import { createRng } from '../core/random.js';
import type { Item } from '../models/item.js';
import type { ScoredResponse } from '../models/response.js';
import type { SessionEstimator } from '../session/estimator.js';
import { NORMAL_95 } from './conditional.js';
import { simulateResponses } from './respondent.js';

export interface RecoveryOptions {
  /** A name for the estimator under test, reproduced in the report. */
  readonly name: string;
  /** The estimator under test. */
  readonly estimator: SessionEstimator;
  /** The fixed form every simulated examinee answers. */
  readonly items: readonly Item[];
  /** Ability points to evaluate. Default 13 points from -3 to 3. */
  readonly abilities?: readonly number[];
  /** Replications per ability point. Default 200. */
  readonly replications?: number;
  /** Base seed. Default 20260101. */
  readonly seed?: number;
}

/** Estimator behaviour at one ability point. */
export interface RecoveryPoint {
  /** The generating ability. */
  readonly trueTheta: number;
  /** Replications attempted at this point. */
  readonly replications: number;
  /** Replications that produced an interior estimate. */
  readonly finite: number;
  /**
   * Fraction of replications whose response pattern had no finite maximiser.
   *
   * All-correct and all-incorrect patterns, which a likelihood estimator cannot
   * place. At the extremes of the ability range this is most of them, and it is
   * the single most useful number on the row: a form that cannot distinguish
   * between "very able" and "off the top of the scale" is not measuring there,
   * however small its RMSE looks over the replications that did resolve.
   */
  readonly boundaryFraction: number;
  /** Mean of the interior estimates. */
  readonly meanEstimate: number;
  /** Mean signed error of the interior estimates. */
  readonly bias: number;
  /** Root mean square error of the interior estimates. */
  readonly rmse: number;
  /** Standard deviation of the errors. */
  readonly errorSd: number;
  /** Mean reported standard error. */
  readonly meanStandardError: number;
  /** Reported precision over realised precision. */
  readonly calibration: number;
  /** Fraction of interior estimates whose 95% interval contained the generating ability. */
  readonly coverage: number;
}

export interface RecoveryStudy {
  readonly estimator: string;
  readonly formLength: number;
  readonly replications: number;
  readonly points: readonly RecoveryPoint[];
  /** Mean absolute bias across the evaluated points. */
  readonly meanAbsoluteBias: number;
  /** Root mean square error pooled across the evaluated points. */
  readonly pooledRmse: number;
}

const EMPTY_POINT = (trueTheta: number, replications: number): RecoveryPoint => ({
  trueTheta,
  replications,
  finite: 0,
  boundaryFraction: 1,
  meanEstimate: Number.NaN,
  bias: Number.NaN,
  rmse: Number.NaN,
  errorSd: Number.NaN,
  meanStandardError: Number.NaN,
  calibration: Number.NaN,
  coverage: Number.NaN,
});

/**
 * Repeatedly simulate a fixed form at a set of known abilities and measure how
 * well an estimator recovers them.
 *
 * This is the calibration check for the whole estimation layer, and the reason
 * it is a library function rather than a script is that its results are
 * assertions: maximum likelihood should be biased outward in the tails, EAP
 * should be shrunk toward the prior mean, and Warm's estimator should sit
 * between them. Those are properties of the estimators, not of any particular
 * run, so they belong in the test suite where a regression will trip them.
 *
 * Boundary patterns are excluded from the summary statistics rather than folded
 * in at the edge of the search window. A pattern with no finite maximiser has
 * not produced an estimate at all; averaging in the window edge would make an
 * estimator's apparent bias depend on how wide its window happens to be, which
 * is a property of the configuration rather than of the estimator. The fraction
 * excluded is reported instead, which is the honest form of the same
 * information.
 */
export function recoveryStudy(options: RecoveryOptions): RecoveryStudy {
  const { name, estimator, items } = options;
  if (items.length === 0) throw new RangeError('recoveryStudy: the form must have at least one item');
  const abilities = options.abilities ?? linspace(-3, 3, 13);
  if (abilities.length === 0) {
    throw new RangeError('recoveryStudy: at least one ability point is required');
  }
  const replications = options.replications ?? 200;
  if (!Number.isInteger(replications) || replications < 1) {
    throw new RangeError(
      `recoveryStudy: replications must be a positive integer, received ${replications}`,
    );
  }
  const seed = options.seed ?? 20260101;

  const points: RecoveryPoint[] = abilities.map((trueTheta, index) => {
    // Each ability point gets its own generator so a point's numbers do not
    // depend on how many points precede it in the list.
    const rng = createRng(seed + index * 104729);
    const errors: number[] = [];
    const estimates: number[] = [];
    const standardErrors: number[] = [];
    let covered = 0;
    let boundary = 0;

    for (let replication = 0; replication < replications; replication += 1) {
      const responses: ScoredResponse[] = simulateResponses(items, trueTheta, rng);
      const estimate = estimator(responses);
      if (estimate.boundary !== 'none' || !Number.isFinite(estimate.theta)) {
        boundary += 1;
        continue;
      }
      const error = estimate.theta - trueTheta;
      errors.push(error);
      estimates.push(estimate.theta);
      standardErrors.push(estimate.standardError);
      if (Number.isFinite(estimate.standardError) && Math.abs(error) <= NORMAL_95 * estimate.standardError) {
        covered += 1;
      }
    }

    if (errors.length === 0) return EMPTY_POINT(trueTheta, replications);

    const bias = mean(errors);
    const meanSquare = mean(errors.map((error) => error * error));
    const rmse = Math.sqrt(meanSquare);
    const meanStandardError = mean(standardErrors);

    return {
      trueTheta,
      replications,
      finite: errors.length,
      boundaryFraction: boundary / replications,
      meanEstimate: mean(estimates),
      bias,
      rmse,
      errorSd: Math.sqrt(Math.max(meanSquare - bias * bias, 0)),
      meanStandardError,
      calibration: rmse === 0 ? Number.POSITIVE_INFINITY : meanStandardError / rmse,
      coverage: covered / errors.length,
    };
  });

  const usable = points.filter((point) => point.finite > 0);
  const totalFinite = usable.reduce((acc, point) => acc + point.finite, 0);

  return {
    estimator: name,
    formLength: items.length,
    replications,
    points,
    meanAbsoluteBias:
      usable.length === 0 ? Number.NaN : mean(usable.map((point) => Math.abs(point.bias))),
    pooledRmse:
      totalFinite === 0
        ? Number.NaN
        : Math.sqrt(
            usable.reduce((acc, point) => acc + point.rmse * point.rmse * point.finite, 0) /
              totalFinite,
          ),
  };
}

function padStart(text: string, width: number): string {
  return text.length >= width ? text : ' '.repeat(width - text.length) + text;
}

function fixed(value: number, digits: number): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '-';
}

/** A fixed-width view of a recovery study. */
export function recoveryTable(study: RecoveryStudy): string {
  const header =
    padStart('theta', 8) +
    padStart('n', 6) +
    padStart('mean', 9) +
    padStart('bias', 8) +
    padStart('rmse', 8) +
    padStart('se', 8) +
    padStart('calib', 8) +
    padStart('cover', 8) +
    padStart('bound', 8) +
    '\n';

  let out = header + '-'.repeat(header.length - 1) + '\n';
  for (const point of study.points) {
    out +=
      padStart(fixed(point.trueTheta, 2), 8) +
      padStart(String(point.finite), 6) +
      padStart(fixed(point.meanEstimate, 3), 9) +
      padStart(fixed(point.bias, 3), 8) +
      padStart(fixed(point.rmse, 3), 8) +
      padStart(fixed(point.meanStandardError, 3), 8) +
      padStart(fixed(point.calibration, 2), 8) +
      padStart(fixed(point.coverage, 3), 8) +
      padStart(fixed(point.boundaryFraction, 3), 8) +
      '\n';
  }
  return out;
}
