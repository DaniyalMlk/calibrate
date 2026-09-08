import { mean } from '../core/numeric.js';
import type { SessionOutcome, StudyResult } from './study.js';

/**
 * Measurement quality at one point on the ability scale.
 *
 * The interesting field is not `rmse` on its own but `rmse` next to
 * `meanStandardError`. The first is how wrong the test actually was; the second
 * is how wrong it claimed it might be. A bank that is thin at the top of the
 * scale produces a conditional standard error that stays flat while the RMSE
 * climbs — the test keeps promising a precision it is no longer delivering —
 * and no marginal summary will show that, because the tails are exactly where
 * the fewest examinees are.
 */
export interface ConditionalBin {
  /** Inclusive lower edge of the ability interval. */
  readonly lower: number;
  /** Exclusive upper edge, except in the last bin where it is inclusive. */
  readonly upper: number;
  /** Midpoint of the interval, for plotting. */
  readonly center: number;
  /** Examinees falling in this bin. */
  readonly count: number;
  /** Mean true ability of the examinees in the bin. */
  readonly meanTrueTheta: number;
  /** Mean signed error, `estimate - true`. */
  readonly bias: number;
  /** Root mean square error against the generating ability. */
  readonly rmse: number;
  /** Standard deviation of the errors — RMSE with the bias removed. */
  readonly errorSd: number;
  /** Mean of the standard errors the sessions reported. */
  readonly meanStandardError: number;
  /**
   * Reported precision divided by realised precision.
   *
   * One means the reported standard error is honest; below one means it is
   * optimistic, which is the failure mode that matters, because a pass/fail
   * decision taken at a reported precision the test does not have is a decision
   * taken on a coin flip nobody knows they are tossing.
   */
  readonly calibration: number;
  /** Fraction of examinees whose true ability fell inside the reported 95% interval. */
  readonly coverage: number;
  /** Mean number of items administered. */
  readonly meanLength: number;
}

export interface ConditionalOptions {
  /** Lower edge of the reported range. Default -3. */
  readonly lower?: number;
  /** Upper edge of the reported range. Default 3. */
  readonly upper?: number;
  /** Number of equal-width bins. Default 6. */
  readonly bins?: number;
}

/** The multiplier for a two-sided 95% normal interval. */
export const NORMAL_95 = 1.959963984540054;

function binIndex(theta: number, lower: number, width: number, bins: number): number {
  const raw = Math.floor((theta - lower) / width);
  // Examinees outside the reported range are folded into the edge bins rather
  // than dropped. Dropping them would quietly shrink the denominator of exactly
  // the bins where the bank is weakest, making a coverage gap look like good
  // measurement of a small group.
  if (raw < 0) return 0;
  if (raw >= bins) return bins - 1;
  return raw;
}

function summariseBin(
  outcomes: readonly SessionOutcome[],
  lower: number,
  upper: number,
): ConditionalBin {
  const center = (lower + upper) / 2;
  if (outcomes.length === 0) {
    return {
      lower,
      upper,
      center,
      count: 0,
      meanTrueTheta: Number.NaN,
      bias: Number.NaN,
      rmse: Number.NaN,
      errorSd: Number.NaN,
      meanStandardError: Number.NaN,
      calibration: Number.NaN,
      coverage: Number.NaN,
      meanLength: Number.NaN,
    };
  }

  const errors = outcomes.map((outcome) => outcome.estimate - outcome.trueTheta);
  const bias = mean(errors);
  const meanSquare = mean(errors.map((error) => error * error));
  const rmse = Math.sqrt(meanSquare);
  // RMSE^2 = bias^2 + variance, so the error standard deviation falls out
  // without a second pass. Clamped at zero against float cancellation when the
  // bias accounts for essentially all of the error.
  const errorSd = Math.sqrt(Math.max(meanSquare - bias * bias, 0));
  const meanStandardError = mean(outcomes.map((outcome) => outcome.standardError));
  const covered = outcomes.filter(
    (outcome) =>
      Math.abs(outcome.estimate - outcome.trueTheta) <= NORMAL_95 * outcome.standardError,
  ).length;

  return {
    lower,
    upper,
    center,
    count: outcomes.length,
    meanTrueTheta: mean(outcomes.map((outcome) => outcome.trueTheta)),
    bias,
    rmse,
    errorSd,
    meanStandardError,
    calibration: rmse === 0 ? Number.POSITIVE_INFINITY : meanStandardError / rmse,
    coverage: covered / outcomes.length,
    meanLength: mean(outcomes.map((outcome) => outcome.length)),
  };
}

/**
 * Bin a study's outcomes by true ability and report measurement quality in each
 * bin.
 *
 * Binning is on the *generating* ability, not the estimate. Conditioning on the
 * estimate instead is a classic way to make a test look unbiased that is not:
 * the examinees a test overestimates are precisely the ones it places in a
 * higher bin, so the error is smuggled into the conditioning variable and
 * disappears from the report.
 */
export function conditionalReport(
  result: StudyResult,
  options: ConditionalOptions = {},
): ConditionalBin[] {
  const lower = options.lower ?? -3;
  const upper = options.upper ?? 3;
  const bins = options.bins ?? 6;

  if (!(lower < upper)) {
    throw new RangeError(`conditionalReport: lower (${lower}) must be below upper (${upper})`);
  }
  if (!Number.isInteger(bins) || bins < 1) {
    throw new RangeError(`conditionalReport: bins must be a positive integer, received ${bins}`);
  }

  const width = (upper - lower) / bins;
  const buckets: SessionOutcome[][] = Array.from({ length: bins }, () => []);
  for (const outcome of result.outcomes) {
    (buckets[binIndex(outcome.trueTheta, lower, width, bins)] as SessionOutcome[]).push(outcome);
  }

  return buckets.map((bucket, index) =>
    summariseBin(bucket, lower + index * width, lower + (index + 1) * width),
  );
}
