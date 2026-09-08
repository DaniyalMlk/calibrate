import { correlation, mean } from '../core/numeric.js';
import {
  exposureChiSquare,
  exposureRatesFromIds,
  exposureVariance,
  overlapRate,
} from '../selection/exposure.js';
import { conditionalReport, NORMAL_95, type ConditionalBin, type ConditionalOptions } from './conditional.js';
import type { StudyResult } from './study.js';

/** How heavily a study used the bank it was given. */
export interface ExposureSummary {
  /** Items administered at least once. */
  readonly itemsUsed: number;
  /** Fraction of the bank never administered. */
  readonly unused: number;
  /** Highest exposure rate any single item reached. */
  readonly maxRate: number;
  /** Mean exposure rate across the whole bank — necessarily `meanLength / bankSize`. */
  readonly meanRate: number;
  /** Population variance of the exposure rates across the whole bank. */
  readonly rateVariance: number;
  /** Expected proportion of items shared by two randomly chosen examinees. */
  readonly overlap: number;
  /** Chi-square index of exposure skew. */
  readonly chiSquare: number;
  /** Number of items exposed above the target rate, if one was given. */
  readonly aboveTarget: number;
  /** The target the count above was taken against. */
  readonly target: number;
}

/** Everything a study says about one policy, in one object. */
export interface StudySummary {
  readonly policy: string;
  readonly population: string;
  readonly examinees: number;
  readonly poolSize: number;
  /** Mean signed error across the whole population. */
  readonly bias: number;
  /** Root mean square error across the whole population. */
  readonly rmse: number;
  /** Mean of the standard errors the sessions reported. */
  readonly meanStandardError: number;
  /** Reported precision over realised precision; one is honest, below one is optimistic. */
  readonly calibration: number;
  /** Fraction of examinees inside their reported 95% interval. */
  readonly coverage: number;
  /** Mean test length. */
  readonly meanLength: number;
  /** Longest and shortest test administered. */
  readonly lengthRange: readonly [number, number];
  /** Correlation between true and estimated ability. */
  readonly correlation: number;
  readonly exposure: ExposureSummary;
  readonly conditional: readonly ConditionalBin[];
}

export interface SummaryOptions extends ConditionalOptions {
  /** Exposure rate above which an item counts as over-exposed. Default 0.2. */
  readonly exposureTarget?: number;
}

/** Exposure statistics for a completed study. */
export function exposureSummary(result: StudyResult, target = 0.2): ExposureSummary {
  const rates = exposureRatesFromIds(result.outcomes.map((outcome) => outcome.itemIds));
  const meanLength = mean(result.outcomes.map((outcome) => outcome.length));
  const variance = exposureVariance(result.poolSize, rates);

  let maxRate = 0;
  let aboveTarget = 0;
  for (const rate of rates.values()) {
    if (rate > maxRate) maxRate = rate;
    if (rate > target) aboveTarget += 1;
  }

  return {
    itemsUsed: rates.size,
    unused: (result.poolSize - rates.size) / result.poolSize,
    maxRate,
    meanRate: meanLength / result.poolSize,
    rateVariance: variance,
    overlap: overlapRate(result.poolSize, meanLength, variance),
    chiSquare: exposureChiSquare(result.poolSize, meanLength, rates),
    aboveTarget,
    target,
  };
}

/** Reduce a study to the numbers a bank owner would compare policies on. */
export function summarise(result: StudyResult, options: SummaryOptions = {}): StudySummary {
  if (result.outcomes.length === 0) {
    throw new RangeError('summarise: the study has no outcomes');
  }
  const trueThetas = result.outcomes.map((outcome) => outcome.trueTheta);
  const estimates = result.outcomes.map((outcome) => outcome.estimate);
  const errors = result.outcomes.map((outcome) => outcome.estimate - outcome.trueTheta);
  const lengths = result.outcomes.map((outcome) => outcome.length);
  const rmse = Math.sqrt(mean(errors.map((error) => error * error)));
  const meanStandardError = mean(result.outcomes.map((outcome) => outcome.standardError));
  const covered = result.outcomes.filter(
    (outcome) =>
      Math.abs(outcome.estimate - outcome.trueTheta) <= NORMAL_95 * outcome.standardError,
  ).length;

  return {
    policy: result.policy,
    population: result.population,
    examinees: result.outcomes.length,
    poolSize: result.poolSize,
    bias: mean(errors),
    rmse,
    meanStandardError,
    calibration: rmse === 0 ? Number.POSITIVE_INFINITY : meanStandardError / rmse,
    coverage: covered / result.outcomes.length,
    meanLength: mean(lengths),
    lengthRange: [Math.min(...lengths), Math.max(...lengths)] as const,
    correlation: correlation(trueThetas, estimates),
    exposure: exposureSummary(result, options.exposureTarget ?? 0.2),
    conditional: conditionalReport(result, options),
  };
}

function pad(text: string, width: number): string {
  return text.length >= width ? text : text + ' '.repeat(width - text.length);
}

function padStart(text: string, width: number): string {
  return text.length >= width ? text : ' '.repeat(width - text.length) + text;
}

function fixed(value: number, digits: number): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '-';
}

/**
 * A fixed-width comparison of several policies.
 *
 * Deliberately plain text rather than a chart or a data frame: the audience for
 * this table is somebody deciding whether a two-item saving is worth a five-point
 * rise in overlap, and that decision is made by reading two rows against each
 * other.
 */
export function comparisonTable(summaries: readonly StudySummary[]): string {
  if (summaries.length === 0) return '(no policies)\n';
  const nameWidth = Math.max(6, ...summaries.map((s) => s.policy.length)) + 2;

  const header =
    pad('policy', nameWidth) +
    padStart('items', 7) +
    padStart('bias', 8) +
    padStart('rmse', 8) +
    padStart('se', 8) +
    padStart('calib', 8) +
    padStart('cover', 8) +
    padStart('r', 7) +
    padStart('max_x', 8) +
    padStart('overlap', 9) +
    padStart('unused', 8) +
    '\n';

  let out = header + '-'.repeat(header.length - 1) + '\n';
  for (const summary of summaries) {
    out +=
      pad(summary.policy, nameWidth) +
      padStart(fixed(summary.meanLength, 1), 7) +
      padStart(fixed(summary.bias, 3), 8) +
      padStart(fixed(summary.rmse, 3), 8) +
      padStart(fixed(summary.meanStandardError, 3), 8) +
      padStart(fixed(summary.calibration, 2), 8) +
      padStart(fixed(summary.coverage, 3), 8) +
      padStart(fixed(summary.correlation, 3), 7) +
      padStart(fixed(summary.exposure.maxRate, 3), 8) +
      padStart(fixed(summary.exposure.overlap, 3), 9) +
      padStart(fixed(summary.exposure.unused, 3), 8) +
      '\n';
  }
  return out;
}

/** A fixed-width conditional report for one policy. */
export function conditionalTable(summary: StudySummary): string {
  const header =
    pad('ability', 16) +
    padStart('n', 6) +
    padStart('bias', 8) +
    padStart('rmse', 8) +
    padStart('se', 8) +
    padStart('calib', 8) +
    padStart('cover', 8) +
    padStart('items', 8) +
    '\n';

  let out = header + '-'.repeat(header.length - 1) + '\n';
  for (const bin of summary.conditional) {
    out +=
      pad(`[${fixed(bin.lower, 1)}, ${fixed(bin.upper, 1)})`, 16) +
      padStart(String(bin.count), 6) +
      padStart(fixed(bin.bias, 3), 8) +
      padStart(fixed(bin.rmse, 3), 8) +
      padStart(fixed(bin.meanStandardError, 3), 8) +
      padStart(fixed(bin.calibration, 2), 8) +
      padStart(fixed(bin.coverage, 3), 8) +
      padStart(fixed(bin.meanLength, 1), 8) +
      '\n';
  }
  return out;
}
