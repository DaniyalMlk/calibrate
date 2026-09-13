import type { QuadratureRule } from '../core/quadrature.js';
import { maximumTestScore, type AnyItem, type AnyResponse } from '../models/mixed.js';
import { logLikelihood } from '../estimation/likelihood.js';
import { scoreLikelihoods } from './summed.js';

/** One respondent's ability estimate, with the uncertainty around it. */
export interface Score {
  /** Posterior mean of ability. */
  readonly ability: number;
  /** Posterior standard deviation, the standard error of the estimate. */
  readonly standardError: number;
}

/** A row of a score conversion table. */
export interface ScoreRow extends Score {
  /** The total score this row converts. */
  readonly score: number;
  /** Share of the population expected to obtain this total. */
  readonly proportion: number;
}

export interface Reliability {
  /**
   * Marginal reliability: the share of population ability variance the form
   * recovers, `1 - E[SE^2] / Var(theta)`.
   */
  readonly marginal: number;
  /** Population-average squared standard error, the numerator above. */
  readonly meanSquaredError: number;
  /** Variance of the population the reliability is quoted against. */
  readonly populationVariance: number;
  /** The average standard error itself, on the ability scale. */
  readonly meanStandardError: number;
}

/**
 * Posterior mean and standard deviation of ability, on an arbitrary population.
 *
 * The estimator in `bayes.ts` takes a normal prior and builds its own rule from
 * the prior's mean and standard deviation. That is the right interface for an
 * adaptive session, where the prior is a modelling choice made in advance. It is
 * the wrong one here: after a marginal calibration the population is not a
 * modelling choice but an estimate, and it need not be normal at all. This
 * version takes the population as a rule and uses it exactly as given.
 */
export function scorePattern(
  responses: readonly AnyResponse[],
  rule: QuadratureRule,
): Score {
  const count = rule.nodes.length;
  if (count !== rule.weights.length) {
    throw new RangeError('scorePattern: rule has mismatched nodes and weights');
  }

  const logs = new Array<number>(count);
  let peak = Number.NEGATIVE_INFINITY;
  for (let k = 0; k < count; k += 1) {
    const value = logLikelihood(responses, rule.nodes[k] as number);
    logs[k] = value;
    if (value > peak) peak = value;
  }

  return momentsOf(rule, (k) => (rule.weights[k] as number) * Math.exp((logs[k] as number) - peak));
}

/**
 * Posterior mean and standard deviation for one total score.
 *
 * Takes the likelihood table from `scoreLikelihoods`, so the expensive part —
 * which does not depend on the candidate — is done once for the whole form.
 */
export function scoreFromTotal(
  likelihoods: readonly Float64Array[],
  total: number,
  rule: QuadratureRule,
): Score {
  const row = likelihoods[total];
  if (row === undefined) {
    throw new RangeError(
      `scoreFromTotal: no total ${total} on a form scored 0 to ${likelihoods.length - 1}`,
    );
  }
  return momentsOf(rule, (k) => (rule.weights[k] as number) * (row[k] as number));
}

/** Mean and standard deviation of the posterior implied by an unnormalised weight. */
function momentsOf(rule: QuadratureRule, weightAt: (k: number) => number): Score {
  let mass = 0;
  let first = 0;
  let second = 0;
  for (let k = 0; k < rule.nodes.length; k += 1) {
    const theta = rule.nodes[k] as number;
    const weight = weightAt(k);
    mass += weight;
    first += weight * theta;
    second += weight * theta * theta;
  }
  if (!(mass > 0) || !Number.isFinite(mass)) {
    throw new Error(
      'scoring: the posterior has no mass on the quadrature grid; ' +
        'widen the population or add nodes',
    );
  }

  const ability = first / mass;
  // Computed as E[theta^2] - E[theta]^2, which can go very slightly negative
  // through cancellation when the posterior is concentrated on one node.
  const variance = Math.max(0, second / mass - ability * ability);
  return { ability, standardError: Math.sqrt(variance) };
}

/**
 * The score conversion table for a fixed form.
 *
 * One row per attainable total, giving the ability it converts to, the standard
 * error at that total, and the share of the population expected to land there.
 * This is the artefact a fixed-form programme publishes: candidates are reported
 * from their total, not from which particular items they happened to get right.
 *
 * **Under the Rasch model that loses nothing.** The total score is a sufficient
 * statistic for ability there — two candidates with the same total have exactly
 * proportional likelihood functions, so they have the same posterior, and
 * reading their ability off the table is not an approximation but an identity.
 * The test suite checks it as one.
 *
 * **Under the 2PL it does lose something.** Items with different
 * discriminations carry different amounts of evidence, so getting the two most
 * discriminating items right is worth more than getting two others right, and
 * the total cannot tell those cases apart. Whether that matters is a policy
 * question — reporting from the total is far easier to explain and to appeal —
 * but it should be a decision rather than an accident, so the size of what is
 * given up is measurable here rather than assumed away.
 */
export function conversionTable(items: readonly AnyItem[], rule: QuadratureRule): ScoreRow[] {
  const likelihoods = scoreLikelihoods(items, rule);
  const rows: ScoreRow[] = [];

  for (let total = 0; total <= maximumTestScore(items); total += 1) {
    const row = likelihoods[total] as Float64Array;
    let proportion = 0;
    for (let k = 0; k < rule.nodes.length; k += 1) {
      proportion += (rule.weights[k] as number) * (row[k] as number);
    }
    const score = scoreFromTotal(likelihoods, total, rule);
    rows.push({ score: total, proportion, ...score });
  }

  // The proportions are a distribution over totals up to quadrature accuracy;
  // renormalising keeps them one.
  let mass = 0;
  for (const row of rows) mass += row.proportion;
  return mass > 0 ? rows.map((row) => ({ ...row, proportion: row.proportion / mass })) : rows;
}

/**
 * How much of the population's ability variance a form recovers.
 *
 * Classical reliability is the ratio of true-score variance to observed-score
 * variance, which is a single number for a whole test and assumes the error
 * variance is the same for everyone. Under item response theory it is not: a
 * form measures precisely where its items are concentrated and poorly at the
 * ends, so the standard error is a function of ability rather than a constant.
 *
 * Marginal reliability is the honest version of the same summary. It averages
 * the squared standard error over the population rather than pretending it is
 * constant, and reports `1 - E[SE^2] / Var(theta)`. The number means what
 * classical reliability is usually taken to mean — the share of the spread in
 * ability that the form can actually see — without the assumption that does not
 * hold.
 *
 * It is computed here from the conversion table, so it is the reliability of
 * *summed-score* reporting specifically, including whatever that reporting gives
 * up. A form reported from response patterns is at least as reliable.
 */
export function marginalReliability(
  items: readonly AnyItem[],
  rule: QuadratureRule,
): Reliability {
  const rows = conversionTable(items, rule);

  let meanSquaredError = 0;
  let meanStandardError = 0;
  for (const row of rows) {
    meanSquaredError += row.proportion * row.standardError * row.standardError;
    meanStandardError += row.proportion * row.standardError;
  }

  let mass = 0;
  let mean = 0;
  for (const [k, weight] of rule.weights.entries()) {
    mass += weight;
    mean += weight * (rule.nodes[k] as number);
  }
  mean = mass > 0 ? mean / mass : 0;
  let populationVariance = 0;
  for (const [k, weight] of rule.weights.entries()) {
    const centred = (rule.nodes[k] as number) - mean;
    populationVariance += weight * centred * centred;
  }
  populationVariance = mass > 0 ? populationVariance / mass : 0;

  return {
    marginal:
      populationVariance > 0 ? 1 - meanSquaredError / populationVariance : Number.NaN,
    meanSquaredError,
    populationVariance,
    meanStandardError,
  };
}

/** Render a conversion table as aligned text. */
export function conversionText(rows: readonly ScoreRow[]): string {
  let out = ' score     theta       se    share\n';
  out += '-'.repeat(34) + '\n';
  for (const row of rows) {
    out +=
      `${String(row.score).padStart(6)}` +
      `${row.ability.toFixed(3).padStart(10)}` +
      `${row.standardError.toFixed(3).padStart(9)}` +
      `${(row.proportion * 100).toFixed(1).padStart(8)}%\n`;
  }
  return out;
}
