import { createRng } from '../core/random.js';
import type { EstimatorName } from '../estimation/mle.js';
import type { Item } from '../models/item.js';
import type { Response } from '../models/response.js';
import type { Selector } from '../selection/selector.js';
import type { SessionEstimator } from '../session/estimator.js';
import type { ItemPool } from '../session/pool.js';
import { AdaptiveSession } from '../session/session.js';
import type { StoppingRule } from '../session/stopping.js';
import { simulateResponse } from './respondent.js';

/**
 * A named configuration of the three things that make an adaptive test a
 * policy: how it chooses items, how it scores, and when it stops.
 *
 * Bundling them under one name is what makes a comparison meaningful. The
 * interesting trade-offs — precision against test length, precision against
 * bank exposure — only appear when the whole configuration is varied together,
 * because a selector that exposes the bank evenly is only worth its cost if the
 * stopping rule is not then forced to run longer to reach the same precision.
 */
export interface Policy {
  /** Stable identifier, reproduced in every report. */
  readonly name: string;
  readonly selector: Selector;
  readonly stopping: StoppingRule;
  /** Defaults to the session's own default, the hybrid EAP/WLE estimator. */
  readonly estimator?: SessionEstimator;
}

export interface StudyOptions {
  readonly pool: ItemPool;
  /** True abilities, one per simulated examinee. */
  readonly abilities: readonly number[];
  /** Base seed for the study. Default 20260101. */
  readonly seed?: number;
  /** Name of the population the abilities came from, carried into the report. */
  readonly population?: string;
}

/** What one simulated examinee produced. */
export interface SessionOutcome {
  /** The ability the responses were generated from. */
  readonly trueTheta: number;
  /** The ability the session reported. */
  readonly estimate: number;
  /** The standard error the session reported alongside its estimate. */
  readonly standardError: number;
  /** Estimator that produced the final estimate. */
  readonly method: EstimatorName;
  /** Number of items administered. */
  readonly length: number;
  /** Items administered, in order. */
  readonly itemIds: readonly string[];
  /** Which stopping rule ended the session. */
  readonly stopRule: string;
}

export interface StudyResult {
  readonly policy: string;
  readonly population: string;
  readonly poolSize: number;
  readonly outcomes: readonly SessionOutcome[];
}

/**
 * Run one simulated adaptive session and record its outcome.
 *
 * Kept separate from `runStudy` so a caller can drive a single examinee — the
 * demo command does exactly this — without constructing a whole study.
 */
export function simulateSession(
  policy: Policy,
  pool: ItemPool,
  trueTheta: number,
  seed: number,
): SessionOutcome {
  const session = new AdaptiveSession({
    pool,
    selector: policy.selector,
    stopping: policy.stopping,
    ...(policy.estimator === undefined ? {} : { estimator: policy.estimator }),
    seed,
  });

  // A generator distinct from the session's own, so that changing the selection
  // policy does not change the answers a given examinee would have given to an
  // item both policies happened to administer. Without the split, a comparison
  // between two policies would be confounded by the responses themselves.
  const answers = createRng(seed ^ 0x5f3759df);
  const snapshot = session.run((item: Item): Response => simulateResponse(item, trueTheta, answers));

  return {
    trueTheta,
    estimate: snapshot.theta,
    standardError: snapshot.standardError,
    method: snapshot.method,
    length: snapshot.transcript.length,
    itemIds: snapshot.administeredIds,
    stopRule: snapshot.stopReason?.rule ?? 'none',
  };
}

/**
 * Run a policy against a whole simulated population.
 *
 * Each examinee gets a seed derived from their index rather than from a single
 * generator threaded through the loop. That makes examinee `i` identical
 * whether the study ran 10 examinees or 10,000, and identical across policies,
 * so two studies differ only by the thing that was actually varied. A single
 * shared stream would make every examinee after the first depend on how much
 * randomness the ones before them happened to consume — which itself depends on
 * the policy, and would quietly turn a controlled comparison into an
 * uncontrolled one.
 */
export function runStudy(policy: Policy, options: StudyOptions): StudyResult {
  const { pool, abilities } = options;
  if (abilities.length === 0) {
    throw new RangeError('runStudy: at least one examinee is required');
  }
  const seed = options.seed ?? 20260101;

  const outcomes: SessionOutcome[] = new Array<SessionOutcome>(abilities.length);
  for (let i = 0; i < abilities.length; i += 1) {
    outcomes[i] = simulateSession(policy, pool, abilities[i] as number, seed + i * 7919);
  }

  return {
    policy: policy.name,
    population: options.population ?? 'unnamed',
    poolSize: pool.size,
    outcomes,
  };
}

/** Run several policies against the same population and pool. */
export function compareStudies(
  policies: readonly Policy[],
  options: StudyOptions,
): StudyResult[] {
  if (policies.length === 0) throw new RangeError('compareStudies: at least one policy is required');
  const names = new Set<string>();
  for (const policy of policies) {
    if (names.has(policy.name)) {
      throw new RangeError(`compareStudies: duplicate policy name "${policy.name}"`);
    }
    names.add(policy.name);
  }
  return policies.map((policy) => runStudy(policy, options));
}
