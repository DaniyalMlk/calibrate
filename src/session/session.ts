import { createRng, type Rng } from '../core/random.js';
import type { AbilityEstimate } from '../estimation/mle.js';
import {
  maximumScoreOf,
  requireValidCategory,
  type AnyItem,
  type CategoryResponse,
} from '../models/mixed.js';
import type { Selector } from '../selection/selector.js';
import { hybridEstimator, type SessionEstimator } from './estimator.js';
import type { ItemPool } from './pool.js';
import type { StopReason, StoppingRule } from './stopping.js';

/** One administered item, with the estimate before and after it was answered. */
export interface TranscriptEntry {
  /** 1-based position in the test. */
  readonly position: number;
  readonly itemId: string;
  /**
   * The category scored, `0..maximumScore`. For a dichotomous item this is the
   * familiar 0 or 1; for a polytomous one it is the rubric level awarded.
   */
  readonly response: number;
  /** The highest category the item could have scored, so a transcript row reads on its own. */
  readonly maximumScore: number;
  /** Ability estimate used to select this item. */
  readonly thetaBefore: number;
  /** Standard error before the response. */
  readonly standardErrorBefore: number;
  /** Ability estimate after scoring the response. */
  readonly thetaAfter: number;
  readonly standardErrorAfter: number;
  /** Which estimator produced `thetaAfter`. */
  readonly method: AbilityEstimate['method'];
}

export type SessionStatus = 'ready' | 'awaiting-response' | 'finished';

export interface SessionConfig {
  readonly pool: ItemPool;
  readonly selector: Selector;
  readonly stopping: StoppingRule;
  /** Defaults to the hybrid EAP/WLE estimator. */
  readonly estimator?: SessionEstimator;
  /** Seed for the session's generator. Defaults to 1. */
  readonly seed?: number;
}

/** A serialisable summary of a session, suitable for storage or audit. */
export interface SessionSnapshot {
  readonly status: SessionStatus;
  readonly administeredIds: readonly string[];
  readonly responses: readonly number[];
  readonly theta: number;
  readonly standardError: number;
  readonly method: AbilityEstimate['method'];
  readonly stopReason: StopReason | null;
  readonly transcript: readonly TranscriptEntry[];
}

/**
 * A single adaptive testing session.
 *
 * The loop is: estimate ability from what is known, select the item that best
 * addresses that estimate, score the answer, re-estimate, ask the stopping rule.
 *
 * The status is explicit and enforced. Calling `submit` before `nextItem`, or
 * `nextItem` twice in a row, throws rather than being absorbed — a session that
 * silently accepts a response to an item it never administered produces a
 * transcript that cannot be trusted, and the whole point of a transcript is that
 * it can be.
 */
export class AdaptiveSession {
  readonly #pool: ItemPool;
  readonly #selector: Selector;
  readonly #stopping: StoppingRule;
  readonly #estimator: SessionEstimator;
  readonly #rng: Rng;

  #status: SessionStatus = 'ready';
  #used = new Set<string>();
  #responses: CategoryResponse[] = [];
  #transcript: TranscriptEntry[] = [];
  #pending: AnyItem | null = null;
  #estimate: AbilityEstimate;
  #stopReason: StopReason | null = null;

  constructor(config: SessionConfig) {
    this.#pool = config.pool;
    this.#selector = config.selector;
    this.#stopping = config.stopping;
    this.#estimator = config.estimator ?? hybridEstimator();
    this.#rng = createRng(config.seed ?? 1);
    this.#estimate = this.#estimator([]);
  }

  get status(): SessionStatus {
    return this.#status;
  }

  get estimate(): AbilityEstimate {
    return this.#estimate;
  }

  get responses(): readonly CategoryResponse[] {
    return this.#responses;
  }

  get transcript(): readonly TranscriptEntry[] {
    return this.#transcript;
  }

  get stopReason(): StopReason | null {
    return this.#stopReason;
  }

  /** Items administered so far, in order. */
  get administered(): AnyItem[] {
    return this.#transcript.map((entry) => {
      const item = this.#pool.byId(entry.itemId);
      if (item === undefined) throw new Error(`transcript references unknown item "${entry.itemId}"`);
      return item;
    });
  }

  /**
   * Select and return the next item, or `null` if the session is over.
   *
   * Pool exhaustion is an implicit stop: a selector that cannot choose ends the
   * session with a `pool-exhausted` reason rather than throwing, because running
   * out of eligible items is a normal outcome, not a fault.
   */
  nextItem(): AnyItem | null {
    if (this.#status === 'finished') return null;
    if (this.#status === 'awaiting-response') {
      throw new Error(
        `AdaptiveSession: item "${this.#pending?.id ?? '?'}" is already awaiting a response`,
      );
    }

    const candidates = this.#pool.eligible(this.#used);
    const item =
      candidates.length === 0
        ? null
        : this.#selector.select({
            candidates,
            theta: this.#estimate.theta,
            responses: this.#responses,
            rng: this.#rng,
          });

    if (item === null) {
      this.#finish({ rule: 'pool-exhausted', detail: 'no eligible items remain' });
      return null;
    }

    this.#pending = item;
    this.#status = 'awaiting-response';
    return item;
  }

  /**
   * Score the pending item, re-estimate, and evaluate the stopping rule.
   *
   * The argument is a category index, validated against the pending item's own
   * maximum. A dichotomous item takes 0 or 1 as it always did; a rubric-scored
   * item takes the level awarded. Passing a category the item cannot produce
   * throws rather than being clamped, because it means the response was scored
   * against a different item than the one being administered.
   */
  submit(category: number): void {
    if (this.#status !== 'awaiting-response' || this.#pending === null) {
      throw new Error('AdaptiveSession: no item is awaiting a response; call nextItem() first');
    }
    const item = this.#pending;
    const response = requireValidCategory(item, category);

    const before = this.#estimate;

    this.#responses.push({ item, category: response });
    this.#used.add(item.id);
    this.#estimate = this.#estimator(this.#responses);
    this.#pending = null;
    this.#status = 'ready';

    this.#transcript.push({
      position: this.#transcript.length + 1,
      itemId: item.id,
      response,
      maximumScore: maximumScoreOf(item),
      thetaBefore: before.theta,
      standardErrorBefore: before.standardError,
      thetaAfter: this.#estimate.theta,
      standardErrorAfter: this.#estimate.standardError,
      method: this.#estimate.method,
    });

    const reason = this.#stopping.shouldStop({
      administered: this.#responses.length,
      estimate: this.#estimate,
      remaining: this.#pool.size - this.#used.size,
    });
    if (reason !== null) this.#finish(reason);
  }

  /**
   * Run the session to completion, asking `respond` for each item.
   *
   * The convenience path used by the simulation harness and by any caller that
   * already has the answers.
   */
  run(respond: (item: AnyItem) => number): SessionSnapshot {
    for (;;) {
      const item = this.nextItem();
      if (item === null) break;
      this.submit(respond(item));
    }
    return this.snapshot();
  }

  /** A plain-object view of the session, safe to serialise. */
  snapshot(): SessionSnapshot {
    return {
      status: this.#status,
      administeredIds: this.#transcript.map((entry) => entry.itemId),
      responses: this.#transcript.map((entry) => entry.response),
      theta: this.#estimate.theta,
      standardError: this.#estimate.standardError,
      method: this.#estimate.method,
      stopReason: this.#stopReason,
      transcript: [...this.#transcript],
    };
  }

  #finish(reason: StopReason): void {
    this.#stopReason = reason;
    this.#status = 'finished';
    this.#pending = null;
  }
}
