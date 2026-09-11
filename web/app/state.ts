import {
  AdaptiveSession,
  blueprint,
  contentBalanced,
  createRng,
  credibleInterval,
  formatCounts,
  hybridEstimator,
  ItemPool,
  kullbackLeiblerSelector,
  maximumInformationSelector,
  maximumTestScore,
  posteriorDensity,
  precisionTarget,
  randomesque,
  simulateCategory,
  syntheticMixedBank,
  type AnyItem,
  type CategoryResponse,
  type CredibleInterval,
  type Posterior,
  type Selector,
  type SessionSnapshot,
} from '../../src/index.js';

export type PolicyName = 'max-information' | 'kullback-leibler' | 'randomesque-5' | 'balanced';

export interface Configuration {
  readonly policy: PolicyName;
  /** Standard error at which the session stops. */
  readonly target: number;
  /** Ceiling on test length. */
  readonly maximum: number;
  /** Ability of the simulated candidate, for the auto-answer control. */
  readonly trueTheta: number;
  readonly bankSize: number;
  /**
   * Share of the bank that is rubric-scored, in [0, 1].
   *
   * Exposed as a control rather than fixed, because the mix is the question a
   * test designer is actually asking. A form is not "mixed" or "not mixed" —
   * it is twenty percent constructed response, and the consequences of that
   * for precision and for exposure are what the panels below are for.
   */
  readonly polytomousFraction: number;
  readonly seed: number;
}

export const DEFAULT_CONFIGURATION: Configuration = {
  policy: 'max-information',
  target: 0.3,
  maximum: 30,
  trueTheta: 0.8,
  bankSize: 300,
  polytomousFraction: 0.2,
  seed: 42,
};

/** The ability domain every view shares. Locked, never fitted to the data. */
export const THETA_DOMAIN: readonly [number, number] = [-4, 4];

const DOMAINS = ['arrays', 'graphs', 'dynamic-programming'] as const;

export function selectorFor(policy: PolicyName): Selector {
  switch (policy) {
    case 'kullback-leibler':
      return kullbackLeiblerSelector();
    case 'randomesque-5':
      return randomesque(maximumInformationSelector(), 5);
    case 'balanced':
      return contentBalanced(
        randomesque(maximumInformationSelector(), 5),
        blueprint({ arrays: 0.4, graphs: 0.3, 'dynamic-programming': 0.3 }),
      );
    case 'max-information':
      return maximumInformationSelector();
  }
}

export interface ViewModel {
  readonly configuration: Configuration;
  readonly bank: readonly AnyItem[];
  /** The same items as a pool, for views that run their own simulations. */
  readonly pool: ItemPool;
  /** Items by identifier, for resolving a transcript back to its items. */
  readonly byId: ReadonlyMap<string, AnyItem>;
  readonly snapshot: SessionSnapshot;
  readonly responses: readonly CategoryResponse[];
  readonly current: AnyItem | null;
  readonly posterior: Posterior;
  readonly intervals: {
    readonly inner: CredibleInterval;
    readonly middle: CredibleInterval;
    readonly outer: CredibleInterval;
  };
  readonly finished: boolean;
  /** Cumulative count of how often each item in the bank has been administered. */
  readonly exposure: ReadonlyMap<string, number>;
  readonly sessionsRun: number;
  /** How the bank splits between formats, and the score a full form could award. */
  readonly formats: {
    readonly dichotomous: number;
    readonly polytomous: number;
    readonly maximumScore: number;
  };
  /** Points actually scored so far, and the most the administered items could give. */
  readonly score: { readonly earned: number; readonly available: number };
}

type Listener = (model: ViewModel) => void;

/**
 * The single source of truth for every view.
 *
 * Views subscribe and re-render from the model they are handed; nothing reads
 * the session directly. That keeps the posterior panel, the information curves
 * and the exposure table showing the same response pattern as the transcript
 * beside them, which is the one invariant a reader will notice being broken.
 */
export class Store {
  private configuration: Configuration = DEFAULT_CONFIGURATION;
  private bank: readonly AnyItem[] = [];
  private pool!: ItemPool;
  private byId: ReadonlyMap<string, AnyItem> = new Map();
  private session!: AdaptiveSession;
  private current: AnyItem | null = null;
  private answered: CategoryResponse[] = [];
  private readonly listeners = new Set<Listener>();
  private readonly exposure = new Map<string, number>();
  private sessionsRun = 0;

  constructor() {
    this.rebuild(DEFAULT_CONFIGURATION);
  }

  subscribe(listener: Listener): void {
    this.listeners.add(listener);
    listener(this.model());
  }

  /** Rebuild bank and session from scratch. Resets accumulated exposure. */
  reconfigure(changes: Partial<Configuration>): void {
    this.exposure.clear();
    this.sessionsRun = 0;
    this.rebuild({ ...this.configuration, ...changes });
    this.emit();
  }

  /** Start a fresh session over the same bank, keeping accumulated exposure. */
  restart(): void {
    this.rebuild(this.configuration, this.bank);
    this.emit();
  }

  private rebuild(configuration: Configuration, reuse?: readonly AnyItem[]): void {
    this.configuration = configuration;
    this.bank =
      reuse ??
      syntheticMixedBank({
        size: configuration.bankSize,
        domains: DOMAINS,
        polytomousFraction: configuration.polytomousFraction,
        seed: configuration.seed,
      });
    this.pool = new ItemPool(this.bank);
    this.byId = new Map(this.bank.map((item) => [item.id, item]));
    this.answered = [];

    this.session = new AdaptiveSession({
      pool: this.pool,
      selector: selectorFor(configuration.policy),
      stopping: precisionTarget(configuration.target, {
        minimum: 5,
        maximum: configuration.maximum,
      }),
      estimator: hybridEstimator(),
      seed: configuration.seed,
    });
    this.current = null;
    this.advance();
  }

  /**
   * Draw the next item, if the session has not stopped.
   *
   * The pending item is resolved back through the index the bank was built
   * from rather than taken as handed over, so the item a view draws is
   * identically the object the model holds — which is what lets a view compare
   * by reference when deciding whether its selection is still valid.
   */
  private advance(): void {
    const next = this.session.status === 'finished' ? null : this.session.nextItem();
    this.current = next === null ? null : (this.byId.get(next.id) ?? null);
  }

  /**
   * Record a response to the current item and draw the next.
   *
   * The argument is a category index: 0 or 1 on a dichotomous item, the rubric
   * level on a polytomous one. The session validates it against the item's own
   * maximum and throws on a category the item cannot produce, which is the
   * behaviour worth keeping — a score of 3 arriving for a two-category item
   * means the button and the pending item have gone out of step, and clamping
   * it to 1 would write a wrong answer into the transcript with no trace.
   */
  answer(category: number): void {
    const item = this.current;
    if (item === null) {
      return;
    }

    this.exposure.set(item.id, (this.exposure.get(item.id) ?? 0) + 1);
    this.answered.push({ item, category });
    this.session.submit(category);
    this.advance();
    if (this.current === null) {
      this.sessionsRun += 1;
    }
    this.emit();
  }

  /**
   * Answer the current item as a candidate of the configured ability would.
   *
   * Draws from the response model rather than comparing ability to difficulty:
   * a candidate above an item's difficulty gets it right with probability
   * `P(theta)`, not with certainty, and a simulator that skipped the coin flip
   * would make every estimate look better than it has any right to. On a
   * rubric item the same draw runs over the whole category distribution, so a
   * candidate near the second threshold lands sometimes on either side of it.
   */
  simulateOne(): void {
    if (this.current === null) {
      return;
    }
    this.answer(simulateCategory(this.current, this.configuration.trueTheta, this.rng));
  }

  /** Answer through to the end of the session. */
  simulateRest(): void {
    let guard = 0;
    while (this.current !== null && guard < 500) {
      this.simulateOne();
      guard += 1;
    }
  }

  private readonly rng = createRng(20260907);

  private model(): ViewModel {
    const snapshot = this.session.snapshot();
    const responses: readonly CategoryResponse[] = this.answered;
    const posterior = posteriorDensity(responses, {
      lower: THETA_DOMAIN[0],
      upper: THETA_DOMAIN[1],
      points: 321,
    });

    // Points earned over points available, not a count of items. On a mixed
    // form those two are different numbers, and the raw score a candidate is
    // reported against is the first of them.
    let earned = 0;
    for (const response of responses) earned += response.category;

    return {
      configuration: this.configuration,
      bank: this.bank,
      pool: this.pool,
      byId: this.byId,
      snapshot,
      responses,
      current: this.current,
      posterior,
      intervals: {
        inner: credibleInterval(posterior, 0.5),
        middle: credibleInterval(posterior, 0.8),
        outer: credibleInterval(posterior, 0.95),
      },
      finished: this.session.status === 'finished' || this.current === null,
      exposure: this.exposure,
      sessionsRun: this.sessionsRun,
      formats: formatCounts(this.bank),
      score: {
        earned,
        available: maximumTestScore(responses.map((response) => response.item)),
      },
    };
  }

  private emit(): void {
    const model = this.model();
    for (const listener of this.listeners) {
      listener(model);
    }
  }
}
