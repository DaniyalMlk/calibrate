import {
  AdaptiveSession,
  blueprint,
  contentBalanced,
  createRng,
  credibleInterval,
  hybridEstimator,
  ItemPool,
  kullbackLeiblerSelector,
  maximumInformationSelector,
  posteriorDensity,
  precisionTarget,
  probabilityCorrect,
  randomesque,
  syntheticBank,
  type CredibleInterval,
  type Item,
  type Posterior,
  type Response,
  type ScoredResponse,
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
  readonly seed: number;
}

export const DEFAULT_CONFIGURATION: Configuration = {
  policy: 'max-information',
  target: 0.3,
  maximum: 30,
  trueTheta: 0.8,
  bankSize: 300,
  seed: 42,
};

/** The ability domain every view shares. Locked, never fitted to the data. */
export const THETA_DOMAIN: readonly [number, number] = [-4, 4];

const DOMAINS = ['arrays', 'graphs', 'dynamic-programming'] as const;

function selectorFor(policy: PolicyName): Selector {
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
  readonly bank: readonly Item[];
  readonly snapshot: SessionSnapshot;
  readonly responses: readonly ScoredResponse[];
  readonly current: Item | null;
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
  private bank: readonly Item[] = [];
  private session!: AdaptiveSession;
  private current: Item | null = null;
  private answered: ScoredResponse[] = [];
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

  private rebuild(configuration: Configuration, reuse?: readonly Item[]): void {
    this.configuration = configuration;
    this.bank =
      reuse ?? syntheticBank({ size: configuration.bankSize, domains: DOMAINS, seed: configuration.seed });
    this.answered = [];

    this.session = new AdaptiveSession({
      pool: new ItemPool(this.bank),
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

  /** Draw the next item, if the session has not stopped. */
  private advance(): void {
    this.current = this.session.status === 'finished' ? null : this.session.nextItem();
  }

  /** Record a response to the current item and draw the next. */
  answer(response: Response): void {
    const item = this.current;
    if (item === null) {
      return;
    }

    this.exposure.set(item.id, (this.exposure.get(item.id) ?? 0) + 1);
    this.answered.push({ item, response });
    this.session.submit(response);
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
   * would make every estimate look better than it has any right to.
   */
  simulateOne(): void {
    if (this.current === null) {
      return;
    }
    const probability = probabilityCorrect(this.current.parameters, this.configuration.trueTheta);
    this.answer(this.rng.next() < probability ? 1 : 0);
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
    const responses: readonly ScoredResponse[] = this.answered;
    const posterior = posteriorDensity(responses, {
      lower: THETA_DOMAIN[0],
      upper: THETA_DOMAIN[1],
      points: 321,
    });

    return {
      configuration: this.configuration,
      bank: this.bank,
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
    };
  }

  private emit(): void {
    const model = this.model();
    for (const listener of this.listeners) {
      listener(model);
    }
  }
}
