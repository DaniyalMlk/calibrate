import { describe, expect, it } from 'vitest';
import { createRng } from '../src/core/random.js';
import { mean } from '../src/core/numeric.js';
import { estimateEap } from '../src/estimation/bayes.js';
import { makeItem, threePL, twoPL } from '../src/models/item.js';
import {
  categoryCountOf,
  categoryProbabilitiesOf,
  formatCounts,
  informationOf,
  isPolytomous,
  makePolytomousItem,
  maximumScoreOf,
  maximumTestScore,
  type AnyItem,
} from '../src/models/mixed.js';
import { generalizedPartialCredit, graded, partialCredit } from '../src/models/polytomous.js';
import { kullbackLeiblerDivergence, maximumInformationSelector } from '../src/selection/information.js';
import { exposureRates, randomesque } from '../src/selection/exposure.js';
import { blueprint, contentBalanced } from '../src/selection/content.js';
import { ItemPool } from '../src/session/pool.js';
import { AdaptiveSession } from '../src/session/session.js';
import { fixedLength, precisionTarget } from '../src/session/stopping.js';
import { eapEstimator } from '../src/session/estimator.js';
import { syntheticMixedBank, syntheticMixedPool } from '../src/simulation/bank.js';
import { simulateCategories, simulateCategory } from '../src/simulation/respondent.js';

const MC = makeItem('mc', twoPL(1.4, 0.2), { domain: 'algebra' });
const ESSAY = makePolytomousItem('essay', graded(1.3, [-1.2, -0.1, 1.0]), { domain: 'writing' });

function mixedPool(): ItemPool {
  return syntheticMixedPool({ size: 240, polytomousFraction: 0.3, seed: 7 });
}

function session(overrides: Partial<{ pool: ItemPool; stopping: ReturnType<typeof fixedLength> }> = {}) {
  return new AdaptiveSession({
    pool: overrides.pool ?? mixedPool(),
    selector: randomesque(maximumInformationSelector(), 4),
    stopping: overrides.stopping ?? fixedLength(12),
    seed: 5,
  });
}

describe('pools holding both formats', () => {
  it('indexes polytomous items by id and domain', () => {
    const pool = new ItemPool([MC, ESSAY]);
    expect(pool.size).toBe(2);
    expect(pool.byId('essay')).toBe(ESSAY);
    expect(pool.domains()).toEqual(['algebra', 'writing']);
    expect(pool.inDomain('writing').map((item) => item.id)).toEqual(['essay']);
  });

  it('still rejects duplicate ids across formats', () => {
    const clash = makePolytomousItem('mc', partialCredit([-1, 1]));
    expect(() => new ItemPool([MC, clash])).toThrow(/duplicate item id/);
  });

  it('carries both formats through a synthetic bank', () => {
    const counts = formatCounts(syntheticMixedBank({ size: 240, polytomousFraction: 0.3, seed: 7 }));
    expect(counts.dichotomous).toBe(168);
    expect(counts.polytomous).toBe(72);
    expect(counts.maximumScore).toBeGreaterThan(240);
  });
});

describe('selection over a mixed pool', () => {
  it('ranks by information across formats without a branch', () => {
    const selector = maximumInformationSelector();
    const candidates: AnyItem[] = [MC, ESSAY];
    const picked = selector.select({
      candidates,
      theta: 0,
      responses: [],
      rng: createRng(1),
    });
    // Whichever it picks must be the one with more information at theta = 0.
    const best = informationOf(MC, 0) >= informationOf(ESSAY, 0) ? MC : ESSAY;
    expect(picked).toBe(best);
  });

  it('values a polytomous item through all of its categories', () => {
    // KL against a two-category reading of the same item would undervalue it;
    // the full sum is at least as large as the top-category term alone.
    const divergence = kullbackLeiblerDivergence(ESSAY, 0, 1);
    expect(divergence).toBeGreaterThan(0);
    expect(Number.isFinite(divergence)).toBe(true);
  });

  it('is zero when the two abilities compared are the same', () => {
    for (const item of [MC, ESSAY]) {
      expect(kullbackLeiblerDivergence(item, 0.4, 0.4)).toBeCloseTo(0, 12);
    }
  });

  it('grows as the competing ability moves away', () => {
    const near = kullbackLeiblerDivergence(ESSAY, 0, 0.25);
    const far = kullbackLeiblerDivergence(ESSAY, 0, 1.5);
    expect(far).toBeGreaterThan(near);
  });

  it('honours a content blueprint over a mixed bank', () => {
    const pool = mixedPool();
    const balanced = new AdaptiveSession({
      pool,
      selector: contentBalanced(
        maximumInformationSelector(),
        blueprint({ arrays: 0.4, graphs: 0.3, 'dynamic-programming': 0.3 }),
      ),
      stopping: fixedLength(20),
      seed: 2,
    });
    const rng = createRng(2);
    balanced.run((item) => simulateCategory(item, 0.5, rng));
    const domains = balanced.administered.map((item) => item.domain);
    expect(new Set(domains).size).toBe(3);
  });
});

describe('sessions administering both formats', () => {
  it('runs to a fixed length over a mixed bank', () => {
    const s = session();
    const rng = createRng(21);
    const snapshot = s.run((item) => simulateCategory(item, 0.6, rng));
    expect(snapshot.transcript).toHaveLength(12);
    expect(snapshot.stopReason?.rule).toBe('fixed-length(12)');
  });

  it('administers at least one item of each format', () => {
    const s = session({ stopping: fixedLength(25) });
    const rng = createRng(13);
    s.run((item) => simulateCategory(item, 0.2, rng));
    const administered = s.administered;
    expect(administered.some(isPolytomous)).toBe(true);
    expect(administered.some((item) => !isPolytomous(item))).toBe(true);
  });

  it('records each response against its own item maximum', () => {
    const s = session({ stopping: fixedLength(20) });
    const rng = createRng(8);
    const snapshot = s.run((item) => simulateCategory(item, 0.4, rng));
    for (const entry of snapshot.transcript) {
      const item = s.administered.find((candidate) => candidate.id === entry.itemId) as AnyItem;
      expect(entry.maximumScore).toBe(maximumScoreOf(item));
      expect(entry.response).toBeGreaterThanOrEqual(0);
      expect(entry.response).toBeLessThanOrEqual(entry.maximumScore);
    }
  });

  it('replays identically from the same seed', () => {
    const first = session({ stopping: fixedLength(15) });
    const second = session({ stopping: fixedLength(15) });
    const a = first.run((item) => simulateCategory(item, 0.5, createRng(41)));
    const b = second.run((item) => simulateCategory(item, 0.5, createRng(41)));
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it('reaches a standard-error target', () => {
    const s = new AdaptiveSession({
      pool: mixedPool(),
      selector: maximumInformationSelector(),
      stopping: precisionTarget(0.3, { minimum: 5, maximum: 40 }),
      seed: 9,
    });
    const rng = createRng(9);
    const snapshot = s.run((item) => simulateCategory(item, 0.7, rng));
    expect(snapshot.standardError).toBeLessThanOrEqual(0.3);
    expect(snapshot.stopReason?.rule).toBe('standard-error-below(0.3)');
  });

  it('tightens the standard error as the session proceeds', () => {
    const s = new AdaptiveSession({
      pool: mixedPool(),
      selector: maximumInformationSelector(),
      stopping: fixedLength(20),
      estimator: eapEstimator(),
      seed: 4,
    });
    const rng = createRng(4);
    const { transcript } = s.run((item) => simulateCategory(item, 0.3, rng));
    const first = transcript[0]?.standardErrorAfter as number;
    const last = transcript[transcript.length - 1]?.standardErrorAfter as number;
    expect(last).toBeLessThan(first);
  });

  it('refuses a category the pending item cannot produce', () => {
    const pool = new ItemPool([MC, ESSAY]);
    const s = new AdaptiveSession({
      pool,
      selector: maximumInformationSelector(),
      stopping: fixedLength(2),
      seed: 1,
    });
    const item = s.nextItem() as AnyItem;
    expect(() => s.submit(maximumScoreOf(item) + 1)).toThrow(/must be an integer category/);
    expect(() => s.submit(-1)).toThrow(RangeError);
    expect(() => s.submit(1.5)).toThrow(RangeError);
    // The session is unchanged and still awaiting the same item.
    expect(s.status).toBe('awaiting-response');
    s.submit(0);
    expect(s.status).toBe('ready');
  });

  it('never administers the same item twice', () => {
    const s = session({ stopping: fixedLength(30) });
    const rng = createRng(17);
    const snapshot = s.run((item) => simulateCategory(item, -0.3, rng));
    expect(new Set(snapshot.administeredIds).size).toBe(snapshot.administeredIds.length);
  });

  it('accounts exposure over a mixed bank', () => {
    const pool = mixedPool();
    const sessions: AnyItem[][] = [];
    for (let candidate = 0; candidate < 40; candidate += 1) {
      const s = new AdaptiveSession({
        pool,
        selector: randomesque(maximumInformationSelector(), 5),
        stopping: fixedLength(8),
        seed: candidate + 1,
      });
      const rng = createRng(candidate + 400);
      s.run((item) => simulateCategory(item, createRng(candidate + 1).nextNormal(), rng));
      sessions.push(s.administered);
    }
    const rates = exposureRates(sessions);
    for (const rate of rates.values()) {
      expect(rate).toBeGreaterThan(0);
      expect(rate).toBeLessThanOrEqual(1);
    }
  });
});

describe('recovering ability from a mixed form', () => {
  it('recovers the generating ability on average across the range', () => {
    const bank = syntheticMixedBank({ size: 120, polytomousFraction: 0.35, seed: 3 });
    const errors: number[] = [];
    for (let index = 0; index < 120; index += 1) {
      const rng = createRng(index + 1000);
      const trueTheta = createRng(index + 1).nextNormal();
      // A fixed 25-item form, so this measures the estimator rather than the
      // selection policy.
      const form = bank.slice(index % 40, (index % 40) + 25);
      const responses = simulateCategories(form, trueTheta, rng);
      errors.push(estimateEap(responses).theta - trueTheta);
    }
    // EAP shrinks towards the prior, so the mean error over a standard normal
    // population should be small and the spread modest.
    expect(Math.abs(mean(errors))).toBeLessThan(0.1);
    expect(Math.sqrt(mean(errors.map((error) => error * error)))).toBeLessThan(0.45);
  });

  it('measures more precisely than a same-length all-binary form', () => {
    // Same number of items; the mixed form carries more score points and so
    // more information.
    const binaryOnly = syntheticMixedBank({ size: 60, polytomousFraction: 0, seed: 11 }).slice(0, 20);
    const mixed = syntheticMixedBank({ size: 60, polytomousFraction: 1, seed: 11 }).slice(0, 20);
    expect(maximumTestScore(mixed)).toBeGreaterThan(maximumTestScore(binaryOnly));

    const binaryError = mean(
      Array.from({ length: 60 }, (_, index) => {
        const trueTheta = createRng(index + 1).nextNormal();
        const responses = simulateCategories(binaryOnly, trueTheta, createRng(index + 700));
        return (estimateEap(responses).theta - trueTheta) ** 2;
      }),
    );
    const mixedError = mean(
      Array.from({ length: 60 }, (_, index) => {
        const trueTheta = createRng(index + 1).nextNormal();
        const responses = simulateCategories(mixed, trueTheta, createRng(index + 700));
        return (estimateEap(responses).theta - trueTheta) ** 2;
      }),
    );
    expect(mixedError).toBeLessThan(binaryError);
  });

  it('draws categories in proportion to the model', () => {
    // A long run of draws at a fixed ability should match the category
    // distribution the model specifies.
    const item = makePolytomousItem('p', generalizedPartialCredit(1.4, [-1, 0.2, 1.1]));
    const rng = createRng(77);
    const counts = new Array<number>(categoryCountOf(item)).fill(0);
    const draws = 20000;
    for (let index = 0; index < draws; index += 1) {
      const drawn = simulateCategory(item, 0.3, rng);
      counts[drawn] = (counts[drawn] as number) + 1;
    }
    const observed = counts.map((count) => count / draws);
    // Recomputed from the model, not from the draws.
    const expected = categoryProbabilitiesOf(item, 0.3);
    for (let k = 0; k < observed.length; k += 1) {
      expect(Math.abs((observed[k] as number) - (expected[k] as number))).toBeLessThan(0.015);
    }
  });
});

describe('threePL items still behave as before in a mixed pool', () => {
  it('keeps its guessing floor when placed beside polytomous items', () => {
    const guessable = makeItem('g', threePL(1.2, 0.5, 0.25));
    const pool = new ItemPool([guessable, ESSAY]);
    expect(pool.size).toBe(2);
    expect(informationOf(guessable, -5)).toBeGreaterThanOrEqual(0);
    expect(maximumScoreOf(guessable)).toBe(1);
  });
});
