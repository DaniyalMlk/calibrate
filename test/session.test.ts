import { describe, expect, it } from 'vitest';
import { linspace, mean } from '../src/core/numeric.js';
import { createRng } from '../src/core/random.js';
import { makeItem, twoPL, type Item } from '../src/models/item.js';
import { standardError, testInformation, type Response } from '../src/models/response.js';
import { maximumInformationSelector } from '../src/selection/information.js';
import { randomesque } from '../src/selection/exposure.js';
import {
  eapEstimator,
  hybridEstimator,
  mleEstimator,
  wleEstimator,
} from '../src/session/estimator.js';
import { ItemPool } from '../src/session/pool.js';
import { AdaptiveSession } from '../src/session/session.js';
import {
  allOf,
  anyOf,
  fixedLength,
  maximumItems,
  precisionTarget,
  standardErrorBelow,
  withMinimumLength,
} from '../src/session/stopping.js';
import { simulateCategory } from '../src/simulation/respondent.js';
import { itemLocation, type AnyItem } from '../src/models/mixed.js';

function bankItems(count: number, a = 1.3): Item[] {
  return linspace(-3, 3, count).map((b, index) => makeItem(`i-${index}`, twoPL(a, b)));
}

const pool = new ItemPool(bankItems(60));

describe('ItemPool', () => {
  it('indexes by id and domain', () => {
    const items = [
      makeItem('a', twoPL(1, 0), { domain: 'arrays' }),
      makeItem('b', twoPL(1, 1), { domain: 'graphs' }),
      makeItem('c', twoPL(1, 2), { domain: 'arrays' }),
      makeItem('d', twoPL(1, 3)),
    ];
    const p = new ItemPool(items);
    expect(p.size).toBe(4);
    expect(itemLocation(p.byId('c') as AnyItem)).toBe(2);
    expect(p.byId('missing')).toBeUndefined();
    expect(p.domains()).toEqual(['arrays', 'graphs']);
    expect(p.inDomain('arrays').map((i) => i.id)).toEqual(['a', 'c']);
    expect(p.inDomain('nothing')).toEqual([]);
  });

  it('rejects duplicate ids', () => {
    expect(() => new ItemPool([makeItem('a', twoPL(1, 0)), makeItem('a', twoPL(1, 1))])).toThrow(
      /duplicate item id/,
    );
  });

  it('rejects an empty bank', () => {
    expect(() => new ItemPool([])).toThrow(RangeError);
  });

  it('filters out used items while preserving order', () => {
    const eligible = pool.eligible(new Set(['i-0', 'i-2']));
    expect(eligible).toHaveLength(58);
    expect(eligible[0]?.id).toBe('i-1');
  });
});

describe('stopping rules', () => {
  const estimate = { theta: 0, standardError: 0.3, method: 'wle', converged: true, boundary: 'none', iterations: 3 } as const;

  it('fixedLength fires at exactly the requested length', () => {
    const rule = fixedLength(5);
    expect(rule.shouldStop({ administered: 4, estimate, remaining: 10 })).toBeNull();
    expect(rule.shouldStop({ administered: 5, estimate, remaining: 10 })?.rule).toBe('fixed-length(5)');
  });

  it('standardErrorBelow fires when precision is reached', () => {
    const rule = standardErrorBelow(0.25);
    expect(rule.shouldStop({ administered: 9, estimate, remaining: 5 })).toBeNull();
    const precise = { ...estimate, standardError: 0.2 };
    expect(rule.shouldStop({ administered: 9, estimate: precise, remaining: 5 })).not.toBeNull();
  });

  it('withMinimumLength suppresses an inner rule until the floor is met', () => {
    const precise = { ...estimate, standardError: 0.05 };
    const guarded = withMinimumLength(8, standardErrorBelow(0.25));
    expect(guarded.shouldStop({ administered: 2, estimate: precise, remaining: 5 })).toBeNull();
    expect(guarded.shouldStop({ administered: 8, estimate: precise, remaining: 5 })).not.toBeNull();
  });

  it('anyOf reports the first rule to fire', () => {
    const rule = anyOf(fixedLength(20), maximumItems(3));
    expect(rule.shouldStop({ administered: 3, estimate, remaining: 5 })?.rule).toBe('maximum-items(3)');
  });

  it('allOf waits for every rule', () => {
    const rule = allOf(maximumItems(3), standardErrorBelow(0.25));
    expect(rule.shouldStop({ administered: 5, estimate, remaining: 5 })).toBeNull();
    const precise = { ...estimate, standardError: 0.2 };
    expect(rule.shouldStop({ administered: 5, estimate: precise, remaining: 5 })).not.toBeNull();
  });

  it('precisionTarget combines the floor, the target and the ceiling', () => {
    const rule = precisionTarget(0.25, { minimum: 5, maximum: 12 });
    const precise = { ...estimate, standardError: 0.1 };
    expect(rule.shouldStop({ administered: 3, estimate: precise, remaining: 30 })).toBeNull();
    expect(rule.shouldStop({ administered: 5, estimate: precise, remaining: 30 })).not.toBeNull();
    expect(rule.shouldStop({ administered: 12, estimate, remaining: 30 })?.detail).toMatch(/ceiling/);
  });

  it('rejects degenerate configuration', () => {
    expect(() => fixedLength(0)).toThrow(RangeError);
    expect(() => standardErrorBelow(0)).toThrow(RangeError);
    expect(() => maximumItems(-1)).toThrow(RangeError);
    expect(() => withMinimumLength(-1, fixedLength(3))).toThrow(RangeError);
    expect(() => anyOf()).toThrow(RangeError);
    expect(() => allOf()).toThrow(RangeError);
    expect(() => precisionTarget(0.2, { minimum: 10, maximum: 5 })).toThrow(RangeError);
  });
});

describe('AdaptiveSession', () => {
  function session(overrides: Partial<ConstructorParameters<typeof AdaptiveSession>[0]> = {}): AdaptiveSession {
    return new AdaptiveSession({
      pool,
      selector: maximumInformationSelector(),
      stopping: fixedLength(10),
      seed: 7,
      ...overrides,
    });
  }

  it('starts ready, with an estimate from the prior alone', () => {
    const s = session();
    expect(s.status).toBe('ready');
    expect(s.estimate.theta).toBeCloseTo(0, 6);
    expect(s.transcript).toHaveLength(0);
    expect(s.stopReason).toBeNull();
  });

  it('runs a fixed-length test and stops for the stated reason', () => {
    const s = session();
    const rng = createRng(99);
    const snapshot = s.run((item) => simulateCategory(item, 0.8, rng));
    expect(snapshot.transcript).toHaveLength(10);
    expect(snapshot.stopReason?.rule).toBe('fixed-length(10)');
    expect(s.status).toBe('finished');
  });

  it('never administers the same item twice', () => {
    const s = session({ stopping: fixedLength(40) });
    const rng = createRng(3);
    const snapshot = s.run((item) => simulateCategory(item, -0.4, rng));
    expect(new Set(snapshot.administeredIds).size).toBe(snapshot.administeredIds.length);
  });

  it('produces an identical transcript on a re-run with the same seed', () => {
    const first = session({ selector: randomesque(maximumInformationSelector(), 4) });
    const second = session({ selector: randomesque(maximumInformationSelector(), 4) });
    const answersA = createRng(41);
    const answersB = createRng(41);
    const a = first.run((item) => simulateCategory(item, 0.5, answersA));
    const b = second.run((item) => simulateCategory(item, 0.5, answersB));
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it('records the estimate before and after each response', () => {
    const s = session({ stopping: fixedLength(6) });
    const rng = createRng(5);
    const { transcript } = s.run((item) => simulateCategory(item, 1.2, rng));
    for (let i = 1; i < transcript.length; i += 1) {
      // Each entry's "before" is the previous entry's "after".
      expect(transcript[i]?.thetaBefore).toBeCloseTo(transcript[i - 1]?.thetaAfter as number, 12);
    }
    expect(transcript[0]?.thetaBefore).toBeCloseTo(0, 6);
    expect(transcript.map((entry) => entry.position)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('tightens the standard error as the session proceeds', () => {
    const s = session({ stopping: fixedLength(25), estimator: eapEstimator() });
    const rng = createRng(11);
    const { transcript } = s.run((item) => simulateCategory(item, 0.3, rng));
    const first = transcript[0]?.standardErrorAfter as number;
    const last = transcript[transcript.length - 1]?.standardErrorAfter as number;
    expect(last).toBeLessThan(first);
    expect(last).toBeLessThan(0.35);
  });

  it('stops when the precision target is met, before the ceiling', () => {
    const s = session({ stopping: precisionTarget(0.4, { minimum: 5, maximum: 40 }) });
    const rng = createRng(23);
    const snapshot = s.run((item) => simulateCategory(item, 0.2, rng));
    expect(snapshot.transcript.length).toBeGreaterThanOrEqual(5);
    expect(snapshot.transcript.length).toBeLessThan(40);
    expect(snapshot.standardError).toBeLessThanOrEqual(0.4);
    expect(snapshot.stopReason?.rule).toMatch(/standard-error-below/);
  });

  it('honours the minimum length even when two answers look precise', () => {
    const s = session({ stopping: precisionTarget(5, { minimum: 6, maximum: 40 }) });
    const rng = createRng(2);
    const snapshot = s.run((item) => simulateCategory(item, 0, rng));
    // The target is absurdly loose, so only the floor keeps the test going.
    expect(snapshot.transcript).toHaveLength(6);
  });

  it('stops when the pool runs out', () => {
    const small = new ItemPool(bankItems(4));
    const s = new AdaptiveSession({
      pool: small,
      selector: maximumInformationSelector(),
      stopping: fixedLength(50),
      seed: 1,
    });
    const rng = createRng(1);
    const snapshot = s.run((item) => simulateCategory(item, 0, rng));
    expect(snapshot.transcript).toHaveLength(4);
    expect(snapshot.stopReason?.rule).toBe('pool-exhausted');
  });

  it('rejects a response when no item is pending', () => {
    const s = session();
    expect(() => s.submit(1)).toThrow(/no item is awaiting a response/);
  });

  it('rejects a second nextItem before the first is answered', () => {
    const s = session();
    s.nextItem();
    expect(() => s.nextItem()).toThrow(/already awaiting a response/);
  });

  it('rejects a response outside {0, 1}', () => {
    const s = session();
    s.nextItem();
    expect(() => s.submit(2 as Response)).toThrow(RangeError);
  });

  it('returns null from nextItem once finished', () => {
    const s = session({ stopping: fixedLength(2) });
    const rng = createRng(1);
    s.run((item) => simulateCategory(item, 0, rng));
    expect(s.nextItem()).toBeNull();
  });

  it('exposes administered items and a serialisable snapshot', () => {
    const s = session({ stopping: fixedLength(5) });
    const rng = createRng(8);
    s.run((item) => simulateCategory(item, 0.9, rng));
    expect(s.administered.map((item) => item.id)).toEqual(s.snapshot().administeredIds);
    expect(() => JSON.parse(JSON.stringify(s.snapshot()))).not.toThrow();
  });

  it('recovers a known ability over a long adaptive test', () => {
    for (const trueTheta of [-1.2, 0, 1.2]) {
      const s = session({ stopping: fixedLength(35), estimator: wleEstimator() });
      const rng = createRng(1000 + trueTheta * 10);
      const snapshot = s.run((item) => simulateCategory(item, trueTheta, rng));
      expect(Math.abs(snapshot.theta - trueTheta)).toBeLessThan(3 * snapshot.standardError);
      expect(snapshot.standardError).toBeLessThan(0.4);
    }
  });

  it('measures more precisely than a linear test of the same length', () => {
    // This is the whole claim of adaptive testing, so it is worth asserting
    // rather than assuming: 20 adaptively chosen items beat 20 items spread
    // evenly across the bank, at the same true ability.
    const length = 20;
    const trueTheta = 1.1;

    const adaptive = session({ stopping: fixedLength(length) });
    const snapshot = adaptive.run((item) => simulateCategory(item, trueTheta, createRng(31)));

    const linear = linspace(-3, 3, length).map((b, index) => makeItem(`lin-${index}`, twoPL(1.3, b)));
    const linearError = standardError(testInformation(linear, snapshot.theta));

    expect(snapshot.standardError).toBeLessThan(linearError);
    // And the adaptive test concentrates its items near the candidate.
    const chosen = adaptive.administered.map((item) => itemLocation(item));
    expect(mean(chosen)).toBeGreaterThan(0.5);
  });

  it('reaches a stated precision target and stops there', () => {
    const s = session({ stopping: precisionTarget(0.35, { minimum: 5, maximum: 60 }) });
    const rng = createRng(31);
    const snapshot = s.run((item) => simulateCategory(item, 0.4, rng));
    expect(snapshot.standardError).toBeLessThanOrEqual(0.35);
    expect(snapshot.stopReason?.rule).toMatch(/standard-error-below/);
    // Stopping is not delayed past the point the target is met.
    const beforeLast = snapshot.transcript[snapshot.transcript.length - 2];
    expect(beforeLast?.standardErrorAfter).toBeGreaterThan(0.35);
  });
});

describe('session estimators', () => {
  it('the hybrid switches from EAP to WLE once the pattern is mixed', () => {
    const estimator = hybridEstimator();
    const items = bankItems(4);
    expect(estimator([]).method).toBe('eap');
    expect(estimator([{ item: items[0] as Item, response: 1 }]).method).toBe('eap');
    expect(
      estimator([
        { item: items[0] as Item, response: 1 },
        { item: items[1] as Item, response: 1 },
      ]).method,
    ).toBe('eap');
    expect(
      estimator([
        { item: items[0] as Item, response: 1 },
        { item: items[1] as Item, response: 0 },
      ]).method,
    ).toBe('wle');
  });

  it('each estimator is defined for the empty pattern', () => {
    for (const estimator of [eapEstimator(), mleEstimator(), wleEstimator(), hybridEstimator()]) {
      const estimate = estimator([]);
      expect(Number.isFinite(estimate.theta)).toBe(true);
      expect(estimate.theta).toBeCloseTo(0, 6);
    }
  });

  it('a session driven by maximum likelihood still starts from a finite estimate', () => {
    const s = new AdaptiveSession({
      pool,
      selector: maximumInformationSelector(),
      stopping: fixedLength(8),
      estimator: mleEstimator(),
      seed: 4,
    });
    const rng = createRng(6);
    const snapshot = s.run((item) => simulateCategory(item, 1.5, rng));
    expect(Number.isFinite(snapshot.theta)).toBe(true);
    expect(snapshot.transcript).toHaveLength(8);
  });
});
