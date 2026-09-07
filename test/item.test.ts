import { describe, expect, it } from 'vitest';
import {
  describeModel,
  fourPL,
  makeItem,
  metricScale,
  onePL,
  rasch,
  threePL,
  twoPL,
  validateItemParameters,
} from '../src/models/item.js';

describe('item constructors', () => {
  it('pin the parameters each model holds fixed', () => {
    expect(rasch(0.5)).toMatchObject({ a: 1, b: 0.5, c: 0, d: 1 });
    expect(onePL(0.5, 1.4)).toMatchObject({ a: 1.4, b: 0.5, c: 0, d: 1 });
    expect(twoPL(1.2, -0.3)).toMatchObject({ a: 1.2, b: -0.3, c: 0, d: 1 });
    expect(threePL(1.2, -0.3, 0.2)).toMatchObject({ a: 1.2, b: -0.3, c: 0.2, d: 1 });
    expect(fourPL(1.2, -0.3, 0.2, 0.95)).toMatchObject({ a: 1.2, b: -0.3, c: 0.2, d: 0.95 });
  });

  it('default to the logistic metric and expose the scale', () => {
    expect(twoPL(1, 0).metric).toBe('logistic');
    expect(twoPL(1, 0, 'normal').metric).toBe('normal');
    expect(metricScale('logistic')).toBe(1);
    expect(metricScale('normal')).toBeCloseTo(1.702, 12);
  });

  it('returns frozen parameters', () => {
    const p = twoPL(1, 0);
    expect(Object.isFrozen(p)).toBe(true);
  });
});

describe('parameter validation', () => {
  const base = { a: 1, b: 0, c: 0, d: 1, metric: 'logistic' as const };

  it('names the offending field', () => {
    expect(() => validateItemParameters({ ...base, a: 0 })).toThrow(/discrimination \(a\)/);
    expect(() => validateItemParameters({ ...base, b: Number.NaN })).toThrow(/difficulty \(b\)/);
    expect(() => validateItemParameters({ ...base, c: -0.1 })).toThrow(/lower asymptote \(c\)/);
    expect(() => validateItemParameters({ ...base, d: 1.2 })).toThrow(/upper asymptote \(d\)/);
  });

  it('rejects non-positive and implausibly large discriminations', () => {
    expect(() => validateItemParameters({ ...base, a: -1 })).toThrow(RangeError);
    expect(() => validateItemParameters({ ...base, a: 25 })).toThrow(/step function/);
    expect(() => validateItemParameters({ ...base, a: 20 })).not.toThrow();
  });

  it('rejects difficulties outside the admissible window', () => {
    expect(() => validateItemParameters({ ...base, b: 21 })).toThrow(RangeError);
    expect(() => validateItemParameters({ ...base, b: -21 })).toThrow(RangeError);
  });

  it('rejects asymptotes that cross', () => {
    expect(() => validateItemParameters({ ...base, c: 0.6, d: 0.5 })).toThrow(/strictly below/);
    expect(() => validateItemParameters({ ...base, c: 0.5, d: 0.5 })).toThrow(/strictly below/);
    expect(() => validateItemParameters({ ...base, c: 0.4, d: 0.5 })).not.toThrow();
  });

  it('rejects a guessing parameter of one, which would make the item uninformative', () => {
    expect(() => validateItemParameters({ ...base, c: 1 })).toThrow(RangeError);
  });

  it('rejects an unknown metric', () => {
    // Deliberately bypassing the type system to model a bad JSON payload.
    const bad = { ...base, metric: 'probit' } as unknown as Parameters<
      typeof validateItemParameters
    >[0];
    expect(() => validateItemParameters(bad)).toThrow(/metric/);
  });
});

describe('makeItem', () => {
  it('carries identity and metadata', () => {
    const item = makeItem('q-1', twoPL(1.1, 0.2), { domain: 'algorithms', tags: ['graphs'] });
    expect(item.id).toBe('q-1');
    expect(item.domain).toBe('algorithms');
    expect(item.tags).toEqual(['graphs']);
    expect(Object.isFrozen(item)).toBe(true);
  });

  it('omits absent metadata rather than storing undefined', () => {
    const item = makeItem('q-2', twoPL(1, 0));
    expect('domain' in item).toBe(false);
    expect('tags' in item).toBe(false);
  });

  it('rejects an empty id', () => {
    expect(() => makeItem('', twoPL(1, 0))).toThrow(/non-empty/);
  });

  it('validates the parameters it is handed', () => {
    expect(() =>
      makeItem('q-3', { a: 1, b: 0, c: 0.9, d: 0.5, metric: 'logistic' }),
    ).toThrow(RangeError);
  });
});

describe('describeModel', () => {
  it('names the most constrained model consistent with the parameters', () => {
    expect(describeModel(rasch(0))).toBe('1PL');
    expect(describeModel(twoPL(1.3, 0))).toBe('2PL');
    expect(describeModel(threePL(1.3, 0, 0.2))).toBe('3PL');
    expect(describeModel(fourPL(1.3, 0, 0.2, 0.98))).toBe('4PL');
  });

  it('reports a 3PL item with no guessing as a 2PL item', () => {
    expect(describeModel(threePL(1.3, 0, 0))).toBe('2PL');
  });
});
