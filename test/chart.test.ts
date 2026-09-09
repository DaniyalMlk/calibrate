import { describe, expect, it } from 'vitest';

import { clampTo, linearScale, niceTicks } from '../web/app/chart/scale.js';
import { describeParameters, fixed, interval, percent, signed } from '../web/app/format.js';

/**
 * The pure parts of the web interface: axis ticks, scales and number
 * formatting. None of it touches the DOM, and all of it is the sort of code
 * whose failures are visible in a screenshot but hard to spot in one.
 */
describe('niceTicks', () => {
  it('lands on the numbers a reader would have chosen', () => {
    expect(niceTicks([0, 1], 5)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(niceTicks([0, 10], 6)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(niceTicks([0, 2], 5)).toEqual([0, 0.5, 1, 1.5, 2]);
  });

  it('covers the locked ability domain at every tick target', () => {
    // The regression this exists for. Over [-4, 4] a target of four ticks gives
    // an ideal spacing of 2.67; rounding that up to the next nice value gives 5,
    // whose only multiple inside the domain is zero — a one-label axis, which is
    // exactly what the narrow layout used to render.
    for (const target of [3, 4, 5, 6, 7, 8]) {
      const ticks = niceTicks([-4, 4], target);
      expect(ticks.length).toBeGreaterThanOrEqual(3);
      expect(ticks).toContain(0);
    }
  });

  it('never emits a tick outside the domain', () => {
    const domains: readonly (readonly [number, number])[] = [
      [-4, 4],
      [0, 1],
      [0, 0.85],
      [-3, 3],
      [0.15, 7.3],
      [-0.004, 0.004],
      [1000, 9500],
    ];

    for (const domain of domains) {
      for (const target of [3, 5, 7]) {
        for (const tick of niceTicks(domain, target)) {
          expect(tick).toBeGreaterThanOrEqual(domain[0] - 1e-9);
          expect(tick).toBeLessThanOrEqual(domain[1] + 1e-9);
        }
      }
    }
  });

  it('emits strictly increasing, evenly spaced ticks', () => {
    for (const domain of [[-4, 4], [0, 1], [0.15, 7.3]] as const) {
      const ticks = niceTicks(domain, 6);
      const step = (ticks[1] as number) - (ticks[0] as number);
      for (let i = 1; i < ticks.length; i += 1) {
        expect((ticks[i] as number) - (ticks[i - 1] as number)).toBeCloseTo(step, 9);
      }
    }
  });

  it('does not leak binary floating-point error into a label', () => {
    // 0.1 + 0.2 territory: an unrounded accumulator produces 0.30000000000000004,
    // which renders as an axis label exactly that long.
    for (const tick of niceTicks([0, 1], 11)) {
      expect(String(tick).length).toBeLessThanOrEqual(4);
    }
  });

  it('degrades safely on a domain with no width', () => {
    expect(niceTicks([2, 2], 5)).toEqual([2]);
    expect(niceTicks([5, 1], 5)).toEqual([5]);
    expect(niceTicks([0, 1], 1)).toEqual([0]);
  });
});

describe('linearScale', () => {
  it('maps the domain onto the range and back', () => {
    const scale = linearScale([-4, 4], [40, 440]);

    expect(scale.to(-4)).toBeCloseTo(40, 9);
    expect(scale.to(4)).toBeCloseTo(440, 9);
    expect(scale.to(0)).toBeCloseTo(240, 9);
    expect(scale.from(240)).toBeCloseTo(0, 9);
    expect(scale.from(scale.to(1.37))).toBeCloseTo(1.37, 9);
  });

  it('handles an inverted range, which every y axis has', () => {
    const scale = linearScale([0, 1], [300, 20]);

    expect(scale.to(0)).toBeCloseTo(300, 9);
    expect(scale.to(1)).toBeCloseTo(20, 9);
    expect(scale.from(160)).toBeCloseTo(0.5, 9);
  });

  it('does not divide by zero on a degenerate domain', () => {
    const scale = linearScale([3, 3], [0, 100]);

    expect(Number.isFinite(scale.to(3))).toBe(true);
    expect(scale.from(50)).toBe(3);
  });
});

describe('clampTo', () => {
  it('clamps to the closed interval', () => {
    expect(clampTo(5, 0, 1)).toBe(1);
    expect(clampTo(-5, 0, 1)).toBe(0);
    expect(clampTo(0.4, 0, 1)).toBe(0.4);
  });
});

describe('fixed', () => {
  it('never renders a negative zero', () => {
    // A posterior mean of -1e-17 is zero. Rendering it "-0.00" advertises a sign
    // the number has not earned, and it is the first thing a reader notices.
    expect(fixed(-1e-17)).toBe('0.00');
    expect(fixed(-0)).toBe('0.00');
    expect(fixed(-0.0004, 2)).toBe('0.00');
  });

  it('uses a typographic minus, which aligns with digits', () => {
    expect(fixed(-1.5)).toBe('−1.50');
    expect(fixed(1.5)).toBe('1.50');
  });

  it('renders a non-finite value as a dash rather than as "NaN"', () => {
    expect(fixed(Number.NaN)).toBe('—');
    expect(fixed(Infinity)).toBe('—');
    expect(fixed(-Infinity)).toBe('—');
  });

  it('respects the requested precision', () => {
    expect(fixed(1.23456, 3)).toBe('1.235');
    expect(fixed(1.23456, 0)).toBe('1');
  });
});

describe('signed', () => {
  it('always carries a sign', () => {
    expect(signed(0.8)).toBe('+0.80');
    expect(signed(-0.8)).toBe('−0.80');
  });

  it('renders a value that rounds to zero as positive zero', () => {
    expect(signed(-1e-9)).toBe('+0.00');
    expect(signed(0)).toBe('+0.00');
  });

  it('renders a non-finite value as a dash', () => {
    expect(signed(Number.NaN)).toBe('—');
  });
});

describe('interval and percent', () => {
  it('reads the way an interval is written in a report', () => {
    expect(interval(-1.96, 1.96)).toBe('−1.96 to 1.96');
  });

  it('formats a proportion as a percentage', () => {
    expect(percent(0.95)).toBe('95%');
    expect(percent(0.0512, 2)).toBe('5.12%');
    expect(percent(Number.NaN)).toBe('—');
  });
});

describe('describeParameters', () => {
  it('reports only the parameters that are doing something', () => {
    // The engine stores every item as a 4PL, so a Rasch item still carries a
    // c of 0 and a d of 1. Printing those would suggest they were estimated.
    expect(describeParameters({ a: 1, b: 0.4, c: 0, d: 1 })).toBe('a = 1.00   b = +0.40');
    expect(describeParameters({ a: 1.42, b: -0.31, c: 0.25, d: 1 })).toBe(
      'a = 1.42   b = −0.31   c = 0.25',
    );
    expect(describeParameters({ a: 1.42, b: -0.31, c: 0.25, d: 0.98 })).toBe(
      'a = 1.42   b = −0.31   c = 0.25   d = 0.98',
    );
  });
});
