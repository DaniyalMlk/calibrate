import { describe, expect, it } from 'vitest';
import { bracketSignChange, safeguardedRoot } from '../src/core/root.js';

describe('safeguardedRoot', () => {
  it('finds a simple polynomial root', () => {
    const result = safeguardedRoot(
      (x) => x * x - 2,
      (x) => 2 * x,
      0,
      2,
    );
    expect(result.root).toBeCloseTo(Math.SQRT2, 12);
    expect(result.converged).toBe(true);
  });

  it('converges in few iterations on a well-behaved function', () => {
    const result = safeguardedRoot(
      (x) => Math.cos(x) - x,
      (x) => -Math.sin(x) - 1,
      0,
      1,
    );
    // Dottie number, the unique real fixed point of cosine.
    expect(result.root).toBeCloseTo(0.7390851332151607, 12);
    expect(result.iterations).toBeLessThan(15);
  });

  it('falls back to bisection where a Newton step would escape the bracket', () => {
    // A near-flat region at the left of the bracket sends a raw Newton step far
    // outside it; the safeguard has to catch that.
    const f = (x: number) => Math.atan(x - 5) - 0.5;
    const df = (x: number) => 1 / (1 + (x - 5) * (x - 5));
    const result = safeguardedRoot(f, df, -1000, 1000);
    expect(result.converged).toBe(true);
    expect(result.root).toBeCloseTo(5 + Math.tan(0.5), 8);
    expect(result.bisectionSteps).toBeGreaterThan(0);
  });

  it('returns an endpoint that is already a root', () => {
    const atLo = safeguardedRoot(
      (x) => x,
      () => 1,
      0,
      1,
    );
    expect(atLo.root).toBe(0);
    expect(atLo.iterations).toBe(0);

    const atHi = safeguardedRoot(
      (x) => x - 1,
      () => 1,
      0,
      1,
    );
    expect(atHi.root).toBe(1);
  });

  it('accepts a bracket given in either orientation', () => {
    const ascending = safeguardedRoot(
      (x) => x * x - 2,
      (x) => 2 * x,
      0,
      2,
    );
    const descending = safeguardedRoot(
      (x) => 2 - x * x,
      (x) => -2 * x,
      0,
      2,
    );
    expect(descending.root).toBeCloseTo(ascending.root, 10);
  });

  it('rejects a bracket that does not straddle a root', () => {
    expect(() =>
      safeguardedRoot(
        (x) => x * x + 1,
        (x) => 2 * x,
        -1,
        1,
      ),
    ).toThrow(/does not straddle/);
  });

  it('reports non-convergence rather than pretending, when starved of iterations', () => {
    const result = safeguardedRoot(
      (x) => x * x - 2,
      (x) => 2 * x,
      0,
      2,
      { maxIterations: 2, tolerance: 1e-15 },
    );
    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(2);
  });
});

describe('bracketSignChange', () => {
  it('locates an interval containing a sign change', () => {
    const bracket = bracketSignChange((x) => x - 0.3, -1, 1, 8);
    expect(bracket).not.toBeNull();
    const [lo, hi] = bracket as readonly [number, number];
    expect(lo).toBeLessThanOrEqual(0.3);
    expect(hi).toBeGreaterThanOrEqual(0.3);
  });

  it('returns null when the function keeps its sign throughout', () => {
    expect(bracketSignChange((x) => x * x + 1, -5, 5, 32)).toBeNull();
  });

  it('finds an exact zero sitting on a grid point', () => {
    expect(bracketSignChange((x) => x, -1, 1, 4)).toEqual([0, 0]);
  });

  it('covers the right endpoint exactly', () => {
    // A root at the very top of the range must still be found; floating point
    // accumulation in the scan must not step past it.
    const bracket = bracketSignChange((x) => x - 1, -1, 1, 10);
    expect(bracket).toEqual([1, 1]);
  });

  it('rejects a non-positive step count', () => {
    expect(() => bracketSignChange((x) => x, 0, 1, 0)).toThrow(RangeError);
  });
});
