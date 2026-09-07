/**
 * Deterministic pseudo-random number generation.
 *
 * Simulation studies and adaptive sessions both need randomness, and both need
 * to be replayable: an exposure-control policy that behaves differently on every
 * run cannot be compared against another one, and a session that cannot be
 * replayed cannot be audited after a candidate disputes their score. `Math.random`
 * offers neither, so the engine carries its own seeded generator everywhere.
 */

export interface Rng {
  /** Uniform on [0, 1). */
  next(): number;
  /** Standard normal deviate. */
  nextNormal(): number;
  /** Uniform integer in [0, n). */
  nextInt(n: number): number;
}

/**
 * A 32-bit `splitmix`-derived generator (mulberry32).
 *
 * Chosen over a linear congruential generator because the low bits of an LCG
 * are notoriously non-random, and simulation code tends to reach for exactly
 * those bits via a modulo. Fast, tiny, and good enough for Monte Carlo work of
 * this size; it is not a cryptographic generator and is not used as one.
 */
export function createRng(seed: number): Rng {
  if (!Number.isFinite(seed)) {
    throw new RangeError(`createRng: seed must be a finite number, received ${String(seed)}`);
  }
  let state = Math.trunc(seed) >>> 0;
  // A zero state is a fixed point for some generators of this family; nudge it.
  if (state === 0) state = 0x9e3779b9;

  let spareNormal: number | null = null;

  const next = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  return {
    next,
    /**
     * Box–Muller, generating deviates in pairs and holding the spare.
     *
     * The polar form is usually preferred for speed, but it rejects samples,
     * which means the number of uniforms consumed depends on their values. That
     * breaks the property this module exists for: with the transform form, a
     * given seed consumes a fixed number of uniforms and replays identically.
     */
    nextNormal(): number {
      if (spareNormal !== null) {
        const value = spareNormal;
        spareNormal = null;
        return value;
      }
      // Guard against log(0); next() can return exactly 0.
      let u1 = next();
      while (u1 <= Number.MIN_VALUE) u1 = next();
      const u2 = next();
      const radius = Math.sqrt(-2 * Math.log(u1));
      const angle = 2 * Math.PI * u2;
      spareNormal = radius * Math.sin(angle);
      return radius * Math.cos(angle);
    },
    nextInt(n: number): number {
      if (!Number.isInteger(n) || n <= 0) {
        throw new RangeError(`nextInt: n must be a positive integer, received ${n}`);
      }
      return Math.floor(next() * n);
    },
  };
}

/** Draw a normal deviate with the given mean and standard deviation. */
export function normalDeviate(rng: Rng, mean: number, standardDeviation: number): number {
  if (standardDeviation < 0) {
    throw new RangeError(
      `normalDeviate: standard deviation must be non-negative, received ${standardDeviation}`,
    );
  }
  return mean + standardDeviation * rng.nextNormal();
}
