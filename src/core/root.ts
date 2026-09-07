/**
 * A safeguarded one-dimensional root finder.
 *
 * Newton's method converges quadratically when it converges and diverges
 * spectacularly when it does not — and an IRT score function under the 3PL is
 * exactly the kind of function that tempts it into a runaway step. Keeping a
 * bracket around the root and falling back to bisection whenever a Newton step
 * would leave that bracket gives Newton's speed with bisection's guarantee.
 */

export interface RootResult {
  /** The located root. */
  readonly root: number;
  /** Whether the tolerance was met before the iteration budget ran out. */
  readonly converged: boolean;
  /** Iterations actually performed. */
  readonly iterations: number;
  /** How many of those iterations fell back to bisection. */
  readonly bisectionSteps: number;
}

export interface RootOptions {
  /** Absolute tolerance on the width of the bracket. Default 1e-10. */
  readonly tolerance?: number;
  /** Maximum iterations before giving up. Default 100. */
  readonly maxIterations?: number;
}

/**
 * Find the root of `f` known to lie in `[lo, hi]`, using its derivative `df`.
 *
 * `f(lo)` and `f(hi)` must have opposite signs; a bracket that does not
 * straddle a sign change is a programming error and throws rather than
 * returning a plausible-looking wrong answer.
 */
export function safeguardedRoot(
  f: (x: number) => number,
  df: (x: number) => number,
  lo: number,
  hi: number,
  options: RootOptions = {},
): RootResult {
  const tolerance = options.tolerance ?? 1e-10;
  const maxIterations = options.maxIterations ?? 100;

  let fLo = f(lo);
  let fHi = f(hi);

  if (fLo === 0) return { root: lo, converged: true, iterations: 0, bisectionSteps: 0 };
  if (fHi === 0) return { root: hi, converged: true, iterations: 0, bisectionSteps: 0 };
  if (fLo * fHi > 0) {
    throw new RangeError(
      `safeguardedRoot: bracket [${lo}, ${hi}] does not straddle a root ` +
        `(f(lo) = ${fLo}, f(hi) = ${fHi})`,
    );
  }

  // Orient the bracket so that f is negative at `low` and positive at `high`.
  let low = fLo < 0 ? lo : hi;
  let high = fLo < 0 ? hi : lo;

  let x = 0.5 * (lo + hi);
  let step = Math.abs(hi - lo);
  let previousStep = step;
  let fx = f(x);
  let dfx = df(x);
  let bisectionSteps = 0;

  for (let iteration = 1; iteration <= maxIterations; iteration += 1) {
    const newtonOutOfBracket =
      ((x - high) * dfx - fx) * ((x - low) * dfx - fx) > 0;
    const newtonTooSlow = Math.abs(2 * fx) > Math.abs(previousStep * dfx);

    previousStep = step;
    if (newtonOutOfBracket || newtonTooSlow || dfx === 0) {
      step = 0.5 * (high - low);
      x = low + step;
      bisectionSteps += 1;
    } else {
      step = fx / dfx;
      x -= step;
    }

    if (Math.abs(step) < tolerance) {
      return { root: x, converged: true, iterations: iteration, bisectionSteps };
    }

    fx = f(x);
    dfx = df(x);
    if (fx < 0) low = x;
    else high = x;
  }

  return { root: x, converged: false, iterations: maxIterations, bisectionSteps };
}

/**
 * Scan `[lo, hi]` on a uniform grid and return the first sub-interval on which
 * `f` changes sign, or `null` if it never does.
 *
 * Used to bracket a root before handing it to `safeguardedRoot`. A grid scan is
 * cheap relative to evaluating a likelihood and it is the only honest way to
 * bracket a function that may not be monotone.
 */
export function bracketSignChange(
  f: (x: number) => number,
  lo: number,
  hi: number,
  steps = 64,
): readonly [number, number] | null {
  if (!Number.isInteger(steps) || steps < 1) {
    throw new RangeError(`bracketSignChange: steps must be a positive integer, received ${steps}`);
  }
  const width = (hi - lo) / steps;
  let left = lo;
  let fLeft = f(left);
  if (fLeft === 0) return [left, left];
  for (let i = 1; i <= steps; i += 1) {
    const right = i === steps ? hi : lo + width * i;
    const fRight = f(right);
    if (fRight === 0) return [right, right];
    if (fLeft * fRight < 0) return [left, right];
    left = right;
    fLeft = fRight;
  }
  return null;
}
