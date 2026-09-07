import { requireFinite } from './numeric.js';

/**
 * A quadrature rule: nodes on the ability scale paired with weights.
 *
 * When the rule represents a prior distribution the weights sum to one, so a
 * weighted sum over the nodes approximates an expectation directly.
 */
export interface QuadratureRule {
  readonly nodes: readonly number[];
  readonly weights: readonly number[];
}

/** Newton tolerance for locating Hermite roots. */
const HERMITE_TOLERANCE = 3e-14;
const HERMITE_MAX_ITERATIONS = 64;
/** pi^(-1/4), the normalisation of the zeroth Hermite function. */
const PI_TO_MINUS_QUARTER = 0.7511255444649425;

/**
 * Gauss–Hermite quadrature for `integral over R of exp(-x^2) f(x) dx`.
 *
 * Nodes are the roots of the degree-`n` physicists' Hermite polynomial, found by
 * Newton iteration on the three-term recurrence in its normalised form. The
 * normalised recurrence is used rather than the raw one because raw Hermite
 * values overflow a double by about n = 30, and quadrature at order 40 or more
 * is routine here.
 *
 * The initial guesses are the standard asymptotic approximations to the largest
 * roots plus the observed spacing of the rest; each converges in a handful of
 * iterations.
 */
export function gaussHermite(n: number): QuadratureRule {
  if (!Number.isInteger(n) || n < 2) {
    throw new RangeError(`gaussHermite: order must be an integer >= 2, received ${n}`);
  }

  const nodes = new Array<number>(n).fill(0);
  const weights = new Array<number>(n).fill(0);
  // The initial guesses are expressed in terms of previously located *positive*
  // roots, so they are kept separately from the symmetric node array.
  const positiveRoots = new Array<number>(n).fill(0);
  const half = Math.floor((n + 1) / 2);
  let z = 0;
  let pp = 0;

  for (let i = 0; i < half; i += 1) {
    if (i === 0) {
      z = Math.sqrt(2 * n + 1) - 1.85575 * Math.pow(2 * n + 1, -1 / 6);
    } else if (i === 1) {
      z -= (1.14 * Math.pow(n, 0.426)) / z;
    } else if (i === 2) {
      z = 1.86 * z - 0.86 * (positiveRoots[0] as number);
    } else if (i === 3) {
      z = 1.91 * z - 0.91 * (positiveRoots[1] as number);
    } else {
      z = 2 * z - (positiveRoots[i - 2] as number);
    }

    let converged = false;
    for (let iteration = 0; iteration < HERMITE_MAX_ITERATIONS; iteration += 1) {
      let p1 = PI_TO_MINUS_QUARTER;
      let p2 = 0;
      for (let j = 1; j <= n; j += 1) {
        const p3 = p2;
        p2 = p1;
        p1 = z * Math.sqrt(2 / j) * p2 - Math.sqrt((j - 1) / j) * p3;
      }
      // Derivative of the normalised polynomial at z.
      pp = Math.sqrt(2 * n) * p2;
      const previous = z;
      z = previous - p1 / pp;
      if (Math.abs(z - previous) <= HERMITE_TOLERANCE) {
        converged = true;
        break;
      }
    }
    if (!converged) {
      throw new Error(`gaussHermite: root ${i} failed to converge at order ${n}`);
    }

    positiveRoots[i] = z;
    nodes[i] = -z;
    nodes[n - 1 - i] = z;
    const weight = 2 / (pp * pp);
    weights[i] = weight;
    weights[n - 1 - i] = weight;
  }

  return { nodes, weights };
}

/**
 * A Gauss–Hermite rule transformed to represent a normal prior.
 *
 * Substituting `theta = mean + sd * sqrt(2) * x` turns
 * `integral f(theta) N(theta; mean, sd) dtheta` into
 * `(1 / sqrt(pi)) * sum_i w_i f(mean + sd sqrt(2) x_i)`, so the returned weights
 * sum to one (to quadrature accuracy) and a weighted sum over the nodes is an
 * expectation under the prior.
 */
export function normalGaussHermiteRule(mean: number, sd: number, points = 41): QuadratureRule {
  requireFinite(mean, 'prior mean');
  requireFinite(sd, 'prior standard deviation');
  if (sd <= 0) {
    throw new RangeError(`prior standard deviation must be positive, received ${sd}`);
  }
  const base = gaussHermite(points);
  const scale = sd * Math.SQRT2;
  const normaliser = 1 / Math.sqrt(Math.PI);
  return {
    nodes: base.nodes.map((x) => mean + scale * x),
    weights: base.weights.map((w) => w * normaliser),
  };
}

/**
 * A fixed rectangular grid weighted by a normal density and renormalised to
 * sum to one.
 *
 * Less accurate than Gauss–Hermite at equal cost, but it is what most published
 * IRT software uses — commonly 41 points over `mean +/- 4 sd` — and matching that
 * convention is what makes results comparable against a reference
 * implementation. The renormalisation absorbs the mass outside the truncation.
 */
export function normalGridRule(mean: number, sd: number, points = 41, halfWidth = 4): QuadratureRule {
  requireFinite(mean, 'prior mean');
  requireFinite(sd, 'prior standard deviation');
  if (sd <= 0) {
    throw new RangeError(`prior standard deviation must be positive, received ${sd}`);
  }
  if (!Number.isInteger(points) || points < 2) {
    throw new RangeError(`normalGridRule: points must be an integer >= 2, received ${points}`);
  }
  if (halfWidth <= 0) {
    throw new RangeError(`normalGridRule: halfWidth must be positive, received ${halfWidth}`);
  }

  const lo = mean - halfWidth * sd;
  const hi = mean + halfWidth * sd;
  const step = (hi - lo) / (points - 1);
  const nodes = new Array<number>(points);
  const raw = new Array<number>(points);
  let total = 0;
  for (let i = 0; i < points; i += 1) {
    const theta = i === points - 1 ? hi : lo + step * i;
    const z = (theta - mean) / sd;
    const density = Math.exp(-0.5 * z * z);
    nodes[i] = theta;
    raw[i] = density;
    total += density;
  }
  return { nodes, weights: raw.map((d) => d / total) };
}

/** Integrate `f` against a rule: `sum_k w_k f(theta_k)`. */
export function integrate(rule: QuadratureRule, f: (theta: number) => number): number {
  if (rule.nodes.length !== rule.weights.length) {
    throw new RangeError('quadrature rule has mismatched node and weight counts');
  }
  let total = 0;
  for (let i = 0; i < rule.nodes.length; i += 1) {
    total += (rule.weights[i] as number) * f(rule.nodes[i] as number);
  }
  return total;
}
