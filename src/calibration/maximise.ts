import { clamp } from '../core/numeric.js';
import type { QuadratureRule } from '../core/quadrature.js';
import { pointProbability, type ItemPoint } from './expected.js';
import type { CalibrationModel } from './jmle.js';

/** Bounds and stopping rules for one maximisation step. */
export interface MaximiseOptions {
  readonly model: CalibrationModel;
  readonly discriminationBounds: readonly [number, number];
  readonly difficultyBounds: readonly [number, number];
  /** Largest absolute Newton step accepted before damping. Default 0.5. */
  readonly maxStep?: number;
  /** Newton iterations inside one maximisation step. Default 40. */
  readonly maxIterations?: number;
  /** Gradient norm treated as stationary. Default 1e-10. */
  readonly tolerance?: number;
}

export interface MaximiseResult {
  readonly item: ItemPoint;
  /** Whether the Newton iteration reached a stationary point inside the bounds. */
  readonly stationary: boolean;
  /** Whether either parameter finished pinned to a bound. */
  readonly atBound: boolean;
  readonly iterations: number;
}

/** The expected-count log-likelihood of one item, and its first two derivatives. */
export interface CountDerivatives {
  readonly value: number;
  readonly gradientA: number;
  readonly gradientB: number;
  readonly hessianAA: number;
  readonly hessianBB: number;
  readonly hessianAB: number;
}

/**
 * The log-likelihood an item's expected counts imply, with derivatives.
 *
 * Reading the counts as a dataset: at node `k`, `answered[k]` respondents of
 * ability `nodes[k]` sat the item and `correct[k]` of them passed. That is a
 * binomial log-likelihood with a logistic mean, i.e. a weighted logistic
 * regression of the response on ability, with the counts as weights. Every
 * person parameter in the original problem has been absorbed into those weights.
 */
export function countDerivatives(
  item: ItemPoint,
  rule: QuadratureRule,
  answered: Float64Array,
  correct: Float64Array,
): CountDerivatives {
  const { discrimination: a, difficulty: b } = item;
  let value = 0;
  let gradientA = 0;
  let gradientB = 0;
  let hessianAA = 0;
  let hessianBB = 0;
  let hessianAB = 0;

  for (let k = 0; k < rule.nodes.length; k += 1) {
    const n = answered[k] as number;
    if (n <= 0) continue;
    const r = correct[k] as number;
    const theta = rule.nodes[k] as number;
    const d = theta - b;
    const p = pointProbability(item, theta);
    const w = p * (1 - p);
    // The residual in count units: observed successes minus expected successes.
    const residual = r - n * p;

    // Guarded logs: a node where the model is certain and the counts disagree
    // would otherwise contribute -Infinity and destroy the comparison that the
    // convergence test makes between successive iterations.
    value += r * safeLog(p) + (n - r) * safeLog(1 - p);
    gradientA += residual * d;
    gradientB -= a * residual;
    hessianAA -= n * w * d * d;
    hessianBB -= a * a * n * w;
    hessianAB += a * n * w * d - residual;
  }

  return { value, gradientA, gradientB, hessianAA, hessianBB, hessianAB };
}

/** `log(x)` floored well below any probability the bounds can produce. */
function safeLog(x: number): number {
  return x > 1e-300 ? Math.log(x) : -690.7755278982137;
}

/**
 * Maximise one item's expected-count log-likelihood.
 *
 * Under the Rasch model the discrimination is fixed at one and what is left is
 * strictly concave in the difficulty, so Newton converges from anywhere in a
 * handful of steps. Under the 2PL the surface in `(a, b)` is not concave
 * everywhere — far from the optimum the cross term can dominate — so the Newton
 * step is taken only where the Hessian is genuinely negative definite, and
 * otherwise the step falls back to the gradient direction. Both are damped to a
 * maximum length, which costs an iteration or two near a flat start and prevents
 * a single wild step from throwing a discrimination onto its bound and stranding
 * it there.
 *
 * Solving each item to convergence rather than taking one step is what makes
 * this expectation-maximisation rather than a gradient method with a
 * probabilistic flavour: the monotonicity guarantee that the test suite checks
 * holds for any maximisation that does not *decrease* the expected complete-data
 * log-likelihood, and reaching the maximum is the cleanest way to be sure.
 */
export function maximiseItem(
  start: ItemPoint,
  rule: QuadratureRule,
  answered: Float64Array,
  correct: Float64Array,
  options: MaximiseOptions,
): MaximiseResult {
  const [minA, maxA] = options.discriminationBounds;
  const [minB, maxB] = options.difficultyBounds;
  const maxStep = options.maxStep ?? 0.5;
  const maxIterations = options.maxIterations ?? 40;
  const tolerance = options.tolerance ?? 1e-10;
  const rasch = options.model === 'rasch';

  let a = rasch ? 1 : clamp(start.discrimination, minA, maxA);
  let b = clamp(start.difficulty, minB, maxB);
  let stationary = false;
  let iterations = 0;

  for (; iterations < maxIterations; iterations += 1) {
    const item: ItemPoint = { discrimination: a, difficulty: b };
    const d = countDerivatives(item, rule, answered, correct);

    if (rasch) {
      if (Math.abs(d.gradientB) < tolerance) {
        stationary = true;
        break;
      }
      // Strictly concave in b, so the second derivative is negative wherever any
      // count is informative and the ratio is a descent direction on -l.
      if (!(d.hessianBB < -1e-14)) break;
      const step = clamp(-d.gradientB / d.hessianBB, -maxStep, maxStep);
      const next = clamp(b + step, minB, maxB);
      if (next === b) {
        stationary = Math.abs(step) < 1e-12;
        break;
      }
      b = next;
      continue;
    }

    const norm = Math.hypot(d.gradientA, d.gradientB);
    if (norm < tolerance) {
      stationary = true;
      break;
    }

    const determinant = d.hessianAA * d.hessianBB - d.hessianAB * d.hessianAB;
    let stepA: number;
    let stepB: number;
    if (determinant > 1e-14 && d.hessianAA < 0) {
      stepA = -(d.hessianBB * d.gradientA - d.hessianAB * d.gradientB) / determinant;
      stepB = -(d.hessianAA * d.gradientB - d.hessianAB * d.gradientA) / determinant;
    } else {
      stepA = 0.1 * d.gradientA;
      stepB = 0.1 * d.gradientB;
    }

    const nextA = clamp(a + clamp(stepA, -maxStep, maxStep), minA, maxA);
    const nextB = clamp(b + clamp(stepB, -maxStep, maxStep), minB, maxB);
    if (nextA === a && nextB === b) break;
    a = nextA;
    b = nextB;
  }

  const atBound = a === minA || a === maxA || b === minB || b === maxB;
  return { item: { discrimination: a, difficulty: b }, stationary, atBound, iterations };
}

/**
 * Standard errors from the curvature of the expected-count log-likelihood.
 *
 * The quantity most software reports, and the one to be careful with. It treats
 * the expected counts as if they had been observed, which they were not: they
 * were reconstructed from responses whose abilities are unknown. Ignoring that
 * reconstruction leaves out the *missing* information, and the result is an
 * interval narrower than the truth — optimistic in the direction that matters.
 *
 * It is still worth computing. It is cheap, it is the number a reference
 * implementation will print, and the ordering it induces over items is right
 * even where the level is not. The cross-product estimator in `mml.ts` is the
 * one to quote.
 */
export function countStandardErrors(
  item: ItemPoint,
  rule: QuadratureRule,
  answered: Float64Array,
  correct: Float64Array,
  model: CalibrationModel,
): { difficulty: number; discrimination: number } {
  const d = countDerivatives(item, rule, answered, correct);
  const infoAA = -d.hessianAA;
  const infoBB = -d.hessianBB;
  const infoAB = -d.hessianAB;

  if (model === 'rasch') {
    return {
      difficulty: infoBB > 0 ? 1 / Math.sqrt(infoBB) : Number.POSITIVE_INFINITY,
      discrimination: 0,
    };
  }

  const determinant = infoAA * infoBB - infoAB * infoAB;
  if (!(determinant > 1e-14) || infoAA <= 0 || infoBB <= 0) {
    return { difficulty: Number.POSITIVE_INFINITY, discrimination: Number.POSITIVE_INFINITY };
  }
  return {
    difficulty: Math.sqrt(infoAA / determinant),
    discrimination: Math.sqrt(infoBB / determinant),
  };
}
