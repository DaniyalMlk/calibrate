import {
  itemInformation,
  probabilityCorrect,
  responseDerivative,
  responseSecondDerivative,
  type ScoredResponse,
} from '../models/response.js';

/**
 * The smallest probability the log-likelihood will take a logarithm of.
 *
 * A response pattern can easily contain an answer whose probability underflows
 * to exactly zero at an extreme trial ability — a correct answer to a very hard
 * item evaluated at theta = -10, say. `Math.log(0)` is `-Infinity`, which turns
 * the whole log-likelihood into `-Infinity` and destroys the information the
 * other items carried. Flooring the probability keeps the surface finite and
 * still steeply penalises the impossible region.
 */
const PROBABILITY_FLOOR = 1e-300;

/**
 * Log-likelihood of a response pattern at a trial ability.
 *
 * `L(theta) = sum_i [ u_i log P_i(theta) + (1 - u_i) log (1 - P_i(theta)) ]`
 */
export function logLikelihood(responses: readonly ScoredResponse[], theta: number): number {
  let total = 0;
  for (const { item, response } of responses) {
    const p = probabilityCorrect(item.parameters, theta);
    const observed = response === 1 ? p : 1 - p;
    total += Math.log(Math.max(observed, PROBABILITY_FLOOR));
  }
  return total;
}

/**
 * Score function: the first derivative of the log-likelihood with respect to
 * ability.
 *
 * `S(theta) = sum_i (u_i - P_i) * P'_i / (P_i * Q_i)`
 *
 * The residual `u_i - P_i` is what makes this intuitive: each item pushes the
 * estimate in the direction of its surprise, weighted by how sharply that item
 * discriminates at the current ability.
 */
export function scoreFunction(responses: readonly ScoredResponse[], theta: number): number {
  let total = 0;
  for (const { item, response } of responses) {
    const p = probabilityCorrect(item.parameters, theta);
    const q = 1 - p;
    const denominator = p * q;
    if (denominator <= Number.MIN_VALUE) continue;
    total += ((response - p) * responseDerivative(item.parameters, theta)) / denominator;
  }
  return total;
}

/**
 * Observed information: the negative second derivative of the log-likelihood.
 *
 * ```
 * -d2L/dtheta2 = sum_i { [u_i/P_i^2 + (1 - u_i)/Q_i^2] (P'_i)^2
 *                        - [u_i/P_i - (1 - u_i)/Q_i] P''_i }
 * ```
 *
 * Unlike Fisher information this depends on the answers actually given, so it
 * can be negative — which is exactly why a raw Newton step on the likelihood is
 * unsafe and the estimator brackets the root instead.
 */
export function observedInformation(responses: readonly ScoredResponse[], theta: number): number {
  let total = 0;
  for (const { item, response } of responses) {
    const p = probabilityCorrect(item.parameters, theta);
    const q = 1 - p;
    if (p <= Number.MIN_VALUE || q <= Number.MIN_VALUE) continue;
    const first = responseDerivative(item.parameters, theta);
    const second = responseSecondDerivative(item.parameters, theta);
    const curvature = response === 1 ? 1 / (p * p) : 1 / (q * q);
    const slope = response === 1 ? 1 / p : -1 / q;
    total += curvature * first * first - slope * second;
  }
  return total;
}

/**
 * Fisher (expected) information for a response pattern — the sum of item
 * informations over the items administered.
 *
 * This is the expectation of the observed information over responses, so the
 * two agree exactly for the 1PL and 2PL (which have a canonical link) and differ
 * for the 3PL and 4PL.
 */
export function expectedInformation(responses: readonly ScoredResponse[], theta: number): number {
  let total = 0;
  for (const { item } of responses) total += itemInformation(item.parameters, theta);
  return total;
}

/**
 * Classify a response pattern by whether its likelihood has a finite maximiser.
 *
 * An all-correct pattern has a log-likelihood that increases monotonically in
 * ability, so its maximum likelihood estimate is `+Infinity`; an all-incorrect
 * pattern is the mirror image. Reporting these as `unbounded` rather than
 * silently returning the edge of the search window is the difference between an
 * estimator that is honest and one that is merely convenient.
 *
 * Under the 3PL a pattern is only genuinely unbounded downwards if it is
 * all-incorrect; a correct answer to an item with a nonzero lower asymptote
 * still carries a finite amount of evidence, so no special case is needed.
 */
export function patternBoundedness(
  responses: readonly ScoredResponse[],
): 'bounded' | 'unbounded-high' | 'unbounded-low' | 'empty' {
  if (responses.length === 0) return 'empty';
  let correct = 0;
  for (const { response } of responses) correct += response;
  if (correct === responses.length) return 'unbounded-high';
  if (correct === 0) return 'unbounded-low';
  return 'bounded';
}
