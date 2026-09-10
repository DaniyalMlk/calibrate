import {
  categoryDerivativesOf,
  categoryProbabilitiesOf,
  categorySecondDerivativesOf,
  informationOf,
  maximumScoreOf,
  responseCategory,
  type AnyResponse,
} from '../models/mixed.js';

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
 * `L(theta) = sum_i log P_{i,k_i}(theta)`, where `k_i` is the category item `i`
 * was scored into.
 *
 * The dichotomous form `u log P + (1 - u) log (1 - P)` is this same sum: a
 * dichotomous item has categories `[1 - P, P]`, so selecting category `u` picks
 * out exactly the term that form keeps. Writing it once over categories means a
 * mixed form needs no branch, and the dichotomous results are unchanged to the
 * last bit.
 */
export function logLikelihood(responses: readonly AnyResponse[], theta: number): number {
  let total = 0;
  for (const response of responses) {
    const probabilities = categoryProbabilitiesOf(response.item, theta);
    const observed = probabilities[responseCategory(response)] as number;
    total += Math.log(Math.max(observed, PROBABILITY_FLOOR));
  }
  return total;
}

/**
 * Score function: the first derivative of the log-likelihood with respect to
 * ability.
 *
 * `S(theta) = sum_i P'_{i,k_i} / P_{i,k_i}`.
 *
 * The dichotomous residual form `(u - P) P' / (P Q)` falls out of this: for a
 * correct answer the ratio is `P' / P`, and for an incorrect one the category
 * probability is `1 - P` with derivative `-P'`, giving `-P' / Q`. Both are what
 * the residual form evaluates to, so the intuition survives — each item pushes
 * the estimate in the direction of its surprise, weighted by how sharply it
 * discriminates — and it now extends to an item answered halfway.
 */
export function scoreFunction(responses: readonly AnyResponse[], theta: number): number {
  let total = 0;
  for (const response of responses) {
    const category = responseCategory(response);
    const probability = categoryProbabilitiesOf(response.item, theta)[category] as number;
    if (probability <= Number.MIN_VALUE) continue;
    const derivative = categoryDerivativesOf(response.item, theta)[category] as number;
    total += derivative / probability;
  }
  return total;
}

/**
 * Observed information: the negative second derivative of the log-likelihood.
 *
 * `-d2L/dtheta2 = sum_i [ (P'_{i,k} / P_{i,k})^2 - P''_{i,k} / P_{i,k} ]`
 *
 * Unlike Fisher information this depends on the categories actually scored, so
 * it can be negative — which is exactly why a raw Newton step on the likelihood
 * is unsafe and the estimator brackets the root instead.
 */
export function observedInformation(responses: readonly AnyResponse[], theta: number): number {
  let total = 0;
  for (const response of responses) {
    const category = responseCategory(response);
    const probability = categoryProbabilitiesOf(response.item, theta)[category] as number;
    if (probability <= Number.MIN_VALUE) continue;
    const first = categoryDerivativesOf(response.item, theta)[category] as number;
    const second = categorySecondDerivativesOf(response.item, theta)[category] as number;
    const ratio = first / probability;
    total += ratio * ratio - second / probability;
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
export function expectedInformation(responses: readonly AnyResponse[], theta: number): number {
  let total = 0;
  for (const response of responses) total += informationOf(response.item, theta);
  return total;
}

/**
 * Classify a response pattern by whether its likelihood has a finite maximiser.
 *
 * A pattern in which every item scored its own maximum has a log-likelihood
 * that increases monotonically in ability, so its maximum likelihood estimate is
 * `+Infinity`; a pattern scoring zero everywhere is the mirror image. Reporting
 * these as `unbounded` rather than silently returning the edge of the search
 * window is the difference between an estimator that is honest and one that is
 * merely convenient.
 *
 * On a mixed form "every item at its maximum" is not the same as "all correct":
 * a candidate who answers every multiple-choice item correctly and scores 2 of 3
 * on the essay has a perfectly finite estimate, because the essay category
 * carries evidence about where on the scale they actually sit. Comparing each
 * response against its own item's maximum, rather than against 1, is what
 * distinguishes the two.
 *
 * Under the 3PL a pattern is only genuinely unbounded downwards if it scores
 * zero throughout; a correct answer to an item with a nonzero lower asymptote
 * still carries a finite amount of evidence, so no special case is needed.
 */
export function patternBoundedness(
  responses: readonly AnyResponse[],
): 'bounded' | 'unbounded-high' | 'unbounded-low' | 'empty' {
  if (responses.length === 0) return 'empty';
  let allAtMaximum = true;
  let allAtZero = true;
  for (const response of responses) {
    const category = responseCategory(response);
    if (category !== maximumScoreOf(response.item)) allAtMaximum = false;
    if (category !== 0) allAtZero = false;
  }
  if (allAtMaximum) return 'unbounded-high';
  if (allAtZero) return 'unbounded-low';
  return 'bounded';
}
