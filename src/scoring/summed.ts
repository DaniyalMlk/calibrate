import { requireFinite } from '../core/numeric.js';
import type { QuadratureRule } from '../core/quadrature.js';
import {
  categoryProbabilitiesOf,
  maximumScoreOf,
  maximumTestScore,
  type AnyItem,
} from '../models/mixed.js';

/**
 * The distribution of total scores on a form, at one ability.
 *
 * Returned as `P[s]` for every attainable total `s`, from zero to the sum of the
 * item maxima, so the array has `maximumTestScore(items) + 1` entries and sums
 * to one.
 *
 * The obvious way to compute this is to enumerate response patterns and add up
 * the ones that total `s`. On a form of `L` binary items that is `2^L` patterns,
 * which is a few thousand at ten items and a hundred million at twenty-seven —
 * and rubric items make it worse, since a four-level item multiplies the count
 * by four rather than two.
 *
 * The recursion avoids the enumeration entirely. Add one item at a time: if the
 * distribution over totals on the first `j` items is known, then after
 * administering item `j + 1` a total of `s` can only have arisen from a total of
 * `s - c` before it and a score of `c` on the new item. Summing over the item's
 * categories convolves its distribution into the running one, and the whole
 * form costs one pass per item over an array no longer than the maximum score.
 *
 * This works because responses are independent *given ability*, which is the
 * local independence assumption the whole model rests on. It is not an
 * approximation: the result is exactly the enumeration, and the test suite
 * checks that on forms short enough to enumerate.
 */
export function scoreDistribution(items: readonly AnyItem[], theta: number): number[] {
  requireFinite(theta, 'ability (theta)');
  if (items.length === 0) throw new RangeError('scoreDistribution: at least one item is required');

  // Before any item has been administered the total is zero with certainty.
  let distribution = [1];
  for (const item of items) {
    const probabilities = categoryProbabilitiesOf(item, theta);
    const maximum = maximumScoreOf(item);
    const next = new Array<number>(distribution.length + maximum).fill(0);
    for (const [before, mass] of distribution.entries()) {
      if (mass === 0) continue;
      for (let category = 0; category <= maximum; category += 1) {
        next[before + category] =
          (next[before + category] as number) + mass * (probabilities[category] as number);
      }
    }
    distribution = next;
  }
  return distribution;
}

/**
 * The distribution of total scores a population will produce on a form.
 *
 * The conditional distribution above, averaged over the ability distribution.
 * This is what a form actually generates when a cohort sits it, and it is
 * usually the first thing anyone wants from a new form: whether the scores pile
 * up at one end, whether the middle of the range is populated enough to set a
 * cut score on, whether a ceiling is being hit.
 */
export function marginalScoreDistribution(
  items: readonly AnyItem[],
  rule: QuadratureRule,
): number[] {
  if (rule.nodes.length !== rule.weights.length) {
    throw new RangeError('marginalScoreDistribution: rule has mismatched nodes and weights');
  }
  const total = new Array<number>(maximumTestScore(items) + 1).fill(0);
  for (const [k, node] of rule.nodes.entries()) {
    const weight = rule.weights[k] as number;
    if (weight === 0) continue;
    const conditional = scoreDistribution(items, node);
    for (const [score, mass] of conditional.entries()) {
      total[score] = (total[score] as number) + weight * mass;
    }
  }

  // The rule's weights sum to one only to quadrature accuracy; renormalising
  // means callers can treat the result as a distribution without checking.
  let mass = 0;
  for (const value of total) mass += value;
  return mass > 0 ? total.map((value) => value / mass) : total;
}

/**
 * The likelihood of each total score across a grid of abilities.
 *
 * `likelihood[s][k]` is the probability of scoring `s` at node `k`. This is the
 * object summed-score scoring needs: for a candidate whose only reported datum
 * is their total, the likelihood of ability is the row of this table at their
 * score, and everything else follows by Bayes.
 *
 * Computed once for the whole form rather than once per candidate, which is the
 * reason a conversion table is cheap to publish: the expensive part does not
 * depend on who took the test.
 */
export function scoreLikelihoods(
  items: readonly AnyItem[],
  rule: QuadratureRule,
): Float64Array[] {
  const maximum = maximumTestScore(items);
  const table = Array.from({ length: maximum + 1 }, () => new Float64Array(rule.nodes.length));
  for (const [k, node] of rule.nodes.entries()) {
    const conditional = scoreDistribution(items, node);
    for (const [score, mass] of conditional.entries()) {
      (table[score] as Float64Array)[k] = mass;
    }
  }
  return table;
}
