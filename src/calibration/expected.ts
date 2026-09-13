import { logistic } from '../core/numeric.js';
import type { QuadratureRule } from '../core/quadrature.js';
import { MISSING, type Cell, type ResponseMatrix } from './matrix.js';

/**
 * Item parameters in the form the expectation step works with.
 *
 * Deliberately not `Item`: the E-step runs once per person per node per item in
 * the inner loop of every EM iteration, and reading two numbers out of a plain
 * record is measurably cheaper than going through the model dispatch. The
 * conversion back to `Item` happens once, at the end.
 */
export interface ItemPoint {
  readonly discrimination: number;
  readonly difficulty: number;
}

/**
 * Expected counts from one expectation step.
 *
 * `answered[j][k]` is the expected number of respondents at ability node `k`
 * who sat item `j`, and `correct[j][k]` how many of them got it right. Both are
 * expectations under each respondent's own ability posterior, so they are real
 * numbers rather than integers, and both are indexed by item as well as node
 * because missing responses make the two differ: a respondent who skipped item
 * `j` contributes to no other item's absence but must contribute nothing to
 * `j`'s counts.
 */
export interface ExpectedCounts {
  readonly answered: readonly Float64Array[];
  readonly correct: readonly Float64Array[];
  /** Total posterior mass at each node, summed over respondents. */
  readonly density: Float64Array;
  /** The marginal log-likelihood of the whole matrix at these parameters. */
  readonly logLikelihood: number;
}

/**
 * The log of the conditional likelihood of one response pattern at one ability.
 *
 * Summed in logs rather than multiplied in probabilities. A forty-item pattern
 * whose responses are individually unremarkable still has a likelihood around
 * `1e-13`, and at the far nodes of the quadrature rule — where the pattern is
 * wildly improbable — the product underflows to exactly zero well before the
 * rule's weight makes it negligible. Zero times a weight is zero, the node
 * silently drops out of the posterior, and the estimate moves. In logs nothing
 * underflows until the exponent is taken, by which point the log-sum-exp below
 * has already subtracted the maximum.
 */
export function patternLogLikelihood(
  row: readonly Cell[],
  items: readonly ItemPoint[],
  theta: number,
): number {
  let total = 0;
  for (const [index, cell] of row.entries()) {
    if (cell === MISSING) continue;
    const item = items[index] as ItemPoint;
    const exponent = item.discrimination * (theta - item.difficulty);
    // log(sigma(x)) = -log(1 + exp(-x)) and log(1 - sigma(x)) = -log(1 + exp(x)),
    // both evaluated through log1p on a non-positive exponent so neither the
    // exponential nor the logarithm loses precision at the tails.
    total += cell === 1 ? logSigmoid(exponent) : logSigmoid(-exponent);
  }
  return total;
}

/** `log(1 / (1 + exp(-x)))`, accurate for either sign of `x`. */
function logSigmoid(x: number): number {
  return x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x));
}

/**
 * The posterior over ability nodes for one respondent, and the log of the
 * marginal likelihood of their pattern.
 *
 * Bayes on a grid: multiply the population weight at each node by the
 * conditional likelihood of the pattern there, and normalise. The normalising
 * constant is exactly the marginal likelihood of the pattern, so it comes out of
 * the same pass rather than needing a second one.
 *
 * The maximum log-likelihood is subtracted before exponentiating. Without it a
 * pattern of thirty confident responses has a log-likelihood around `-60` at its
 * own mode and `-400` at the far nodes; `exp(-400)` is zero in a double, and the
 * ratio that Bayes actually needs is perfectly representable.
 */
export function posteriorOverNodes(
  row: readonly Cell[],
  items: readonly ItemPoint[],
  rule: QuadratureRule,
  into: Float64Array,
): number {
  const count = rule.nodes.length;
  let peak = Number.NEGATIVE_INFINITY;
  for (let k = 0; k < count; k += 1) {
    const value = patternLogLikelihood(row, items, rule.nodes[k] as number);
    into[k] = value;
    if (value > peak) peak = value;
  }

  let total = 0;
  for (let k = 0; k < count; k += 1) {
    const weight = (rule.weights[k] as number) * Math.exp((into[k] as number) - peak);
    into[k] = weight;
    total += weight;
  }

  if (!(total > 0)) {
    // Every node carries zero posterior mass. Reachable only if the rule's own
    // weights are degenerate, since the likelihood ratios were rescaled above;
    // falling back to the population weights keeps the counts finite and lets
    // the caller's convergence test see a flat likelihood rather than a NaN.
    for (let k = 0; k < count; k += 1) into[k] = rule.weights[k] as number;
    return Math.log(sumOf(rule.weights)) + peak;
  }

  for (let k = 0; k < count; k += 1) into[k] = (into[k] as number) / total;
  return Math.log(total) + peak;
}

function sumOf(values: readonly number[]): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

/**
 * The expectation step: posterior-weighted response counts at every ability node.
 *
 * This is the step that removes the person parameters. Rather than committing
 * each respondent to a point estimate of their ability and counting them there,
 * every respondent is spread across the whole ability range in proportion to how
 * plausible each ability is given their answers. A respondent who answered
 * fifteen of thirty items correctly contributes a broad smear of mass near the
 * middle; one who answered all thirty contributes mass piled up at the top but
 * still finite — which is why marginal estimation, unlike joint estimation, does
 * not need perfect scorers removed before it can run.
 *
 * What comes out is a synthetic dataset: at each ability node, how many people
 * of that ability sat each item and how many passed it. The maximisation step
 * then fits each item to its own column of that table, one small weighted
 * logistic regression per item, with no person parameters anywhere in it.
 */
export function expectationStep(
  matrix: ResponseMatrix,
  items: readonly ItemPoint[],
  rule: QuadratureRule,
): ExpectedCounts {
  const nodeCount = rule.nodes.length;
  const itemCount = matrix.itemCount;
  if (items.length !== itemCount) {
    throw new RangeError(
      `expectationStep: ${items.length} item parameters for ${itemCount} items`,
    );
  }

  const answered = Array.from({ length: itemCount }, () => new Float64Array(nodeCount));
  const correct = Array.from({ length: itemCount }, () => new Float64Array(nodeCount));
  const density = new Float64Array(nodeCount);
  const posterior = new Float64Array(nodeCount);
  let logLikelihood = 0;

  for (let person = 0; person < matrix.personCount; person += 1) {
    const row = matrix.row(person);
    logLikelihood += posteriorOverNodes(row, items, rule, posterior);

    for (let k = 0; k < nodeCount; k += 1) {
      density[k] = (density[k] as number) + (posterior[k] as number);
    }
    for (const [item, cell] of row.entries()) {
      if (cell === MISSING) continue;
      const answeredRow = answered[item] as Float64Array;
      const correctRow = correct[item] as Float64Array;
      for (let k = 0; k < nodeCount; k += 1) {
        const mass = posterior[k] as number;
        answeredRow[k] = (answeredRow[k] as number) + mass;
        if (cell === 1) correctRow[k] = (correctRow[k] as number) + mass;
      }
    }
  }

  return { answered, correct, density, logLikelihood };
}

/**
 * The marginal log-likelihood of the matrix on its own, without the counts.
 *
 * The expectation step already returns this, so the only reason to compute it
 * separately is to check a set of parameters that did not come from an EM
 * iteration — comparing a joint calibration against a marginal one on the same
 * objective, for instance.
 */
export function marginalLogLikelihood(
  matrix: ResponseMatrix,
  items: readonly ItemPoint[],
  rule: QuadratureRule,
): number {
  const posterior = new Float64Array(rule.nodes.length);
  let total = 0;
  for (let person = 0; person < matrix.personCount; person += 1) {
    total += posteriorOverNodes(matrix.row(person), items, rule, posterior);
  }
  return total;
}

/** The response probability of an item at a node, as the E and M steps see it. */
export function pointProbability(item: ItemPoint, theta: number): number {
  return logistic(item.discrimination * (theta - item.difficulty));
}
