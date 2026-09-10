import { mean, variance } from '../core/numeric.js';
import { normalGridRule, type QuadratureRule } from '../core/quadrature.js';
import { minimiseSimplex } from '../core/simplex.js';
import { categoryProbabilitiesOf, expectedScoreOf, isPolytomous } from '../models/mixed.js';
import {
  discriminationParameters,
  locationParameters,
  type CommonItemPair,
} from './common.js';
import { scaleTransform, transformItem, type ScaleTransform } from './scale.js';

/** Which method produced a set of linking coefficients. */
export type LinkingMethod = 'mean-mean' | 'mean-sigma' | 'haebara' | 'stocking-lord';

export interface LinkingResult {
  readonly transform: ScaleTransform;
  readonly method: LinkingMethod;
  /** Common items the coefficients were estimated from. */
  readonly commonItems: number;
  /** Objective at the solution, for the criterion methods. Zero for the moment methods. */
  readonly criterion: number;
  /** True unless an iterative method exhausted its budget. Always true for moment methods. */
  readonly converged: boolean;
}

/**
 * Mean/mean linking.
 *
 * `A` is the ratio of mean discriminations and `B` places the mean location.
 * Since a target discrimination is `a / A`, averaging both sides gives
 * `A = mean(a_source) / mean(a_target)` directly, and `B` then follows from
 * `mean(b_target) = A mean(b_source) + B`.
 *
 * Cheap and unbiased in the noiseless case, but the least stable of the four in
 * practice: discrimination estimates are the noisiest thing a calibration
 * produces, and this method puts all of `A` on their ratio.
 */
export function meanMean(pairs: readonly CommonItemPair[]): LinkingResult {
  requirePairs(pairs, 'meanMean');
  const sourceItems = pairs.map((pair) => pair.source);
  const targetItems = pairs.map((pair) => pair.target);

  const slope =
    mean(discriminationParameters(sourceItems)) / mean(discriminationParameters(targetItems));
  const intercept =
    mean(locationParameters(targetItems)) - slope * mean(locationParameters(sourceItems));

  return {
    transform: scaleTransform(slope, intercept),
    method: 'mean-mean',
    commonItems: pairs.length,
    criterion: 0,
    converged: true,
  };
}

/**
 * Mean/sigma linking.
 *
 * `A` is the ratio of location standard deviations and `B` places the mean,
 * which follows from requiring the transformed source locations to have the
 * same first two moments as the target locations.
 *
 * More stable than mean/mean because difficulty estimates are better determined
 * than discrimination estimates, but it uses only the locations and ignores the
 * discriminations entirely — so two banks whose items sit in the same places but
 * discriminate quite differently link identically under it.
 */
export function meanSigma(pairs: readonly CommonItemPair[]): LinkingResult {
  requirePairs(pairs, 'meanSigma');
  const sourceLocations = locationParameters(pairs.map((pair) => pair.source));
  const targetLocations = locationParameters(pairs.map((pair) => pair.target));

  const sourceSpread = Math.sqrt(variance(sourceLocations));
  if (sourceSpread === 0) {
    throw new RangeError(
      'meanSigma: the common items have no spread of locations on the source scale, ' +
        'so the slope is not identified; use meanMean or a criterion method',
    );
  }
  const slope = Math.sqrt(variance(targetLocations)) / sourceSpread;
  if (slope === 0) {
    throw new RangeError(
      'meanSigma: the common items have no spread of locations on the target scale, ' +
        'so the slope is not identified',
    );
  }
  const intercept = mean(targetLocations) - slope * mean(sourceLocations);

  return {
    transform: scaleTransform(slope, intercept),
    method: 'mean-sigma',
    commonItems: pairs.length,
    criterion: 0,
    converged: true,
  };
}

export interface CriterionOptions {
  /** Quadrature the criterion is summed over. Defaults to a 41-point standard normal grid. */
  readonly quadrature?: QuadratureRule;
  /** Starting coefficients for the search. Defaults to the mean/sigma solution. */
  readonly start?: ScaleTransform;
  /** Evaluation budget for the optimiser. Default 4000. */
  readonly maxEvaluations?: number;
}

function defaultRule(options: CriterionOptions): QuadratureRule {
  return options.quadrature ?? normalGridRule(0, 1, 41);
}

/**
 * Starting points for a criterion search.
 *
 * Both criteria have a spurious basin at large slopes, and it is not a
 * numerical artefact but a property of the objective. Dividing every
 * discrimination by a large `A` flattens every transformed response function
 * towards its own midpoint, and a flat curve near the middle of the probability
 * range scores better against a set of target curves than a steep curve in the
 * wrong place does. So from a badly wrong starting point the criterion falls
 * monotonically as `A` grows — measured on one 30-item anchor, the Haebara
 * criterion drops from 12.09 at the true slope to 1.60 at six times it — until
 * the transformed difficulties leave the range the item constructors accept and
 * the objective becomes infinite. A search started out there slides down that
 * slope and stops against the wall.
 *
 * The defence is to start from somewhere sensible and to try more than one
 * place. Both moment methods are closed-form and effectively free, so the search
 * runs from each of them, from the identity, and from any starting point the
 * caller supplied, keeping whichever run ends at the lowest criterion. Four
 * two-parameter searches cost almost nothing next to being confidently wrong.
 */
function startingPoints(
  pairs: readonly CommonItemPair[],
  options: CriterionOptions,
): ScaleTransform[] {
  const candidates: ScaleTransform[] = [];
  if (options.start !== undefined) candidates.push(options.start);
  for (const method of [meanSigma, meanMean]) {
    try {
      candidates.push(method(pairs).transform);
    } catch {
      // A degenerate anchor defeats that moment method; the others still stand.
    }
  }
  candidates.push(scaleTransform(1, 0));

  // Drop duplicates, which are common: on a well-behaved anchor the two moment
  // methods can agree to the last bit.
  const unique: ScaleTransform[] = [];
  for (const candidate of candidates) {
    const already = unique.some(
      (seen) =>
        Math.abs(seen.slope - candidate.slope) < 1e-12 &&
        Math.abs(seen.intercept - candidate.intercept) < 1e-12,
    );
    if (!already) unique.push(candidate);
  }
  return unique;
}

/**
 * The Haebara criterion at a candidate transformation.
 *
 * ```
 * H(A, B) = sum_theta W(theta) sum_j sum_k [P_jk(theta) - P*_jk(theta)]^2
 * ```
 *
 * where `P_jk` is the target calibration's probability of category `k` on
 * common item `j`, and `P*_jk` is the same for the source item transformed by
 * `(A, B)`. Both are evaluated at the same ability on the target scale, which
 * is what makes the difference meaningful.
 *
 * Haebara asks the two calibrations to agree *item by item*. That is a stricter
 * requirement than Stocking-Lord's, and the difference shows when the anchor is
 * misfitting: an item whose two calibrations disagree contributes to the Haebara
 * criterion no matter what the rest of the anchor does, whereas under
 * Stocking-Lord it can be cancelled by an item that disagrees the other way.
 */
export function haebaraCriterion(
  pairs: readonly CommonItemPair[],
  transform: ScaleTransform,
  rule: QuadratureRule,
): number {
  let total = 0;
  for (let node = 0; node < rule.nodes.length; node += 1) {
    const theta = rule.nodes[node] as number;
    const weight = rule.weights[node] as number;
    let atNode = 0;
    for (const pair of pairs) {
      const moved = transformItem(transform, pair.source);
      const targetProbabilities = categoryProbabilitiesOf(pair.target, theta);
      const movedProbabilities = categoryProbabilitiesOf(moved, theta);
      for (let k = 0; k < targetProbabilities.length; k += 1) {
        const gap = (targetProbabilities[k] as number) - (movedProbabilities[k] as number);
        atNode += gap * gap;
      }
    }
    total += weight * atNode;
  }
  return total;
}

/**
 * The Stocking-Lord criterion at a candidate transformation.
 *
 * ```
 * SL(A, B) = sum_theta W(theta) [sum_j E_j(theta) - sum_j E*_j(theta)]^2
 * ```
 *
 * The difference is taken between the two *test* characteristic curves — the
 * summed expected scores — rather than item by item. That is the quantity a
 * reported number-correct score actually depends on, which is the argument for
 * preferring it: it optimises the thing the test is used for, and tolerates
 * item-level disagreement that cancels in the total.
 */
export function stockingLordCriterion(
  pairs: readonly CommonItemPair[],
  transform: ScaleTransform,
  rule: QuadratureRule,
): number {
  let total = 0;
  for (let node = 0; node < rule.nodes.length; node += 1) {
    const theta = rule.nodes[node] as number;
    const weight = rule.weights[node] as number;
    let targetScore = 0;
    let movedScore = 0;
    for (const pair of pairs) {
      targetScore += expectedScoreOf(pair.target, theta);
      movedScore += expectedScoreOf(transformItem(transform, pair.source), theta);
    }
    const gap = targetScore - movedScore;
    total += weight * gap * gap;
  }
  return total;
}

/** Minimise a criterion over `(A, B)`, with `A` kept positive by construction. */
function minimiseCriterion(
  pairs: readonly CommonItemPair[],
  options: CriterionOptions,
  method: LinkingMethod,
  criterion: (transform: ScaleTransform) => number,
): LinkingResult {
  // The search runs on log(A) rather than A. The slope must stay positive — a
  // negative one reverses the ability order and the item constructors reject it
  // — and a simplex given a raw A will happily reflect across zero and spend
  // evaluations in a region where every candidate throws. Optimising the
  // logarithm makes positivity a property of the parameterisation instead of a
  // constraint the search has to be told about.
  const objective = (point: readonly number[]): number => {
    const slope = Math.exp(point[0] as number);
    const intercept = point[1] as number;
    if (!Number.isFinite(slope) || slope <= 0) return Number.POSITIVE_INFINITY;
    try {
      return criterion(scaleTransform(slope, intercept));
    } catch {
      return Number.POSITIVE_INFINITY;
    }
  };

  const budget = options.maxEvaluations ?? 4000;
  let best: { transform: ScaleTransform; value: number; converged: boolean } | null = null;
  for (const start of startingPoints(pairs, options)) {
    const result = minimiseSimplex(objective, [Math.log(start.slope), start.intercept], {
      maxEvaluations: budget,
      step: 0.05,
    });
    if (best === null || result.value < best.value) {
      best = {
        transform: scaleTransform(
          Math.exp(result.point[0] as number),
          result.point[1] as number,
        ),
        value: result.value,
        converged: result.converged,
      };
    }
  }
  if (best === null) {
    throw new Error(`${method}: no usable starting point for the criterion search`);
  }

  return {
    transform: best.transform,
    method,
    commonItems: pairs.length,
    criterion: best.value,
    converged: best.converged,
  };
}

/** Haebara linking: minimise the item-by-item response function difference. */
export function haebara(
  pairs: readonly CommonItemPair[],
  options: CriterionOptions = {},
): LinkingResult {
  requirePairs(pairs, 'haebara');
  const rule = defaultRule(options);
  return minimiseCriterion(pairs, options, 'haebara', (transform) =>
    haebaraCriterion(pairs, transform, rule),
  );
}

/** Stocking-Lord linking: minimise the test characteristic curve difference. */
export function stockingLord(
  pairs: readonly CommonItemPair[],
  options: CriterionOptions = {},
): LinkingResult {
  requirePairs(pairs, 'stockingLord');
  const rule = defaultRule(options);
  return minimiseCriterion(pairs, options, 'stocking-lord', (transform) =>
    stockingLordCriterion(pairs, transform, rule),
  );
}

/** Run one of the four methods by name. */
export function link(
  pairs: readonly CommonItemPair[],
  method: LinkingMethod,
  options: CriterionOptions = {},
): LinkingResult {
  switch (method) {
    case 'mean-mean':
      return meanMean(pairs);
    case 'mean-sigma':
      return meanSigma(pairs);
    case 'haebara':
      return haebara(pairs, options);
    case 'stocking-lord':
      return stockingLord(pairs, options);
    default: {
      const exhaustive: never = method;
      throw new RangeError(`unknown linking method ${String(exhaustive)}`);
    }
  }
}

/** Run all four methods, for comparison. */
export function linkAll(
  pairs: readonly CommonItemPair[],
  options: CriterionOptions = {},
): LinkingResult[] {
  return (['mean-mean', 'mean-sigma', 'haebara', 'stocking-lord'] as const).map((method) =>
    link(pairs, method, options),
  );
}

function requirePairs(pairs: readonly CommonItemPair[], caller: string): void {
  if (pairs.length === 0) {
    throw new RangeError(`${caller}: at least one common item is required`);
  }
  // Partial credit items cannot be rescaled without leaving their family, which
  // makes the criterion methods undefined for them; say so here rather than
  // failing partway through an optimisation.
  for (const pair of pairs) {
    if (isPolytomous(pair.source) && pair.source.parameters.model === 'partial-credit') {
      throw new RangeError(
        `${caller}: item "${pair.id}" is a partial credit item, whose fixed unit ` +
          'discrimination cannot be rescaled; calibrate the anchor with the ' +
          'generalized partial credit model to link it',
      );
    }
  }
}
