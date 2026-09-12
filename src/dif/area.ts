/**
 * Raju's area measures between two calibrations of the same item.
 *
 * Where the matched-group and regression methods work from responses, this
 * works from parameters: calibrate the item separately in each group, put the
 * two calibrations on one metric, and measure the area between the two response
 * functions. No matching, no stratification, and no criterion to contaminate.
 *
 * Two areas, and the pair is the point. The **signed** area cancels wherever the
 * curves cross, so it measures the uniform component alone. The **unsigned**
 * area cannot cancel, so it measures the total departure. An item with a plain
 * difficulty shift has the two equal in magnitude; an item whose curves cross at
 * the middle of the range has a signed area of zero and a large unsigned one.
 * Neither number alone distinguishes those cases and together they do.
 *
 * Both have exact closed forms for the models without a lower asymptote, which
 * is what makes them checkable: the implementation is validated against
 * numerical integration of the curves it claims to describe, not against itself.
 */

import { requireFinite } from '../core/numeric.js';
import { metricScale, type Item } from '../models/item.js';
import type { Favours } from './mantel.js';

export interface AreaResult {
  /**
   * The integral of the focal curve minus the reference curve.
   *
   * Equal to the difference in difficulty, scaled by the span between the
   * asymptotes, and entirely independent of the discriminations. Negative means
   * the item is harder for the focal group.
   */
  readonly signed: number;
  /** The integral of the absolute difference. Never less than `|signed|`. */
  readonly unsigned: number;
  /**
   * The ability at which the two curves cross, or null if they never do.
   *
   * Only finite when the discriminations differ. Worth reporting because it
   * says *where* an item is unfair, which is a different question from how much
   * — a crossing far outside the range candidates occupy is a smaller practical
   * problem than one in the middle of it.
   */
  readonly crossing: number | null;
  /**
   * How much of the departure the signed area misses, from 0 to 1.
   *
   * Zero when the curves are parallel and the whole effect is uniform; one when
   * they cross at a point that splits the effect evenly and the signed area
   * cancels to nothing. This is the number that says which method to trust:
   * near one, a pooled odds ratio is measuring almost nothing that is there.
   */
  readonly nonUniformShare: number;
  readonly favours: Favours;
}

/** `log(1 + exp(x))`, without overflowing for a large positive `x`. */
function softplus(x: number): number {
  return x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x));
}

/**
 * The signed and unsigned areas between two calibrations of one item.
 *
 * The two items must already be on a common metric — that is what the linking
 * methods are for — and must share their asymptotes. Asymptotes that differ are
 * refused rather than approximated: if the two lower asymptotes are unequal the
 * curves stay a constant distance apart forever below the range, so the
 * unsigned area is infinite and no finite number describes it.
 */
export function rajuArea(reference: Item, focal: Item): AreaResult {
  const first = reference.parameters;
  const second = focal.parameters;
  if (first.metric !== second.metric) {
    throw new RangeError(
      `rajuArea: the calibrations are on different metrics (${first.metric} and ${second.metric})`,
    );
  }
  if (first.c !== second.c || first.d !== second.d) {
    throw new RangeError(
      'rajuArea: the two calibrations have different asymptotes, so the curves do not ' +
        'converge at the tails and the area between them is unbounded',
    );
  }

  const scale = metricScale(first.metric);
  const span = first.d - first.c;
  const { a: aReference, b: bReference } = first;
  const { a: aFocal, b: bFocal } = second;
  const gap = bFocal - bReference;

  // Normalised so an item calibrated identically twice reports +0 rather than
  // the -0 that `span * -0` produces, which compares unequal to 0 under Object.is.
  const rawSigned = span * -gap;
  const signed = rawSigned === 0 ? 0 : rawSigned;

  // Equal discriminations: the curves are translates of each other, never
  // cross, and the unsigned area is the shift itself.
  let unsigned: number;
  let crossing: number | null;
  if (aReference === aFocal) {
    unsigned = span * Math.abs(gap);
    crossing = null;
  } else {
    const difference = aFocal - aReference;
    // The exponent runs to infinity as the discriminations converge; softplus
    // is what keeps the closed form usable in that limit, where it correctly
    // tends to the equal-discrimination answer rather than to infinity.
    const exponent = (scale * aReference * aFocal * gap) / difference;
    const term = ((2 * difference) / (scale * aReference * aFocal)) * softplus(exponent);
    unsigned = span * Math.abs(term - gap);
    crossing = (aReference * bReference - aFocal * bFocal) / (aReference - aFocal);
  }

  // The signed area cannot exceed the unsigned one; rounding on nearly equal
  // discriminations is the only way the comparison ever comes out the other way.
  unsigned = Math.max(unsigned, Math.abs(signed));

  return {
    signed,
    unsigned,
    crossing,
    nonUniformShare: unsigned === 0 ? 0 : 1 - Math.abs(signed) / unsigned,
    favours: signed === 0 ? 'neither' : signed > 0 ? 'focal' : 'reference',
  };
}

/**
 * The area between two curves by direct integration, as a check on the
 * closed form.
 *
 * Simpson's rule over a wide finite window. Not used by `rajuArea` — it exists
 * so that the closed form can be verified against the thing it is a closed form
 * *of*, on parameters the caller chooses, rather than only on the cases someone
 * thought to write a test for.
 */
export function integrateArea(
  reference: Item,
  focal: Item,
  options: { readonly limit?: number; readonly intervals?: number } = {},
): { signed: number; unsigned: number } {
  // The window has to be wide enough that the untruncated tails are negligible
  // for the *flattest* curve in play. A discrimination of 0.35 still has a
  // probability near 1e-6 forty logits out, and the tail of an absolute
  // difference does not cancel — truncating at forty costs five significant
  // digits on exactly the crossing items this exists to check.
  const limit = options.limit ?? 80;
  const intervals = options.intervals ?? 40000;
  requireFinite(limit, 'integrateArea: limit');
  if (!(limit > 0)) throw new RangeError(`integrateArea: limit must be positive, received ${limit}`);
  if (!Number.isInteger(intervals) || intervals < 2 || intervals % 2 !== 0) {
    throw new RangeError(
      `integrateArea: intervals must be an even integer of at least 2, received ${intervals}`,
    );
  }

  const difference = (theta: number): number =>
    probability(focal, theta) - probability(reference, theta);

  // Integrating |difference| directly would put a kink under the quadrature at
  // every crossing, and Simpson's rule loses two orders of convergence across
  // one. Splitting the window at the crossings instead leaves a smooth
  // integrand on each piece — and makes the unsigned area the sum of the
  // absolute *segment* integrals, since the sign cannot change inside a
  // segment. That identity is exact, not an approximation of one.
  const breakpoints = [-limit, ...crossings(difference, limit), limit];

  let signed = 0;
  let unsigned = 0;
  for (let index = 0; index + 1 < breakpoints.length; index += 1) {
    const from = breakpoints[index] as number;
    const to = breakpoints[index + 1] as number;
    const share = Math.max(2, 2 * Math.ceil((intervals * (to - from)) / (4 * limit)));
    const piece = simpson(difference, from, to, share);
    signed += piece;
    unsigned += Math.abs(piece);
  }
  return { signed, unsigned };
}

/** Sign changes of `f` inside the window, refined by bisection. */
function crossings(f: (x: number) => number, limit: number): number[] {
  const SCAN = 2000;
  const step = (2 * limit) / SCAN;
  const found: number[] = [];
  let previous = f(-limit);

  for (let index = 1; index <= SCAN; index += 1) {
    const right = -limit + index * step;
    const current = f(right);
    if (previous === 0) found.push(right - step);
    else if (previous * current < 0) {
      let low = right - step;
      let high = right;
      let atLow = previous;
      for (let step2 = 0; step2 < 80; step2 += 1) {
        const middle = (low + high) / 2;
        const atMiddle = f(middle);
        if (atMiddle === 0) {
          low = middle;
          high = middle;
          break;
        }
        if (atLow * atMiddle < 0) high = middle;
        else {
          low = middle;
          atLow = atMiddle;
        }
      }
      found.push((low + high) / 2);
    }
    previous = current;
  }
  return found;
}

/** Composite Simpson's rule over `intervals` panels, which must be even. */
function simpson(f: (x: number) => number, from: number, to: number, intervals: number): number {
  const step = (to - from) / intervals;
  let total = f(from) + f(to);
  for (let index = 1; index < intervals; index += 1) {
    total += (index % 2 === 1 ? 4 : 2) * f(from + index * step);
  }
  return (total * step) / 3;
}

function probability(item: Item, theta: number): number {
  const { a, b, c, d, metric } = item.parameters;
  return c + (d - c) / (1 + Math.exp(-metricScale(metric) * a * (theta - b)));
}

export interface AreaRow {
  readonly itemId: string;
  readonly area: AreaResult;
}

/**
 * Area measures for every item calibrated in both groups.
 *
 * Matched by identifier rather than by position, because two calibrations of
 * the same bank routinely drop different items — one group may not have reached
 * an item at all — and a positional match would silently compare the wrong
 * pairs rather than reporting a shorter list.
 */
export function areaScan(
  referenceBank: readonly Item[],
  focalBank: readonly Item[],
): AreaRow[] {
  const byId = new Map(focalBank.map((item) => [item.id, item]));
  const rows: AreaRow[] = [];
  for (const item of referenceBank) {
    const counterpart = byId.get(item.id);
    if (counterpart === undefined) continue;
    rows.push({ itemId: item.id, area: rajuArea(item, counterpart) });
  }
  if (rows.length === 0) {
    throw new RangeError('areaScan: the two calibrations share no item identifiers');
  }
  return rows;
}

/** Rows ordered by total departure, largest first. */
export function rankByArea(rows: readonly AreaRow[]): AreaRow[] {
  return [...rows].sort((a, b) => b.area.unsigned - a.area.unsigned);
}
