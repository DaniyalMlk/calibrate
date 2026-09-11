import { linspace, mean } from '../core/numeric.js';
import { informationPeak, standardError } from '../models/response.js';
import type { ItemParameters } from '../models/item.js';
import {
  discriminationOf,
  formatCounts,
  isPolytomous,
  itemLocation,
  testInformationOf,
  type AnyItem,
} from '../models/mixed.js';
import { polytomousInformationPeak } from '../models/polytomous.js';

/**
 * The ability at which an item of either format is most informative.
 *
 * For a dichotomous item this is the closed-form peak of its own information
 * curve. For a rubric-scored item there is no closed form — the sum over
 * categories has no single maximiser to solve for — so it is found
 * numerically. Both answer the same question, which is the one `itemsNearby`
 * asks: where does this item actually do its work?
 *
 * Deliberately not `itemLocation`. A polytomous item's location is the mean of
 * its thresholds, and an item whose thresholds straddle the location informs
 * most somewhere near it but not at it. Reporting the location here would
 * quietly answer a different question than the dichotomous branch does.
 */
function informationPeakOf(item: AnyItem): number {
  return isPolytomous(item)
    ? polytomousInformationPeak(item.parameters)
    : informationPeak(item.parameters as ItemParameters);
}

/** What the bank can do at one point on the ability scale. */
export interface CoveragePoint {
  readonly theta: number;
  /** Test information from the whole bank at this ability. */
  readonly information: number;
  /** Standard error if every item in the bank were administered. */
  readonly standardError: number;
  /** Items whose own information peaks within half a logit of this point. */
  readonly itemsNearby: number;
  /** Whether the bank reaches the target precision here. */
  readonly meetsTarget: boolean;
}

/**
 * A contiguous stretch of the ability range the bank cannot measure to target.
 *
 * The bounds are the first and last evaluated points that miss, not the points
 * either side of them: a gap reported as `[1.75, 3.00]` is a range over which
 * the bank was actually measured and actually failed, which is what an
 * item-writing brief needs.
 */
export interface CoverageGap {
  readonly from: number;
  readonly to: number;
  /** Worst standard error inside the gap. */
  readonly worstStandardError: number;
}

export interface BankHealth {
  readonly items: number;
  readonly range: readonly [number, number];
  readonly target: number;
  readonly points: readonly CoveragePoint[];
  readonly gaps: readonly CoverageGap[];
  /** Fraction of the evaluated range where the bank reaches the target. */
  readonly covered: number;
  /**
   * Mean, minimum and maximum item location.
   *
   * A dichotomous item contributes its difficulty and a rubric-scored one the
   * mean of its thresholds, so on a dichotomous bank this is exactly the
   * difficulty summary it has always been.
   */
  readonly difficulty: { readonly mean: number; readonly min: number; readonly max: number };
  /** Mean item discrimination. */
  readonly meanDiscrimination: number;
  /** Ability at which the bank is most informative. */
  readonly peak: number;
  /**
   * How the bank splits between formats, and the highest total score it could
   * award if every item were administered.
   *
   * Reported because the mix changes what the coverage above means. Sixty
   * dichotomous items and twelve four-category rubrics is not the same bank as
   * seventy-two dichotomous items, even where the two reach identical standard
   * errors: the rubric items carry several times the score range, so a gap in
   * the region they cover costs more than the item count suggests.
   */
  readonly formats: {
    readonly dichotomous: number;
    readonly polytomous: number;
    readonly maximumScore: number;
  };
}

export interface BankHealthOptions {
  /** Ability range to evaluate. Default -3 to 3. */
  readonly range?: readonly [number, number];
  /** Points at which to evaluate it. Default 25. */
  readonly points?: number;
  /**
   * Standard error the bank should be able to reach. Default 0.3.
   *
   * Measured against the *whole bank*, which is deliberately optimistic: no
   * candidate sees every item. A region the full bank cannot measure to target
   * is one no adaptive test over that bank will ever measure to target, which
   * makes this the cheapest possible screen and a sufficient one for finding
   * holes.
   */
  readonly target?: number;
}

/**
 * Where a bank can measure, and where it cannot.
 *
 * The question a bank owner needs answered before writing more items is not how
 * many items they have but where the next ones should sit. A bank of three
 * hundred items clustered around the population mean measures the middle of the
 * scale beautifully and cannot tell a strong candidate from an exceptional one —
 * and the marginal statistics of a live testing programme will not reveal that,
 * because hardly any candidates are out there to be measured badly.
 *
 * Reporting the gaps as intervals rather than as a list of failing grid points
 * is the useful form: "nothing above 1.75" is an item-writing brief, where
 * twelve consecutive rows of a table are a puzzle.
 *
 * The bank may mix formats. Everything here is computed through the
 * format-agnostic accessors, so a rubric-scored item contributes its own
 * information to the coverage rather than being dropped from it — which
 * matters most exactly where such items tend to sit, since a bank that puts
 * its constructed-response items at the top of the scale is a bank whose
 * upper-range coverage is invisible without them.
 */
export function bankHealth(bank: readonly AnyItem[], options: BankHealthOptions = {}): BankHealth {
  if (bank.length === 0) throw new RangeError('bankHealth: the bank is empty');
  const [lower, upper] = options.range ?? [-3, 3];
  const count = options.points ?? 25;
  const target = options.target ?? 0.3;

  if (!(lower < upper)) {
    throw new RangeError(`bankHealth: range must satisfy lower < upper, received [${lower}, ${upper}]`);
  }
  if (!Number.isInteger(count) || count < 2) {
    throw new RangeError(`bankHealth: points must be an integer >= 2, received ${count}`);
  }
  if (!(target > 0)) {
    throw new RangeError(`bankHealth: target must be positive, received ${target}`);
  }

  const peaks = bank.map(informationPeakOf);
  const grid = linspace(lower, upper, count);

  const points: CoveragePoint[] = grid.map((theta) => {
    const information = testInformationOf(bank, theta);
    const error = standardError(information);
    return {
      theta,
      information,
      standardError: error,
      itemsNearby: peaks.filter((peak) => Math.abs(peak - theta) <= 0.5).length,
      meetsTarget: error <= target,
    };
  });

  const gaps: CoverageGap[] = [];
  let open: { from: number; to: number; worst: number } | null = null;
  for (const point of points) {
    if (!point.meetsTarget) {
      if (open === null) open = { from: point.theta, to: point.theta, worst: point.standardError };
      else {
        open.to = point.theta;
        open.worst = Math.max(open.worst, point.standardError);
      }
      continue;
    }
    if (open !== null) {
      gaps.push({ from: open.from, to: open.to, worstStandardError: open.worst });
      open = null;
    }
  }
  if (open !== null) {
    gaps.push({ from: open.from, to: open.to, worstStandardError: open.worst });
  }

  const difficulties = bank.map(itemLocation);
  const best = points.reduce((left, right) => (right.information > left.information ? right : left));

  return {
    items: bank.length,
    range: [lower, upper],
    target,
    points,
    gaps,
    covered: points.filter((point) => point.meetsTarget).length / points.length,
    difficulty: {
      mean: mean(difficulties),
      min: Math.min(...difficulties),
      max: Math.max(...difficulties),
    },
    meanDiscrimination: mean(bank.map(discriminationOf)),
    peak: best.theta,
    formats: formatCounts(bank),
  };
}

function padStart(text: string, width: number): string {
  return text.length >= width ? text : ' '.repeat(width - text.length) + text;
}

/** A fixed-width view of a bank health report. */
export function healthTable(health: BankHealth): string {
  const header =
    padStart('theta', 8) +
    padStart('info', 10) +
    padStart('se', 9) +
    padStart('items', 8) +
    '   target\n';
  let out = header + '-'.repeat(header.length + 1) + '\n';
  for (const point of health.points) {
    out +=
      padStart(point.theta.toFixed(2), 8) +
      padStart(point.information.toFixed(2), 10) +
      padStart(Number.isFinite(point.standardError) ? point.standardError.toFixed(3) : '-', 9) +
      padStart(String(point.itemsNearby), 8) +
      (point.meetsTarget ? '   met' : '   MISSED') +
      '\n';
  }
  return out;
}
