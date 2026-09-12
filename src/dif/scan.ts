/**
 * Scanning a whole bank for differential item functioning, and purifying the
 * matching criterion it is scanned against.
 *
 * One item at a time is the wrong unit of analysis, for a reason that is easy
 * to miss. The matching criterion is the total score, and the total score is
 * made of the items — so if three items in the form are biased against the focal
 * group, the criterion itself is biased against the focal group, and every
 * *unbiased* item will look slightly biased in the other direction when measured
 * against it. Contamination in the criterion spreads to every item's result.
 *
 * Purification is the standard answer: scan, remove the items that flag from the
 * criterion, scan again against what is left, and repeat until the flagged set
 * stops moving. The studied item always stays in its own criterion, flagged or
 * not, because removing it is a different and worse bias.
 */

import type { ResponseMatrix } from '../calibration/matrix.js';
import {
  mantelHaenszel,
  standardizedDifference,
  type DifCategory,
  type Favours,
  type MantelHaenszelResult,
  type StandardizedDifferenceResult,
} from './mantel.js';
import { matchedSample, stratify, type Group } from './strata.js';

const SEVERITY: Readonly<Record<DifCategory, number>> = { A: 0, B: 1, C: 2 };

export interface ScanOptions {
  /**
   * Items forming the matching criterion, before the studied item is added
   * back. Defaults to every item, which is an unpurified scan.
   */
  readonly anchor?: readonly number[];
  /** Keep the studied item in its own matching score. Default true. */
  readonly includeStudied?: boolean;
  /** Merge adjacent matching levels up to this many people. Default 1. */
  readonly minimumStratum?: number;
}

export interface DifRow {
  readonly item: number;
  readonly itemId: string;
  /** Null when the item could not be analysed at all; `note` says why. */
  readonly mantel: MantelHaenszelResult | null;
  readonly standardized: StandardizedDifferenceResult | null;
  readonly classification: DifCategory | null;
  readonly favours: Favours | null;
  /** Why the item was not analysed, or null when it was. */
  readonly note: string | null;
  /** Candidates dropped for an incomplete matching score or studied response. */
  readonly dropped: number;
}

export interface DifScan {
  readonly rows: readonly DifRow[];
  /** The criterion items the scan was run against, before the studied item. */
  readonly anchor: readonly number[];
  /** Items classified at or above the flagging threshold. */
  readonly flagged: readonly number[];
  /** How many items produced a result. */
  readonly analysed: number;
}

function criterionFor(
  anchor: readonly number[],
  studied: number,
  includeStudied: boolean,
): number[] {
  if (!includeStudied) return anchor.filter((index) => index !== studied);
  return anchor.includes(studied) ? [...anchor] : [...anchor, studied];
}

/**
 * Run the Mantel-Haenszel scan over every item in a bank.
 *
 * An item that cannot be analysed does not stop the scan. Three things
 * routinely make one unanalysable — no stratum holding both groups, an item
 * everyone got right, a criterion nobody completed — and all three are
 * properties of the sample rather than errors. They are recorded on the row so
 * that a reader can tell "no evidence of DIF" from "no evidence either way",
 * which is a distinction a bare table of classifications destroys.
 */
export function scanBank(
  matrix: ResponseMatrix,
  groups: readonly Group[],
  options: ScanOptions = {},
): DifScan {
  const includeStudied = options.includeStudied ?? true;
  const anchor = options.anchor ?? Array.from({ length: matrix.itemCount }, (_, i) => i);
  for (const index of anchor) {
    if (!Number.isInteger(index) || index < 0 || index >= matrix.itemCount) {
      throw new RangeError(`scanBank: anchor references no item at index ${index}`);
    }
  }

  const minimumStratum = options.minimumStratum ?? 1;
  const rows: DifRow[] = [];
  const flagged: number[] = [];
  let analysed = 0;

  for (let item = 0; item < matrix.itemCount; item += 1) {
    const itemId = matrix.itemIds[item] as string;
    const criterion = criterionFor(anchor, item, includeStudied);
    if (criterion.length === 0) {
      rows.push(blankRow(item, itemId, 'the matching criterion is empty for this item', 0));
      continue;
    }

    let dropped = 0;
    try {
      const sample = matchedSample(matrix, groups, item, {
        anchor: criterion,
        includeStudied: true,
      });
      dropped = sample.dropped;
      const strata = stratify(sample.observations, { maxScore: 1, minimumStratum });
      const mantel = mantelHaenszel(strata);
      const standardized = standardizedDifference(strata);
      analysed += 1;
      if (SEVERITY[mantel.classification] > 0) flagged.push(item);
      rows.push({
        item,
        itemId,
        mantel,
        standardized,
        classification: mantel.classification,
        favours: mantel.favours,
        note: null,
        dropped,
      });
    } catch (error) {
      rows.push(
        blankRow(item, itemId, error instanceof Error ? error.message : String(error), dropped),
      );
    }
  }

  return { rows, anchor: [...anchor], flagged, analysed };
}

function blankRow(item: number, itemId: string, note: string, dropped: number): DifRow {
  return {
    item,
    itemId,
    mantel: null,
    standardized: null,
    classification: null,
    favours: null,
    note,
    dropped,
  };
}

/** Rows that produced a result, ordered by effect size, largest first. */
export function rankByEffect(scan: DifScan): DifRow[] {
  return scan.rows
    .filter((row) => row.mantel !== null)
    .sort((a, b) => magnitude(b) - magnitude(a));
}

function magnitude(row: DifRow): number {
  const delta = row.mantel?.delta ?? 0;
  return Number.isFinite(delta) ? Math.abs(delta) : Number.POSITIVE_INFINITY;
}

/** Items classified at or above `threshold`, in item order. */
export function flaggedAt(scan: DifScan, threshold: DifCategory): number[] {
  const floor = SEVERITY[threshold];
  return scan.rows
    .filter((row) => row.classification !== null && SEVERITY[row.classification] >= floor)
    .map((row) => row.item);
}

export interface PurifyOptions extends ScanOptions {
  /** Rounds of re-scanning before giving up. Default 10. */
  readonly maxIterations?: number;
  /** Lowest category treated as contamination. Default 'C'. */
  readonly flagAt?: DifCategory;
  /**
   * Refuse to shrink the criterion below this many items. Default 2.
   *
   * A one-item matching criterion is not a matching criterion, and a run that
   * removes its way down to one has usually found a group difference in overall
   * ability rather than a bank full of biased items.
   */
  readonly minimumAnchor?: number;
}

export interface PurifyResult {
  /** The scan run against the final criterion. */
  readonly scan: DifScan;
  /** The purified criterion: every item not flagged in the final round. */
  readonly anchor: readonly number[];
  /** Rounds actually run. */
  readonly iterations: number;
  /** True when the flagged set repeated rather than the budget running out. */
  readonly converged: boolean;
  /** The flagged set after each round, oldest first. */
  readonly history: readonly (readonly number[])[];
  /** Set when purification stopped early; null when it ran to a fixed point. */
  readonly note: string | null;
}

/**
 * Iteratively purify the matching criterion and re-scan.
 *
 * Convergence is declared when a round flags exactly the set the previous round
 * flagged, which is a genuine fixed point: the same criterion produces the same
 * scan, so nothing further can change. A set that oscillates between two states
 * instead will exhaust the budget and be reported as unconverged rather than
 * quietly returning whichever round happened to be last — an oscillation means
 * the flags are not determined by the data alone, and a caller needs to know
 * that rather than receive a clean-looking answer.
 */
export function purifiedScan(
  matrix: ResponseMatrix,
  groups: readonly Group[],
  options: PurifyOptions = {},
): PurifyResult {
  const maxIterations = options.maxIterations ?? 10;
  const flagAt = options.flagAt ?? 'C';
  const minimumAnchor = options.minimumAnchor ?? 2;
  if (!Number.isInteger(maxIterations) || maxIterations < 1) {
    throw new RangeError(
      `purifiedScan: maxIterations must be a positive integer, received ${maxIterations}`,
    );
  }

  const everyItem = Array.from({ length: matrix.itemCount }, (_, i) => i);
  let clean = options.anchor === undefined ? everyItem : [...options.anchor];
  const history: number[][] = [];
  let previous: string | null = null;
  let scan = scanBank(matrix, groups, { ...options, anchor: clean });
  let iterations = 1;
  let converged = false;
  let note: string | null = null;

  for (;;) {
    const flagged = flaggedAt(scan, flagAt);
    history.push(flagged);
    const signature = flagged.join(',');
    if (signature === previous) {
      converged = true;
      break;
    }
    previous = signature;

    const next = everyItem.filter((item) => !flagged.includes(item));
    // A criterion identical to the one just used would produce an identical
    // scan, so this is already the fixed point — most obviously on a clean bank,
    // where the first round flags nothing and there is nothing to remove.
    if (next.length === clean.length && next.every((item, i) => item === clean[i])) {
      converged = true;
      break;
    }
    if (next.length < minimumAnchor) {
      note =
        `purification would leave ${next.length} item(s) in the criterion, ` +
        `below the floor of ${minimumAnchor}; kept the previous criterion`;
      break;
    }
    if (iterations >= maxIterations) {
      note = `the flagged set was still moving after ${maxIterations} rounds`;
      break;
    }

    clean = next;
    scan = scanBank(matrix, groups, { ...options, anchor: clean });
    iterations += 1;
  }

  return {
    scan,
    anchor: everyItem.filter((item) => !flaggedAt(scan, flagAt).includes(item)),
    iterations,
    converged,
    history,
    note,
  };
}
