import { calibrate, toItems } from '../calibration/jmle.js';
import type { ResponseMatrix } from '../calibration/matrix.js';
import { areaScan, rankByArea, type AreaRow } from '../dif/area.js';
import { logisticDif, type LogisticDifResult } from '../dif/logistic.js';
import { matchedSample, groupSubmatrix, type Group } from '../dif/strata.js';
import { commonItems } from '../linking/common.js';
import { stockingLord } from '../linking/methods.js';
import { transformBank } from '../linking/scale.js';
import type { Item } from '../models/item.js';
import { isPolytomous } from '../models/mixed.js';
import { pad, padStart } from './format.js';

/**
 * The regression pass, run against the purified criterion.
 *
 * Reported beside the pooled statistic rather than instead of it. The two
 * disagree in a specific and informative way: they agree closely on an item that
 * is uniformly harder, and diverge on an item whose curves cross, because only
 * one of them has a term for the crossing.
 */
export function writeLogisticSection(
  matrix: ResponseMatrix,
  groups: readonly Group[],
  bank: readonly Item[],
  anchor: readonly number[],
  planted: ReadonlySet<number>,
): Map<number, LogisticDifResult> {
  const results = new Map<number, LogisticDifResult>();
  for (let item = 0; item < matrix.itemCount; item += 1) {
    const criterion = anchor.includes(item) ? [...anchor] : [...anchor, item];
    try {
      const sample = matchedSample(matrix, groups, item, { anchor: criterion });
      results.set(item, logisticDif(sample.observations));
    } catch {
      // Same tolerance as the pooled scan: an item that cannot be analysed is
      // left out of the table rather than ending the command.
    }
  }

  const header =
    pad('item', 9) +
    padStart('uniform', 10) +
    padStart('p', 11) +
    padStart('crossing', 10) +
    padStart('p', 11) +
    padStart('dR2', 8) +
    pad('  class', 8) +
    'planted\n';
  process.stdout.write(header);
  process.stdout.write('-'.repeat(header.length + 1) + '\n');

  const ordered = [...results.entries()].sort(
    (a, b) => b[1].combined.chiSquare - a[1].combined.chiSquare,
  );
  for (const [item, result] of ordered.slice(0, 10)) {
    process.stdout.write(
      pad(bank[item]?.id ?? `#${item}`, 9) +
        padStart(result.uniform.chiSquare.toFixed(1), 10) +
        padStart(tail(result.uniform.pValue), 11) +
        padStart(result.nonUniform.chiSquare.toFixed(1), 10) +
        padStart(tail(result.nonUniform.pValue), 11) +
        padStart(result.deltaRSquared.toFixed(4), 8) +
        pad(`  ${result.classification}`, 8) +
        (planted.has(item) ? 'yes' : '') +
        '\n',
    );
  }

  const unconverged = [...results.entries()].filter(([, result]) => !result.converged);
  if (unconverged.length > 0) {
    process.stdout.write(
      `\n${unconverged.length} fit(s) did not converge and are not to be read as findings: ` +
        `${unconverged.map(([item]) => bank[item]?.id ?? `#${item}`).join(', ')}\n`,
    );
  }
  return results;
}

function tail(p: number): string {
  return p < 1e-4 ? p.toExponential(1) : p.toFixed(4);
}

export interface AreaSection {
  readonly rows: AreaRow[];
  readonly slope: number;
  readonly intercept: number;
  readonly anchorSize: number;
  readonly converged: boolean;
}

/**
 * Calibrate the item bank separately in each group, link, and measure the areas.
 *
 * The only method here that never touches a matching score, and therefore the
 * only one immune to the contamination purification exists to undo. It pays for
 * that with a different dependency: the two calibrations have to be put on one
 * metric first, and the linking needs items believed free of DIF to do it. The
 * purified criterion is exactly that set, so the two halves of the analysis feed
 * each other.
 */
export function computeAreas(
  matrix: ResponseMatrix,
  groups: readonly Group[],
  anchor: readonly number[],
): AreaSection | null {
  if (anchor.length < 2) return null;
  const referenceFit = calibrate(groupSubmatrix(matrix, groups, 'reference'), { model: '2pl' });
  const focalFit = calibrate(groupSubmatrix(matrix, groups, 'focal'), { model: '2pl' });

  const referenceBank = toItems(referenceFit);
  const focalBank = toItems(focalFit);
  const anchorIds = new Set(anchor.map((index) => matrix.itemIds[index] as string));
  const pairs = commonItems(
    focalBank.filter((item) => anchorIds.has(item.id)),
    referenceBank.filter((item) => anchorIds.has(item.id)),
  );
  if (pairs.length < 2) return null;

  const link = stockingLord(pairs);
  // A 2PL calibration produces only dichotomous items, but the transform is
  // declared over either format, so the narrowing has to be stated.
  const linked = transformBank(link.transform, focalBank).filter(
    (item): item is Item => !isPolytomous(item),
  );
  return {
    rows: rankByArea(areaScan(referenceBank, linked)),
    slope: link.transform.slope,
    intercept: link.transform.intercept,
    anchorSize: pairs.length,
    converged: referenceFit.converged && focalFit.converged && link.converged,
  };
}

export function writeAreaSection(section: AreaSection, planted: ReadonlySet<string>): void {
  const header =
    pad('item', 9) +
    padStart('signed', 9) +
    padStart('unsigned', 10) +
    padStart('crossing', 10) +
    padStart('share', 8) +
    '  planted\n';
  process.stdout.write(header);
  process.stdout.write('-'.repeat(header.length + 1) + '\n');
  for (const row of section.rows.slice(0, 10)) {
    process.stdout.write(
      pad(row.itemId, 9) +
        padStart(row.area.signed.toFixed(3), 9) +
        padStart(row.area.unsigned.toFixed(3), 10) +
        padStart(row.area.crossing === null ? '-' : row.area.crossing.toFixed(2), 10) +
        padStart(row.area.nonUniformShare.toFixed(2), 8) +
        (planted.has(row.itemId) ? '  yes' : '') +
        '\n',
    );
  }
}
