import { flaggedAt, purifiedScan, rankByEffect, scanBank, type DifScan } from '../dif/scan.js';
import { makeItem, twoPL, type Item } from '../models/item.js';
import { simulateDif, type DifShift } from '../simulation/dif.js';
import { pad, padStart } from './format.js';
import { parseOptions, type OptionSpecs } from './options.js';

export const DIF_OPTIONS = {
  items: { kind: 'integer', fallback: 18, describe: 'Items in the form' },
  sample: { kind: 'integer', fallback: 2500, describe: 'Candidates in each group' },
  impact: {
    kind: 'number',
    fallback: -0.6,
    describe: 'Genuine ability difference of the focal group, in logits',
  },
  biased: { kind: 'integer', fallback: 6, describe: 'Items given a difficulty shift' },
  shift: {
    kind: 'number',
    fallback: 1.6,
    describe: 'How much harder a biased item is for the focal group, in logits',
  },
  crossing: {
    kind: 'integer',
    fallback: 1,
    describe: 'Items given a flattened curve instead, producing non-uniform DIF',
  },
  seed: { kind: 'integer', fallback: 20260401, describe: 'Seed for the whole run' },
} as const satisfies OptionSpecs;

/**
 * Scan a simulated two-group administration for differential item functioning.
 *
 * The command is built around the two things that make DIF analysis harder than
 * it looks. The focal group is genuinely less able, so every item is answered
 * correctly less often by them and none of that is bias; and the biased items
 * contaminate the very score used to match candidates, so an unpurified scan
 * flags items that are perfectly fair. Both are visible in the output because
 * the planted truth is printed beside what the scan found.
 */
export function runDif(argv: readonly string[]): void {
  const options = parseOptions(argv, DIF_OPTIONS);
  if (options.items < 4) {
    throw new RangeError(`items must be at least 4, received ${options.items}`);
  }
  if (options.sample < 50) {
    throw new RangeError(`sample must be at least 50 per group, received ${options.sample}`);
  }
  if (options.biased < 0 || options.crossing < 0) {
    throw new RangeError('biased and crossing counts must not be negative');
  }
  if (options.biased + options.crossing >= options.items) {
    throw new RangeError(
      `${options.biased + options.crossing} shifted items in a form of ${options.items}: ` +
        'leave some items unbiased to match candidates on',
    );
  }

  const size = options.items;
  const bank: Item[] = Array.from({ length: size }, (_, index) =>
    makeItem(
      `q-${String(index).padStart(3, '0')}`,
      twoPL(0.9 + 0.04 * (index % 7), -1.9 + (3.8 * index) / (size - 1)),
    ),
  );

  // Spread the shifted items through the form rather than clustering them, so
  // the contamination of the matching score is not confined to one difficulty
  // band where it would be easy to spot by eye.
  const shifts: DifShift[] = [];
  const stride = Math.max(1, Math.floor(size / (options.biased + options.crossing + 1)));
  let slot = stride;
  for (let i = 0; i < options.biased; i += 1, slot += stride) {
    shifts.push({ item: slot % size, difficulty: options.shift });
  }
  const crossingItems: number[] = [];
  for (let i = 0; i < options.crossing; i += 1, slot += stride) {
    shifts.push({ item: slot % size, discrimination: 0.35 });
    crossingItems.push(slot % size);
  }

  const simulation = simulateDif({
    bank,
    referenceCount: options.sample,
    focalCount: options.sample,
    focalMean: options.impact,
    shifts,
    seed: options.seed,
  });

  const uniformItems = shifts
    .filter((shift) => shift.difficulty !== undefined)
    .map((shift) => shift.item)
    .sort((a, b) => a - b);

  process.stdout.write(
    `Form: ${size} items, ${options.sample} candidates per group\n` +
      `Impact: the focal group is ${Math.abs(options.impact).toFixed(2)} logits ` +
      `${options.impact < 0 ? 'less' : 'more'} able — a real difference, not bias\n` +
      `Planted uniform DIF (${options.shift.toFixed(2)} logits harder for the focal group): ` +
      `${describeItems(bank, uniformItems)}\n` +
      `Planted non-uniform DIF (flattened curve): ${describeItems(bank, crossingItems)}\n\n`,
  );

  const plain = scanBank(simulation.matrix, simulation.groups);
  const pure = purifiedScan(simulation.matrix, simulation.groups);

  process.stdout.write('Scanned against the whole form:\n\n');
  writeTable(plain, bank, new Set(shifts.map((shift) => shift.item)));

  const plainLarge = flaggedAt(plain, 'C');
  const pureLarge = flaggedAt(pure.scan, 'C');
  const planted = new Set(shifts.map((shift) => shift.item));
  const falsePositives = plainLarge.filter((item) => !planted.has(item));

  process.stdout.write(`\nLarge effects against the whole form: ${describeItems(bank, plainLarge)}\n`);
  process.stdout.write(
    falsePositives.length === 0
      ? 'All of those were planted. The criterion held up here; with more biased items,\n' +
          'or a larger shift, it would not have.\n\n'
      : `Of those, ${describeItems(bank, falsePositives)} ` +
          `${falsePositives.length === 1 ? 'was' : 'were'} never touched.\n` +
          'The biased items are part of the score used to match candidates, so the\n' +
          'criterion itself leans, and fair items measured against a leaning criterion\n' +
          'lean the other way.\n\n',
  );

  process.stdout.write(
    `Purified in ${pure.iterations} round${pure.iterations === 1 ? '' : 's'} ` +
      `(${pure.converged ? 'converged' : 'did not converge'}), ` +
      `criterion now ${pure.anchor.length} items:\n\n`,
  );
  writeTable(pure.scan, bank, new Set(shifts.map((shift) => shift.item)));
  process.stdout.write(`\nLarge effects after purification: ${describeItems(bank, pureLarge)}\n`);
  if (pure.note !== null) process.stdout.write(`note: ${pure.note}\n`);

  const recoveredUniform = uniformItems.filter((item) => pureLarge.includes(item));
  const recoveredCrossing = crossingItems.filter((item) => pureLarge.includes(item));
  const spurious = pureLarge.filter((item) => !planted.has(item));
  process.stdout.write(
    `\nAgainst the planted truth: ${recoveredUniform.length} of ${uniformItems.length} ` +
      `uniform effects recovered as large, ` +
      `${recoveredCrossing.length} of ${crossingItems.length} non-uniform, ` +
      `${spurious.length} spurious.\n`,
  );

  if (crossingItems.length > 0) reportCrossing(bank, crossingItems, pure.scan, options);
}

/**
 * What the scan made of the items whose response curves cross.
 *
 * Worth its own passage because the answer is not fixed. A crossing item
 * favours one group below the crossing point and the other above it, and a
 * pooled odds ratio averages those two advantages together. How much survives
 * depends entirely on where the two groups sit relative to the crossing point —
 * which is why the same item can be invisible in one administration and glaring
 * in the next. The comparison run below removes the ability difference and
 * leaves the crossing item alone, so the two numbers differ only in that.
 */
function reportCrossing(
  bank: readonly Item[],
  crossingItems: readonly number[],
  scan: DifScan,
  options: { readonly sample: number; readonly seed: number },
): void {
  const worst = crossingItems.reduce(
    (best, item) => Math.max(best, Math.abs(scan.rows[item]?.mantel?.delta ?? 0)),
    0,
  );
  const flaggedHere = crossingItems.filter((item) => scan.rows[item]?.classification === 'C');

  const balanced = simulateDif({
    bank,
    referenceCount: options.sample,
    focalCount: options.sample,
    focalMean: 0,
    shifts: crossingItems.map((item) => ({ item, discrimination: 0.35 })),
    seed: options.seed + 1,
  });
  const balancedScan = scanBank(balanced.matrix, balanced.groups);
  const balancedWorst = crossingItems.reduce(
    (best, item) => Math.max(best, Math.abs(balancedScan.rows[item]?.mantel?.delta ?? 0)),
    0,
  );

  const plural = crossingItems.length === 1 ? '' : 's';
  process.stdout.write(
    `\nThe non-uniform item${plural} (${describeItems(bank, crossingItems)}) reached ` +
      `${worst.toFixed(2)} delta here, and ${flaggedHere.length === 0 ? 'none' : String(flaggedHere.length)} ` +
      `of ${crossingItems.length} ${flaggedHere.length === 1 ? 'was' : 'were'} called large.\n` +
      `Running the same item${plural} again with the ability difference removed — no impact,\n` +
      `nothing else shifted — reaches only ${balancedWorst.toFixed(2)}.\n\n` +
      'The crossing point is what makes the difference. With the groups sitting on top\n' +
      'of each other the two advantages cancel in the pooled ratio and the item nearly\n' +
      'disappears; with the focal group displaced down the scale they are sampled on\n' +
      'opposite sides of the crossing and much of the effect reads as a uniform one.\n' +
      'A matched-group method cannot tell those two situations apart, which is the\n' +
      'argument for a method that models the interaction directly.\n',
  );
}

function describeItems(bank: readonly Item[], items: readonly number[]): string {
  if (items.length === 0) return 'none';
  return items.map((item) => bank[item]?.id ?? `#${item}`).join(', ');
}

/** One line per item, with the largest effects first. */
function writeTable(scan: DifScan, bank: readonly Item[], planted: ReadonlySet<number>): void {
  const header =
    pad('item', 9) +
    padStart('delta', 8) +
    padStart('chi2', 9) +
    padStart('p', 11) +
    padStart('std', 8) +
    pad('  class', 8) +
    pad('favours', 11) +
    'planted\n';
  process.stdout.write(header);
  process.stdout.write('-'.repeat(header.length + 1) + '\n');

  for (const row of rankByEffect(scan).slice(0, 12)) {
    const mantel = row.mantel;
    if (mantel === null) continue;
    process.stdout.write(
      pad(bank[row.item]?.id ?? row.itemId, 9) +
        padStart(finite(mantel.delta), 8) +
        padStart(mantel.chiSquare.toFixed(1), 9) +
        padStart(mantel.pValue < 1e-4 ? mantel.pValue.toExponential(1) : mantel.pValue.toFixed(4), 11) +
        padStart((row.standardized?.value ?? 0).toFixed(3), 8) +
        pad(`  ${mantel.classification}`, 8) +
        pad(mantel.favours, 11) +
        (planted.has(row.item) ? 'yes' : '') +
        '\n',
    );
  }

  const unanalysed = scan.rows.filter((row) => row.mantel === null);
  if (unanalysed.length > 0) {
    process.stdout.write(
      `\n${unanalysed.length} item(s) could not be analysed: ` +
        `${unanalysed.map((row) => row.itemId).join(', ')}\n`,
    );
  }
}

function finite(value: number): string {
  if (value === Number.POSITIVE_INFINITY) return '+inf';
  if (value === Number.NEGATIVE_INFINITY) return '-inf';
  return value.toFixed(3);
}
