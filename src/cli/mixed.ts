import { createRng } from '../core/random.js';
import {
  categoryCountOf,
  discriminationOf,
  expectedScoreOf,
  formatCounts,
  isPolytomous,
  itemLocation,
  maximumScoreOf,
  type AnyItem,
} from '../models/mixed.js';
import { randomesque } from '../selection/exposure.js';
import { maximumInformationSelector } from '../selection/information.js';
import { AdaptiveSession } from '../session/session.js';
import { precisionTarget } from '../session/stopping.js';
import { syntheticMixedBank } from '../simulation/bank.js';
import { ItemPool } from '../session/pool.js';
import { simulateCategory } from '../simulation/respondent.js';
import { pad, padStart, scoreLabel } from './format.js';
import { parseOptions, type OptionSpecs } from './options.js';

export const MIXED_OPTIONS = {
  theta: { kind: 'number', fallback: 0.8, describe: 'True ability of the simulated candidate' },
  target: { kind: 'number', fallback: 0.3, describe: 'Standard-error target at which to stop' },
  min: { kind: 'integer', fallback: 5, describe: 'Minimum test length' },
  max: { kind: 'integer', fallback: 25, describe: 'Maximum test length' },
  bank: { kind: 'integer', fallback: 300, describe: 'Synthetic bank size' },
  polytomous: {
    kind: 'number',
    fallback: 0.25,
    describe: 'Fraction of the bank scored in more than two categories',
  },
  seed: { kind: 'integer', fallback: 20260101, describe: 'Seed for the whole run' },
} as const satisfies OptionSpecs;

/** Describe an item's format compactly: `2` for dichotomous, `4 cat` otherwise. */
function formatLabel(item: AnyItem | undefined): string {
  if (item === undefined) return '-';
  return isPolytomous(item) ? `${categoryCountOf(item)} cat` : 'binary';
}

/**
 * Run one adaptive session against a bank mixing both item formats.
 *
 * The point of the command is to make the mixed case visible: the transcript
 * carries the format and the score out of the score available, so a rubric item
 * scored 2 of 4 is legible as such rather than collapsed into a right/wrong
 * column that cannot represent it.
 */
export function runMixed(argv: readonly string[]): void {
  const options = parseOptions(argv, MIXED_OPTIONS);
  const bank = syntheticMixedBank({
    size: options.bank,
    polytomousFraction: options.polytomous,
    seed: options.seed,
  });
  const pool = new ItemPool(bank);
  const counts = formatCounts(bank);

  const session = new AdaptiveSession({
    pool,
    selector: randomesque(maximumInformationSelector(), 5),
    stopping: precisionTarget(options.target, { minimum: options.min, maximum: options.max }),
    seed: options.seed,
  });

  const answers = createRng(options.seed + 1);
  const snapshot = session.run((item) => simulateCategory(item, options.theta, answers));

  process.stdout.write(
    `Bank: ${pool.size} items — ${counts.dichotomous} binary, ` +
      `${counts.polytomous} scored in more than two categories\n` +
      `Candidate: true ability ${options.theta.toFixed(2)}\n` +
      `Stopping: standard error <= ${options.target}, ` +
      `between ${options.min} and ${options.max} items\n\n`,
  );

  const header =
    pad('#', 4) +
    pad('item', 28) +
    padStart('format', 8) +
    padStart('loc', 7) +
    padStart('a', 6) +
    padStart('score', 7) +
    padStart('theta', 9) +
    padStart('se', 8) +
    '  est\n';
  process.stdout.write(header);
  process.stdout.write('-'.repeat(header.length + 2) + '\n');

  let pointsEarned = 0;
  let pointsAvailable = 0;
  for (const entry of snapshot.transcript) {
    const item = pool.byId(entry.itemId);
    pointsEarned += entry.response;
    pointsAvailable += entry.maximumScore;
    process.stdout.write(
      pad(String(entry.position), 4) +
        pad(entry.itemId, 28) +
        padStart(formatLabel(item), 8) +
        padStart(item === undefined ? '-' : itemLocation(item).toFixed(2), 7) +
        padStart(item === undefined ? '-' : discriminationOf(item).toFixed(2), 6) +
        padStart(scoreLabel(entry.response, entry.maximumScore), 7) +
        padStart(entry.thetaAfter.toFixed(3), 9) +
        padStart(entry.standardErrorAfter.toFixed(3), 8) +
        '  ' +
        entry.method +
        '\n',
    );
  }

  const administered = session.administered;
  const polytomousAdministered = administered.filter(isPolytomous).length;
  // What the engine expected this candidate to score on exactly these items,
  // which is the test characteristic curve evaluated at the final estimate.
  let expected = 0;
  for (const item of administered) expected += expectedScoreOf(item, snapshot.theta);

  const error = snapshot.theta - options.theta;
  process.stdout.write(
    `\nFinished after ${snapshot.transcript.length} items ` +
      `(${polytomousAdministered} of them scored in more than two categories): ` +
      `${snapshot.stopReason?.detail ?? ''}\n` +
      `Raw score ${pointsEarned} of ${pointsAvailable} points; ` +
      `the model expected ${expected.toFixed(2)} at this ability\n` +
      `Estimate ${snapshot.theta.toFixed(3)} ` +
      `(true ${options.theta.toFixed(3)}, error ${error >= 0 ? '+' : ''}${error.toFixed(3)}, ` +
      `${Math.abs(error / snapshot.standardError).toFixed(2)} standard errors)\n` +
      `Standard error ${snapshot.standardError.toFixed(3)}\n`,
  );

  // A mixed form's points are not its item count, and conflating the two is the
  // most common way a mixed test gets misreported.
  const maximumPerItem = administered.map((item) => maximumScoreOf(item));
  const widest = Math.max(...maximumPerItem, 0);
  if (widest > 1) {
    process.stdout.write(
      `Item maxima ranged from 1 to ${widest} point${widest === 1 ? '' : 's'}, ` +
        `so ${snapshot.transcript.length} items were worth ${pointsAvailable} points.\n`,
    );
  }
}
