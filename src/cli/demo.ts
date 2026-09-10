import { createRng } from '../core/random.js';
import { discriminationOf, itemLocation } from '../models/mixed.js';
import { blueprint, contentBalanced } from '../selection/content.js';
import { randomesque } from '../selection/exposure.js';
import { maximumInformationSelector } from '../selection/information.js';
import { AdaptiveSession } from '../session/session.js';
import { precisionTarget } from '../session/stopping.js';
import { syntheticPool } from '../simulation/bank.js';
import { simulateCategory } from '../simulation/respondent.js';
import { pad, padStart, scoreLabel } from './format.js';
import { parseOptions, type OptionSpecs } from './options.js';

export const DEMO_OPTIONS = {
  theta: { kind: 'number', fallback: 0.8, describe: 'True ability of the simulated candidate' },
  target: { kind: 'number', fallback: 0.3, describe: 'Standard-error target at which to stop' },
  min: { kind: 'integer', fallback: 5, describe: 'Minimum test length' },
  max: { kind: 'integer', fallback: 30, describe: 'Maximum test length' },
  bank: { kind: 'integer', fallback: 300, describe: 'Synthetic bank size' },
  seed: { kind: 'integer', fallback: 20260101, describe: 'Seed for the whole run' },
} as const satisfies OptionSpecs;

/** Run one adaptive session against a synthetic bank and print its transcript. */
export function runDemo(argv: readonly string[]): void {
  const options = parseOptions(argv, DEMO_OPTIONS);
  const pool = syntheticPool({ size: options.bank, seed: options.seed });
  const selector = contentBalanced(
    randomesque(maximumInformationSelector(), 5),
    blueprint({ arrays: 0.4, graphs: 0.3, 'dynamic-programming': 0.3 }),
  );
  const session = new AdaptiveSession({
    pool,
    selector,
    stopping: precisionTarget(options.target, { minimum: options.min, maximum: options.max }),
    seed: options.seed,
  });

  const answers = createRng(options.seed + 1);
  const snapshot = session.run((item) => simulateCategory(item, options.theta, answers));

  process.stdout.write(
    `Bank: ${pool.size} items across ${pool.domains().length} domains ` +
      `(${pool.domains().join(', ')})\n` +
      `Candidate: true ability ${options.theta.toFixed(2)}\n` +
      `Stopping: standard error <= ${options.target}, ` +
      `between ${options.min} and ${options.max} items\n\n`,
  );

  const header =
    pad('#', 4) +
    pad('item', 26) +
    padStart('b', 7) +
    padStart('a', 7) +
    padStart('score', 7) +
    padStart('theta', 9) +
    padStart('se', 8) +
    '  est\n';
  process.stdout.write(header);
  process.stdout.write('-'.repeat(header.length + 2) + '\n');

  for (const entry of snapshot.transcript) {
    const item = pool.byId(entry.itemId);
    process.stdout.write(
      pad(String(entry.position), 4) +
        pad(entry.itemId, 26) +
        padStart(item === undefined ? '-' : itemLocation(item).toFixed(2), 7) +
        padStart(item === undefined ? '-' : discriminationOf(item).toFixed(2), 7) +
        padStart(scoreLabel(entry.response, entry.maximumScore), 7) +
        padStart(entry.thetaAfter.toFixed(3), 9) +
        padStart(entry.standardErrorAfter.toFixed(3), 8) +
        '  ' +
        entry.method +
        '\n',
    );
  }

  const error = snapshot.theta - options.theta;
  process.stdout.write(
    `\nFinished after ${snapshot.transcript.length} items: ${snapshot.stopReason?.detail ?? ''}\n` +
      `Estimate ${snapshot.theta.toFixed(3)} ` +
      `(true ${options.theta.toFixed(3)}, error ${error >= 0 ? '+' : ''}${error.toFixed(3)}, ` +
      `${Math.abs(error / snapshot.standardError).toFixed(2)} standard errors)\n` +
      `Standard error ${snapshot.standardError.toFixed(3)}\n`,
  );
}
