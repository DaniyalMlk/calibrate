import { createRng } from '../core/random.js';
import { blueprint, contentBalanced } from '../selection/content.js';
import { randomesque } from '../selection/exposure.js';
import { maximumInformationSelector } from '../selection/information.js';
import { AdaptiveSession } from '../session/session.js';
import { precisionTarget } from '../session/stopping.js';
import { syntheticPool } from '../simulation/bank.js';
import { simulateResponse } from '../simulation/respondent.js';

const USAGE = `calibrate — adaptive testing engine

Usage:
  calibrate demo [options]

Options:
  --theta <number>    True ability of the simulated candidate (default 0.8)
  --target <number>   Standard-error target at which to stop (default 0.3)
  --min <integer>     Minimum test length (default 5)
  --max <integer>     Maximum test length (default 30)
  --bank <integer>    Synthetic bank size (default 300)
  --seed <integer>    Seed for the whole run (default 20260101)
  -h, --help          Show this message
`;

interface Options {
  readonly theta: number;
  readonly target: number;
  readonly min: number;
  readonly max: number;
  readonly bank: number;
  readonly seed: number;
}

function parseOptions(argv: readonly string[]): Options {
  const defaults: Options = { theta: 0.8, target: 0.3, min: 5, max: 30, bank: 300, seed: 20260101 };
  const parsed: Record<string, number> = {};
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i] as string;
    if (!flag.startsWith('--')) throw new Error(`unexpected argument "${flag}"`);
    const key = flag.slice(2);
    if (!(key in defaults)) throw new Error(`unknown option "${flag}"`);
    const raw = argv[i + 1];
    if (raw === undefined) throw new Error(`option "${flag}" needs a value`);
    const value = Number(raw);
    if (!Number.isFinite(value)) throw new Error(`option "${flag}" needs a number, got "${raw}"`);
    parsed[key] = value;
    i += 1;
  }
  return { ...defaults, ...parsed };
}

function pad(text: string, width: number): string {
  return text.length >= width ? text : text + ' '.repeat(width - text.length);
}

function padStart(text: string, width: number): string {
  return text.length >= width ? text : ' '.repeat(width - text.length) + text;
}

function runDemo(options: Options): void {
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
  const snapshot = session.run((item) => simulateResponse(item, options.theta, answers));

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
        padStart((item?.parameters.b ?? 0).toFixed(2), 7) +
        padStart((item?.parameters.a ?? 0).toFixed(2), 7) +
        padStart(entry.response === 1 ? 'right' : 'wrong', 7) +
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

export function main(argv: readonly string[]): number {
  const [command, ...rest] = argv;
  if (command === undefined || command === '-h' || command === '--help' || command === 'help') {
    process.stdout.write(USAGE);
    return command === undefined ? 1 : 0;
  }
  if (command !== 'demo') {
    process.stderr.write(`unknown command "${command}"\n\n${USAGE}`);
    return 1;
  }
  try {
    runDemo(parseOptions(rest));
    return 0;
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n\n${USAGE}`);
    return 1;
  }
}
