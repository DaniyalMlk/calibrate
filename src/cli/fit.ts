import { readFileSync } from 'node:fs';
import { correlation, mean } from '../core/numeric.js';
import { itemFit } from '../calibration/fit.js';
import { calibrate, toItems, type CalibrationModel, type CalibrationResult } from '../calibration/jmle.js';
import { parseResponseCsv, ResponseMatrix } from '../calibration/matrix.js';
import { bankHealth, healthTable } from '../calibration/health.js';
import type { Item } from '../models/item.js';
import { syntheticBank } from '../simulation/bank.js';
import { drawPopulation, normalPopulation } from '../simulation/population.js';
import { simulateMatrix } from '../simulation/respondent.js';
import { pad, padStart } from './format.js';
import { parseOptions, type OptionSpecs } from './options.js';

export const FIT_OPTIONS = {
  file: {
    kind: 'string',
    fallback: '',
    describe: 'CSV of responses; omit to calibrate a simulated bank instead',
  },
  index: { kind: 'integer', fallback: 0, describe: 'Read person ids from the first column (1 = yes)' },
  model: { kind: 'string', fallback: 'rasch', describe: 'Model to fit', choices: ['rasch', '2pl'] },
  respondents: { kind: 'integer', fallback: 800, describe: 'Simulated respondents, when simulating' },
  bank: { kind: 'integer', fallback: 200, describe: 'Simulated bank size, when simulating' },
  target: { kind: 'number', fallback: 0.35, describe: 'Standard error the bank should reach' },
  seed: { kind: 'integer', fallback: 20260101, describe: 'Seed for the simulation' },
} as const satisfies OptionSpecs;

/** "1 item", "3 items" — a report that says "1 items" reads as a broken one. */
function plural(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

interface Source {
  readonly matrix: ResponseMatrix;
  readonly description: string;
  /** Generating parameters, when the data were simulated and recovery can be shown. */
  readonly truth?: readonly Item[];
}

function load(options: ReturnType<typeof parseOptions<typeof FIT_OPTIONS>>): Source {
  if (options.file.length > 0) {
    const text = readFileSync(options.file, 'utf8');
    const matrix = parseResponseCsv(text, { index: options.index === 1 });
    return {
      matrix,
      description: `${options.file}: ${matrix.personCount} respondents, ${matrix.itemCount} items`,
    };
  }

  // Simulating 2PL data and fitting a Rasch model would be a demonstration of
  // misfit rather than of calibration, so the generating model follows the one
  // being fitted.
  const truth = syntheticBank({
    size: options.bank,
    seed: options.seed,
    ...(options.model === 'rasch'
      ? { discrimination: { logMean: 0, logSd: 0 } }
      : { discrimination: { logMean: Math.log(1.1), logSd: 0.3 } }),
    guessing: 0,
  });
  const abilities = drawPopulation(normalPopulation(), options.respondents, options.seed + 1);
  return {
    matrix: simulateMatrix(truth, abilities, options.seed + 2),
    description:
      `simulated: ${options.respondents} respondents, ${truth.length} ${options.model} items`,
    truth,
  };
}

/**
 * Compare estimated difficulties against the ones the data were generated from.
 *
 * Both sets are centred first. Joint estimation identifies the parameters only
 * up to the origin of the scale, which the calibrator pins by fixing the mean
 * difficulty at zero; the generating bank has no reason to share that origin.
 * Comparing the raw numbers would therefore report the arbitrary difference
 * between two conventions as though it were estimation error — on a short bank
 * it dominates everything else, and it is not an error at all.
 */
function recovery(result: CalibrationResult, truth: readonly Item[]): string {
  const byId = new Map(truth.map((item) => [item.id, item]));
  const estimates: number[] = [];
  const actual: number[] = [];
  for (const item of result.items) {
    const generating = byId.get(item.id);
    if (generating === undefined) continue;
    estimates.push(item.difficulty);
    actual.push(generating.parameters.b);
  }
  if (estimates.length === 0) return '';

  const me = mean(estimates);
  const ma = mean(actual);
  const centredEstimates = estimates.map((value) => value - me);
  const centredActual = actual.map((value) => value - ma);
  let absolute = 0;
  for (let i = 0; i < centredEstimates.length; i += 1) {
    absolute += Math.abs((centredEstimates[i] as number) - (centredActual[i] as number));
  }

  return (
    `Recovery against the generating difficulties, both centred: ` +
    `correlation ${correlation(centredEstimates, centredActual).toFixed(4)}, ` +
    `mean absolute error ${(absolute / estimates.length).toFixed(4)}\n`
  );
}

/** Calibrate a response matrix and report the parameters, the fit and the bank health. */
export function runFit(argv: readonly string[]): void {
  const options = parseOptions(argv, FIT_OPTIONS);
  const source = load(options);
  const model = options.model as CalibrationModel;

  const result = calibrate(source.matrix, { model, maxIterations: 400 });
  const fit = itemFit(source.matrix, result);

  process.stdout.write(
    `Source: ${source.description}\n` +
      `Model: ${model}, ` +
      `${result.converged ? `converged in ${result.iterations} iterations` : `did NOT converge in ${result.iterations} iterations`} ` +
      `(largest change ${result.maxChange.toExponential(2)})\n` +
      `Screened: ${plural(result.screening.excludedPersons.length, 'respondent')} and ` +
      `${plural(result.screening.excludedItems.length, 'item')} removed over ` +
      `${plural(result.screening.passes, 'pass', 'passes')}; ` +
      `${plural(result.personIds.length, 'respondent')} and ` +
      `${plural(result.items.length, 'item')} calibrated\n\n`,
  );

  const header =
    pad('item', 20) +
    padStart('p+', 7) +
    padStart('b', 8) +
    padStart('se(b)', 8) +
    padStart('a', 7) +
    padStart('infit', 8) +
    padStart('outfit', 8) +
    padStart('t(in)', 8) +
    '   flag\n';
  process.stdout.write(header + '-'.repeat(header.length + 1) + '\n');

  const misfitting = new Set(fit.misfitting.map((item) => item.id));
  for (const [index, item] of result.items.entries()) {
    const itemFitness = fit.items[index];
    process.stdout.write(
      pad(item.id, 20) +
        padStart(item.proportionCorrect.toFixed(3), 7) +
        padStart(item.difficulty.toFixed(3), 8) +
        padStart(item.difficultyStandardError.toFixed(3), 8) +
        padStart(item.discrimination.toFixed(2), 7) +
        padStart((itemFitness?.infit ?? Number.NaN).toFixed(3), 8) +
        padStart((itemFitness?.outfit ?? Number.NaN).toFixed(3), 8) +
        padStart((itemFitness?.infitT ?? Number.NaN).toFixed(2), 8) +
        (misfitting.has(item.id) ? '   MISFIT' : '') +
        '\n',
    );
  }

  process.stdout.write(
    `\n${fit.misfitting.length} of ${fit.items.length} items outside ` +
      `[${fit.range[0]}, ${fit.range[1]}] on infit or outfit\n`,
  );
  if (source.truth !== undefined) process.stdout.write(recovery(result, source.truth));

  const health = bankHealth(toItems(result), { target: options.target });
  process.stdout.write(
    `\nBank health at a target standard error of ${health.target}: ` +
      `${(health.covered * 100).toFixed(0)}% of [-3, 3] covered, ` +
      `most informative at ${health.peak.toFixed(2)}\n`,
  );
  for (const gap of health.gaps) {
    process.stdout.write(
      `  gap from ${gap.from.toFixed(2)} to ${gap.to.toFixed(2)}, ` +
        `worst standard error ${gap.worstStandardError.toFixed(3)}\n`,
    );
  }
  if (health.gaps.length === 0) process.stdout.write('  no gaps\n');
  process.stdout.write('\n' + healthTable(health));
}
