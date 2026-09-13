import { readFileSync } from 'node:fs';
import { correlation, mean } from '../core/numeric.js';
import { normalGaussHermiteRule } from '../core/quadrature.js';
import { marginalLogLikelihood, type ItemPoint } from '../calibration/expected.js';
import { calibrate, type CalibrationModel } from '../calibration/jmle.js';
import { parseResponseCsv, ResponseMatrix } from '../calibration/matrix.js';
import {
  marginalCalibrate,
  type ErrorMethod,
  type LatentDistribution,
  type MarginalResult,
} from '../calibration/mml.js';
import type { Item } from '../models/item.js';
import { toMarginalItems } from '../calibration/mml.js';
import { marginalScoreDistribution } from '../scoring/summed.js';
import { conversionTable, marginalReliability } from '../scoring/score.js';
import { syntheticBank } from '../simulation/bank.js';
import { drawPopulation, normalPopulation } from '../simulation/population.js';
import { simulateMatrix } from '../simulation/respondent.js';
import { pad, padStart } from './format.js';
import { parseOptions, type OptionSpecs } from './options.js';

export const MARGINAL_OPTIONS = {
  file: {
    kind: 'string',
    fallback: '',
    describe: 'CSV of responses; omit to calibrate a simulated bank instead',
  },
  index: {
    kind: 'integer',
    fallback: 0,
    describe: 'Read person ids from the first column (1 = yes)',
  },
  model: { kind: 'string', fallback: 'rasch', describe: 'Model to fit', choices: ['rasch', '2pl'] },
  latent: {
    kind: 'string',
    fallback: 'normal',
    describe: 'Population distribution',
    choices: ['normal', 'empirical'],
  },
  errors: {
    kind: 'string',
    fallback: 'cross-product',
    describe: 'Standard error method',
    choices: ['cross-product', 'expected-counts'],
  },
  points: { kind: 'integer', fallback: 41, describe: 'Quadrature nodes standing in for ability' },
  respondents: {
    kind: 'integer',
    fallback: 1500,
    describe: 'Simulated respondents, when simulating',
  },
  bank: { kind: 'integer', fallback: 20, describe: 'Simulated bank size, when simulating' },
  seed: { kind: 'integer', fallback: 20260101, describe: 'Seed for the simulation' },
  score: {
    kind: 'integer',
    fallback: 1,
    describe: 'Report the score conversion table and reliability (1 = yes)',
  },
} as const satisfies OptionSpecs;

interface Source {
  readonly matrix: ResponseMatrix;
  readonly description: string;
  readonly truth?: readonly Item[];
}

function load(options: ReturnType<typeof parseOptions<typeof MARGINAL_OPTIONS>>): Source {
  if (options.file.length > 0) {
    const text = readFileSync(options.file, 'utf8');
    const matrix = parseResponseCsv(text, { index: options.index === 1 });
    return {
      matrix,
      description: `${options.file}: ${matrix.personCount} respondents, ${matrix.itemCount} items`,
    };
  }

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
    description: `simulated: ${options.respondents} respondents, ${truth.length} ${options.model} items`,
    truth,
  };
}

/** Standard deviation of a sample, about its own mean. */
function spread(values: readonly number[]): number {
  const centre = mean(values);
  let acc = 0;
  for (const value of values) acc += (value - centre) * (value - centre);
  return Math.sqrt(acc / values.length);
}

interface Comparison {
  readonly correlation: number;
  readonly meanAbsoluteError: number;
  /** Estimated spread divided by generating spread: 1 is unbiased, above 1 inflated. */
  readonly spreadRatio: number;
}

/**
 * Compare a set of difficulty estimates against the values they came from.
 *
 * Both sets are centred first, because neither estimator pins the origin of the
 * scale to anything the generating bank knows about. What is *not* removed is
 * the scale's unit: the joint estimator's characteristic defect is that it
 * stretches the scale, and dividing that out before measuring it would hide
 * exactly the thing worth showing.
 */
function compare(
  estimates: readonly { id: string; difficulty: number }[],
  truth: readonly Item[],
): Comparison | null {
  const byId = new Map(truth.map((item) => [item.id, item.parameters.b]));
  const estimated: number[] = [];
  const actual: number[] = [];
  for (const item of estimates) {
    const generating = byId.get(item.id);
    if (generating === undefined) continue;
    estimated.push(item.difficulty);
    actual.push(generating);
  }
  if (estimated.length < 2) return null;

  const centreE = mean(estimated);
  const centreA = mean(actual);
  let absolute = 0;
  for (let i = 0; i < estimated.length; i += 1) {
    absolute += Math.abs((estimated[i] as number) - centreE - ((actual[i] as number) - centreA));
  }
  return {
    correlation: correlation(estimated, actual),
    meanAbsoluteError: absolute / estimated.length,
    spreadRatio: spread(estimated) / spread(actual),
  };
}

function describePopulation(result: MarginalResult): string {
  const skew = result.populationSkewness;
  const kurtosis = result.populationExcessKurtosis;
  const shape =
    Math.abs(skew) < 0.15 && Math.abs(kurtosis) < 0.3
      ? 'close to normal'
      : `${skew > 0 ? 'skewed towards high ability' : 'skewed towards low ability'}, ` +
        `${kurtosis > 0 ? 'heavier-tailed than normal' : 'lighter-tailed than normal'}`;
  return (
    `Fitted population: skewness ${skew.toFixed(3)}, ` +
    `excess kurtosis ${kurtosis.toFixed(3)} — ${shape}\n`
  );
}


/**
 * What the calibrated form does, as opposed to what its items are.
 *
 * Reported from the fitted items against the fitted population, so it answers
 * the question a programme actually asks of a new form: if this cohort sits it,
 * what totals come out, what ability does each total convert to, and how much of
 * the spread in ability does the form recover.
 */
function reportScoring(result: MarginalResult): string {
  const items = toMarginalItems(result);
  const population = result.population;
  const rows = conversionTable(items, population);
  const distribution = marginalScoreDistribution(items, population);
  const reliability = marginalReliability(items, population);

  let out =
    `\nThe form as a whole, on the fitted population:\n` +
    `  marginal reliability ${reliability.marginal.toFixed(4)} ` +
    `(average standard error ${reliability.meanStandardError.toFixed(3)} logits)\n\n` +
    `  ${pad('score', 7)}${padStart('theta', 9)}${padStart('se', 8)}${padStart('share', 9)}\n` +
    `  ${'-'.repeat(31)}\n`;

  for (const row of rows) {
    out +=
      `  ${pad(String(row.score), 7)}` +
      `${padStart(row.ability.toFixed(3), 9)}` +
      `${padStart(row.standardError.toFixed(3), 8)}` +
      `${padStart(`${(distribution[row.score] as number * 100).toFixed(1)}%`, 9)}\n`;
  }

  if (result.model === 'rasch') {
    out +=
      `\n  Under the Rasch model the total score is sufficient for ability, so this\n` +
      `  table loses nothing: two candidates with the same total have the same\n` +
      `  posterior whichever items they answered.\n`;
  } else {
    out +=
      `\n  Under the 2PL the total score is not sufficient — a candidate who passed\n` +
      `  the sharper items knows more than one who passed the flatter ones — so\n` +
      `  reporting from the total gives something up in exchange for being\n` +
      `  explicable.\n`;
  }
  return out;
}

/**
 * Calibrate the same responses jointly and marginally, and show what differs.
 *
 * Running both is the point of the command. The two estimators agree closely on
 * which items are hard, and the interesting disagreement is about how hard: the
 * joint estimator stretches the scale by a factor that depends on the test
 * length and nothing else, so on a short test the two sets of difficulties
 * correlate at better than 0.99 and still differ by a tenth of a logit at the
 * ends. The last line settles which is fitting the data better, on the only
 * objective both can be scored against.
 */
export function runMarginal(argv: readonly string[]): void {
  const options = parseOptions(argv, MARGINAL_OPTIONS);
  const source = load(options);
  const model = options.model as CalibrationModel;
  const rule = normalGaussHermiteRule(0, 1, options.points);

  const marginal = marginalCalibrate(source.matrix, {
    model,
    rule,
    latent: options.latent as LatentDistribution,
    errors: options.errors as ErrorMethod,
  });
  const joint = calibrate(source.matrix, { model, biasCorrection: false, maxIterations: 400 });

  process.stdout.write(
    `Source: ${source.description}\n` +
      `Model: ${model}, ${options.points}-point Gauss-Hermite quadrature, ` +
      `${options.latent} population\n` +
      `Marginal: ${
        marginal.converged
          ? `converged in ${marginal.iterations} cycles`
          : `did NOT converge in ${marginal.iterations} cycles`
      }, log-likelihood ${marginal.logLikelihood.toFixed(3)}\n` +
      `Screened: ${marginal.screening.excludedItems.length} item(s) removed, ` +
      `all ${source.matrix.personCount} respondents kept ` +
      `(joint estimation had to drop ${joint.screening.excludedPersons.length})\n`,
  );
  if (options.latent === 'empirical') process.stdout.write(describePopulation(marginal));
  process.stdout.write('\n');

  const jointById = new Map(joint.items.map((item) => [item.id, item]));

  // The two estimators pin the origin of the scale to different things — joint
  // estimation to a mean difficulty of zero, marginal estimation to a
  // population mean of zero — so the raw gap between them is mostly a constant
  // neither of them claims is meaningful. Centring both first leaves the part
  // that is: how differently they stretch the scale.
  const shared = marginal.items.filter((item) => jointById.has(item.id));
  const marginalCentre = mean(shared.map((item) => item.difficulty));
  const jointCentre = mean(
    shared.map((item) => (jointById.get(item.id) as { difficulty: number }).difficulty),
  );

  const width = Math.max(12, ...marginal.items.map((item) => item.id.length)) + 2;
  const header =
    pad('item', width) +
    padStart('p+', 7) +
    padStart('b(mml)', 9) +
    padStart('se', 7) +
    padStart('b(jml)', 9) +
    padStart('diff*', 8) +
    (model === '2pl' ? padStart('a(mml)', 8) + padStart('a(jml)', 8) : '') +
    '\n';
  process.stdout.write(header + '-'.repeat(header.length - 1) + '\n');

  for (const item of marginal.items) {
    const other = jointById.get(item.id);
    const centredDifference =
      other === undefined
        ? null
        : item.difficulty - marginalCentre - (other.difficulty - jointCentre);
    process.stdout.write(
      pad(item.id, width) +
        padStart(item.proportionCorrect.toFixed(3), 7) +
        padStart(item.difficulty.toFixed(3), 9) +
        padStart(item.difficultyStandardError.toFixed(3), 7) +
        padStart(other === undefined ? '-' : other.difficulty.toFixed(3), 9) +
        padStart(centredDifference === null ? '-' : centredDifference.toFixed(3), 8) +
        (model === '2pl'
          ? padStart(item.discrimination.toFixed(2), 8) +
            padStart(other === undefined ? '-' : other.discrimination.toFixed(2), 8)
          : '') +
        '\n',
    );
  }

  process.stdout.write(
    `\n* diff is the gap after centring both sets of difficulties, since the two\n` +
      `  estimators fix the origin of the scale differently: joint estimation at a\n` +
      `  mean difficulty of zero, marginal estimation at a population mean of zero.\n`,
  );

  const pinned = marginal.items.filter((item) => item.atBound);
  if (pinned.length > 0) {
    process.stdout.write(
      `\n${pinned.length} item(s) finished pinned to a parameter bound: ` +
        `${pinned.map((item) => item.id).join(', ')}\n`,
    );
  }

  if (source.truth !== undefined) {
    const marginalFit = compare(marginal.items, source.truth);
    const jointFit = compare(joint.items, source.truth);
    if (marginalFit !== null && jointFit !== null) {
      process.stdout.write('\nAgainst the generating difficulties, both centred:\n');
      process.stdout.write(
        `  ${pad('', 10)}${padStart('corr', 8)}${padStart('mae', 8)}${padStart('spread', 9)}\n` +
          `  ${pad('marginal', 10)}${padStart(marginalFit.correlation.toFixed(4), 8)}` +
          `${padStart(marginalFit.meanAbsoluteError.toFixed(4), 8)}` +
          `${padStart(marginalFit.spreadRatio.toFixed(4), 9)}\n` +
          `  ${pad('joint', 10)}${padStart(jointFit.correlation.toFixed(4), 8)}` +
          `${padStart(jointFit.meanAbsoluteError.toFixed(4), 8)}` +
          `${padStart(jointFit.spreadRatio.toFixed(4), 9)}\n`,
      );
      process.stdout.write(
        `  A spread ratio above one is a stretched scale. Uncorrected joint estimation\n` +
          `  stretches it by about L/(L-1), which on ${source.matrix.itemCount} items is ` +
          `${(source.matrix.itemCount / (source.matrix.itemCount - 1)).toFixed(3)}.\n`,
      );
    }
  }

  // Both estimators produce item parameters; only one of them was chosen to
  // maximise the marginal likelihood, and it should therefore win on it. This
  // is a check on the implementation as much as a report: a marginal estimator
  // that loses here has not converged to what it claims to.
  const jointPoints: ItemPoint[] = marginal.items.map((item) => {
    const other = jointById.get(item.id);
    return {
      discrimination: other?.discrimination ?? item.discrimination,
      difficulty: other?.difficulty ?? item.difficulty,
    };
  });
  const jointMarginal = marginalLogLikelihood(marginal.screening.matrix, jointPoints, rule);
  process.stdout.write(
    `\nMarginal log-likelihood of each solution, on the same rule:\n` +
      `  marginal estimates ${marginal.logLikelihood.toFixed(3)}\n` +
      `  joint estimates    ${jointMarginal.toFixed(3)}` +
      ` (${(marginal.logLikelihood - jointMarginal).toFixed(3)} worse)\n`,
  );

  if (options.score === 1) process.stdout.write(reportScoring(marginal));
}
