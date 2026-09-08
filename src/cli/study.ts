import { blueprint, contentBalanced } from '../selection/content.js';
import { randomesque } from '../selection/exposure.js';
import { kullbackLeiblerSelector, maximumInformationSelector } from '../selection/information.js';
import { fixedLength, precisionTarget } from '../session/stopping.js';
import { syntheticPool } from '../simulation/bank.js';
import {
  drawPopulation,
  evenGridPopulation,
  normalPopulation,
  uniformPopulation,
  type AbilityDistribution,
} from '../simulation/population.js';
import { comparisonTable, conditionalTable, summarise } from '../simulation/report.js';
import { compareStudies, type Policy } from '../simulation/study.js';
import { parseOptions, type OptionSpecs } from './options.js';

export const STUDY_OPTIONS = {
  examinees: { kind: 'integer', fallback: 200, describe: 'Simulated examinees per policy' },
  population: {
    kind: 'string',
    fallback: 'normal',
    describe: 'Ability distribution',
    choices: ['normal', 'uniform', 'grid'],
  },
  target: { kind: 'number', fallback: 0.3, describe: 'Standard-error target at which to stop' },
  min: { kind: 'integer', fallback: 5, describe: 'Minimum test length' },
  max: { kind: 'integer', fallback: 30, describe: 'Maximum test length' },
  bank: { kind: 'integer', fallback: 300, describe: 'Synthetic bank size' },
  bins: { kind: 'integer', fallback: 6, describe: 'Bins in the conditional report' },
  exposure: { kind: 'number', fallback: 0.2, describe: 'Exposure rate counted as over-exposed' },
  seed: { kind: 'integer', fallback: 20260101, describe: 'Seed for the whole study' },
} as const satisfies OptionSpecs;

function population(name: string): AbilityDistribution {
  if (name === 'uniform') return uniformPopulation(-3, 3);
  if (name === 'grid') return evenGridPopulation(-3, 3, 13);
  return normalPopulation();
}

/**
 * Compare the four policies that bracket the interesting trade-off.
 *
 * Unrestricted maximum information is the precision ceiling and the exposure
 * floor; the two controlled variants give some of that precision back for a
 * bank that survives contact with more than one cohort; the fixed-length form is
 * the non-adaptive baseline every one of them has to beat to justify its
 * existence.
 */
function policies(target: number, min: number, max: number): Policy[] {
  const stopping = precisionTarget(target, { minimum: min, maximum: max });
  const plan = blueprint({ arrays: 0.4, graphs: 0.3, 'dynamic-programming': 0.3 });
  return [
    { name: 'max-information', selector: maximumInformationSelector(), stopping },
    { name: 'kullback-leibler', selector: kullbackLeiblerSelector(), stopping },
    { name: 'randomesque-5', selector: randomesque(maximumInformationSelector(), 5), stopping },
    {
      name: 'balanced+randomesque',
      selector: contentBalanced(randomesque(maximumInformationSelector(), 5), plan),
      stopping,
    },
    {
      name: `fixed-${max}`,
      selector: maximumInformationSelector(),
      stopping: fixedLength(max),
    },
  ];
}

/** Run the policy comparison and print the summary and conditional tables. */
export function runStudyCommand(argv: readonly string[]): void {
  const options = parseOptions(argv, STUDY_OPTIONS);
  const pool = syntheticPool({ size: options.bank, seed: options.seed });
  const distribution = population(options.population);
  const abilities = drawPopulation(distribution, options.examinees, options.seed + 1);

  const results = compareStudies(policies(options.target, options.min, options.max), {
    pool,
    abilities,
    seed: options.seed,
    population: distribution.name,
  });
  const summaries = results.map((result) =>
    summarise(result, { bins: options.bins, exposureTarget: options.exposure }),
  );

  process.stdout.write(
    `Bank: ${pool.size} items across ${pool.domains().length} domains\n` +
      `Population: ${distribution.name}, ${options.examinees} examinees per policy\n` +
      `Stopping: standard error <= ${options.target}, ` +
      `between ${options.min} and ${options.max} items\n\n`,
  );
  process.stdout.write(comparisonTable(summaries));
  process.stdout.write(
    '\nitems  mean test length      calib  reported standard error / realised error\n' +
      'cover  fraction of examinees inside the reported 95% interval\n' +
      'max_x  highest exposure rate reached by any single item\n' +
      'overlap  expected proportion of items shared by two examinees\n',
  );

  for (const summary of summaries) {
    process.stdout.write(`\n${summary.policy} — conditional on true ability\n`);
    process.stdout.write(conditionalTable(summary));
    process.stdout.write(
      `  ${summary.exposure.itemsUsed} of ${summary.poolSize} items used, ` +
        `${summary.exposure.aboveTarget} above an exposure rate of ${summary.exposure.target}, ` +
        `overlap ${summary.exposure.overlap.toFixed(3)}\n`,
    );
  }
}
