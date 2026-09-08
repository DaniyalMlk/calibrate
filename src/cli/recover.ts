import { linspace } from '../core/numeric.js';
import {
  eapEstimator,
  hybridEstimator,
  mapEstimator,
  mleEstimator,
  wleEstimator,
  type SessionEstimator,
} from '../session/estimator.js';
import { syntheticBank } from '../simulation/bank.js';
import { recoveryStudy, recoveryTable } from '../simulation/recovery.js';
import { parseOptions, type OptionSpecs } from './options.js';

export const RECOVER_OPTIONS = {
  estimator: {
    kind: 'string',
    fallback: 'all',
    describe: 'Estimator under test',
    choices: ['all', 'mle', 'eap', 'map', 'wle', 'hybrid'],
  },
  length: { kind: 'integer', fallback: 20, describe: 'Items on the fixed form' },
  points: { kind: 'integer', fallback: 13, describe: 'Ability points evaluated' },
  range: { kind: 'number', fallback: 3, describe: 'Ability range, plus and minus' },
  replications: { kind: 'integer', fallback: 200, describe: 'Replications per ability point' },
  seed: { kind: 'integer', fallback: 20260101, describe: 'Seed for the whole study' },
} as const satisfies OptionSpecs;

const ESTIMATORS: Readonly<Record<string, () => SessionEstimator>> = {
  mle: mleEstimator,
  eap: eapEstimator,
  map: mapEstimator,
  wle: wleEstimator,
  hybrid: hybridEstimator,
};

/** Report how well one or every estimator recovers ability on a fixed form. */
export function runRecover(argv: readonly string[]): void {
  const options = parseOptions(argv, RECOVER_OPTIONS);
  if (options.range <= 0) throw new Error('option "--range" must be positive');

  const items = syntheticBank({ size: options.length, seed: options.seed });
  const abilities = linspace(-options.range, options.range, options.points);
  const names = options.estimator === 'all' ? Object.keys(ESTIMATORS) : [options.estimator];

  process.stdout.write(
    `Form: ${items.length} items, ${options.replications} replications at each of ` +
      `${options.points} ability points from ${-options.range} to ${options.range}\n`,
  );

  for (const name of names) {
    const build = ESTIMATORS[name];
    if (build === undefined) throw new Error(`unknown estimator "${name}"`);
    const study = recoveryStudy({
      name,
      estimator: build(),
      items,
      abilities,
      replications: options.replications,
      seed: options.seed,
    });
    process.stdout.write(
      `\n${name} — mean absolute bias ${study.meanAbsoluteBias.toFixed(4)}, ` +
        `pooled RMSE ${study.pooledRmse.toFixed(4)}\n`,
    );
    process.stdout.write(recoveryTable(study));
  }

  process.stdout.write(
    '\nbound  fraction of patterns with no interior estimate, excluded from the row\n' +
      'calib  reported standard error / realised error; below one is optimistic\n',
  );
}
