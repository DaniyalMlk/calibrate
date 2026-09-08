import { syntheticPool } from './src/simulation/bank.js';
import { normalPopulation, drawPopulation, evenGridPopulation } from './src/simulation/population.js';
import { runStudy, compareStudies, type Policy } from './src/simulation/study.js';
import { summarise, comparisonTable, conditionalTable } from './src/simulation/report.js';
import { maximumInformationSelector } from './src/selection/information.js';
import { randomesque } from './src/selection/exposure.js';
import { contentBalanced, blueprint } from './src/selection/content.js';
import { precisionTarget, fixedLength } from './src/session/stopping.js';
import { recoveryStudy, recoveryTable } from './src/simulation/recovery.js';
import { eapEstimator, mleEstimator, wleEstimator, mapEstimator } from './src/session/estimator.js';
import { syntheticBank } from './src/simulation/bank.js';

const pool = syntheticPool({ size: 300, seed: 20260101 });
const abilities = drawPopulation(normalPopulation(), 400, 5);

const policies: Policy[] = [
  { name: 'max-info', selector: maximumInformationSelector(), stopping: precisionTarget(0.3, { minimum: 5, maximum: 30 }) },
  { name: 'randomesque-5', selector: randomesque(maximumInformationSelector(), 5), stopping: precisionTarget(0.3, { minimum: 5, maximum: 30 }) },
  { name: 'balanced-r5', selector: contentBalanced(randomesque(maximumInformationSelector(), 5), blueprint({ arrays: 0.4, graphs: 0.3, 'dynamic-programming': 0.3 })), stopping: precisionTarget(0.3, { minimum: 5, maximum: 30 }) },
  { name: 'fixed-20', selector: maximumInformationSelector(), stopping: fixedLength(20) },
];
const t0 = Date.now();
const results = compareStudies(policies, { pool, abilities, population: 'normal(0,1)' });
const summaries = results.map((r) => summarise(r));
console.log(comparisonTable(summaries));
console.log('elapsed ms', Date.now() - t0);
console.log(conditionalTable(summaries[1]!));

const form = syntheticBank({ size: 20, seed: 777 });
for (const [name, est] of [['mle', mleEstimator()], ['eap', eapEstimator()], ['wle', wleEstimator()], ['map', mapEstimator()]] as const) {
  const s = recoveryStudy({ name, estimator: est as any, items: form, replications: 200, abilities: [-2.5,-1.5,-0.5,0.5,1.5,2.5] });
  console.log(name, 'meanAbsBias', s.meanAbsoluteBias.toFixed(4), 'pooledRmse', s.pooledRmse.toFixed(4));
  console.log(recoveryTable(s));
}
