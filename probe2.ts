import { syntheticPool } from './src/simulation/bank.js';
import { normalPopulation, drawPopulation } from './src/simulation/population.js';
import { runStudy, type Policy } from './src/simulation/study.js';
import { maximumInformationSelector } from './src/selection/information.js';
import { precisionTarget } from './src/session/stopping.js';
import { eapEstimator, wleEstimator, hybridEstimator } from './src/session/estimator.js';

const pool = syntheticPool({ size: 300, seed: 20260101 });
const abilities = drawPopulation(normalPopulation(), 200, 5);
for (const [name, est] of [['hybrid', hybridEstimator()], ['eap', eapEstimator()], ['wle', wleEstimator()]] as const) {
  const p: Policy = { name, selector: maximumInformationSelector(), stopping: precisionTarget(0.3, { minimum: 5, maximum: 30 }), estimator: est };
  const t = Date.now();
  const r = runStudy(p, { pool, abilities });
  const items = r.outcomes.reduce((a, o) => a + o.length, 0);
  console.log(name, Date.now() - t, 'ms for', items, 'administered items');
}
