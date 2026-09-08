import { describe, expect, it } from 'vitest';
import {
  eapEstimator,
  mapEstimator,
  mleEstimator,
  wleEstimator,
} from '../src/session/estimator.js';
import { syntheticBank } from '../src/simulation/bank.js';
import { recoveryStudy, recoveryTable, type RecoveryPoint } from '../src/simulation/recovery.js';

/** A twenty-item form, fixed across every study in this file. */
const form = syntheticBank({ size: 20, seed: 777 });
const abilities = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5];
const replications = 150;

function study(name: string, estimator: ReturnType<typeof wleEstimator>) {
  return recoveryStudy({ name, estimator, items: form, abilities, replications, seed: 31 });
}

const mle = study('mle', mleEstimator());
const eap = study('eap', eapEstimator());
const wle = study('wle', wleEstimator());
const map = study('map', mapEstimator());

function at(points: readonly RecoveryPoint[], theta: number): RecoveryPoint {
  const point = points.find((candidate) => candidate.trueTheta === theta);
  if (point === undefined) throw new Error(`no recovery point at ${theta}`);
  return point;
}

describe('recoveryStudy', () => {
  it('evaluates every requested ability point', () => {
    expect(mle.points.map((point) => point.trueTheta)).toEqual(abilities);
    expect(mle.formLength).toBe(20);
    expect(mle.replications).toBe(replications);
  });

  it('replays identically for a given seed', () => {
    expect(study('wle', wleEstimator())).toEqual(wle);
  });

  it('decomposes RMSE into bias and error spread at every point', () => {
    for (const point of wle.points) {
      expect(point.rmse * point.rmse).toBeCloseTo(
        point.bias * point.bias + point.errorSd * point.errorSd,
        9,
      );
    }
  });

  it('accounts for every replication as interior or boundary', () => {
    for (const point of mle.points) {
      expect(point.finite + point.boundaryFraction * replications).toBeCloseTo(replications, 9);
    }
  });

  it('rejects an empty form, an empty ability list and a bad replication count', () => {
    expect(() => recoveryStudy({ name: 'x', estimator: wleEstimator(), items: [] })).toThrow(
      RangeError,
    );
    expect(() =>
      recoveryStudy({ name: 'x', estimator: wleEstimator(), items: form, abilities: [] }),
    ).toThrow(RangeError);
    expect(() =>
      recoveryStudy({ name: 'x', estimator: wleEstimator(), items: form, replications: 0 }),
    ).toThrow(RangeError);
  });
});

describe('known estimator behaviour', () => {
  it('shrinks the Bayesian estimates toward the prior mean', () => {
    // The defining property of a posterior mean under a standard normal prior:
    // pulled up from below zero and down from above it, and increasingly so the
    // further out the generating ability lies.
    expect(at(eap.points, -2.5).bias).toBeGreaterThan(0.5);
    expect(at(eap.points, -1.5).bias).toBeGreaterThan(0.2);
    expect(at(eap.points, 1.5).bias).toBeLessThan(-0.2);
    expect(at(eap.points, 2.5).bias).toBeLessThan(-0.5);
    expect(at(eap.points, -2.5).bias).toBeGreaterThan(at(eap.points, -1.5).bias);
  });

  it('shrinks the posterior mode as well, and by a comparable amount', () => {
    expect(at(map.points, -2.5).bias).toBeGreaterThan(0.5);
    expect(at(map.points, 2.5).bias).toBeLessThan(-0.5);
    expect(map.meanAbsoluteBias).toBeGreaterThan(0.25);
  });

  it('leaves Warm’s estimator far less biased than either Bayesian estimator', () => {
    expect(wle.meanAbsoluteBias).toBeLessThan(eap.meanAbsoluteBias / 2);
    expect(wle.meanAbsoluteBias).toBeLessThan(map.meanAbsoluteBias / 2);
    expect(wle.meanAbsoluteBias).toBeLessThan(0.2);
  });

  it('keeps Warm’s estimator nearly unbiased even at the edges of the range', () => {
    expect(Math.abs(at(wle.points, 2.5).bias)).toBeLessThan(0.35);
    expect(Math.abs(at(wle.points, -2.5).bias)).toBeLessThan(0.35);
  });

  it('leaves maximum likelihood undefined for a substantial share of extreme patterns', () => {
    // At the edges of the range most examinees answer everything correctly or
    // everything incorrectly, and the likelihood then has no interior maximum.
    expect(at(mle.points, 2.5).boundaryFraction).toBeGreaterThan(0.05);
    expect(at(mle.points, -2.5).boundaryFraction).toBeGreaterThan(0.05);
    expect(at(mle.points, 0.5).boundaryFraction).toBeLessThan(0.02);
  });

  it('resolves every pattern under the estimators that carry a prior or a weight', () => {
    for (const points of [eap.points, map.points, wle.points]) {
      for (const point of points) expect(point.boundaryFraction).toBe(0);
    }
  });

  it('buys the Bayesian estimators lower error variance for their bias', () => {
    // The bias/variance trade the prior makes: shrinkage costs accuracy at the
    // extremes but tightens the spread of the estimates everywhere.
    expect(at(eap.points, 0.5).errorSd).toBeLessThan(at(wle.points, 0.5).errorSd);
    expect(at(eap.points, -1.5).errorSd).toBeLessThan(at(wle.points, -1.5).errorSd);
  });

  it('reports honest interval coverage near the centre of the scale', () => {
    expect(at(wle.points, 0.5).coverage).toBeGreaterThan(0.85);
    expect(at(wle.points, -0.5).coverage).toBeGreaterThan(0.85);
    // Shrinkage puts the generating ability outside the reported interval far
    // more often at the edges, which is the cost of the prior stated plainly.
    expect(at(eap.points, -2.5).coverage).toBeLessThan(at(wle.points, -2.5).coverage);
  });

  it('pools an RMSE across points that is dominated by the tails', () => {
    expect(wle.pooledRmse).toBeGreaterThan(at(wle.points, 0.5).rmse);
    expect(wle.pooledRmse).toBeLessThan(at(wle.points, -2.5).rmse);
  });
});

describe('recoveryTable', () => {
  it('renders one row per ability point under a header', () => {
    const lines = recoveryTable(wle).trimEnd().split('\n');
    expect(lines).toHaveLength(abilities.length + 2);
    expect(lines[0]).toContain('theta');
    expect(lines[0]).toContain('bound');
    expect(recoveryTable(wle)).not.toContain('NaN');
  });
});
