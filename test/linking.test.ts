import { describe, expect, it } from 'vitest';
import { linspace, mean } from '../src/core/numeric.js';
import { normalGridRule } from '../src/core/quadrature.js';
import { createRng } from '../src/core/random.js';
import { calibrate, toItems } from '../src/calibration/jmle.js';
import { makeItem, threePL, twoPL } from '../src/models/item.js';
import {
  expectedScoreOf,
  informationOf,
  makePolytomousItem,
  testCharacteristicCurve,
  type AnyItem,
} from '../src/models/mixed.js';
import { generalizedPartialCredit, graded, partialCredit } from '../src/models/polytomous.js';
import { probabilityCorrect } from '../src/models/response.js';
import {
  commonItems,
  discriminationParameters,
  locationParameters,
  requireCommonItems,
} from '../src/linking/common.js';
import {
  haebara,
  haebaraCriterion,
  link,
  linkAll,
  meanMean,
  meanSigma,
  stockingLord,
  stockingLordCriterion,
} from '../src/linking/methods.js';
import {
  composeTransforms,
  describeTransform,
  IDENTITY_TRANSFORM,
  invertTransform,
  scaleTransform,
  transformAbility,
  transformBank,
  transformItem,
  transformItemParameters,
  transformPolytomousParameters,
} from '../src/linking/scale.js';
import { simulateMatrix } from '../src/simulation/respondent.js';
import { syntheticBank, syntheticMixedBank } from '../src/simulation/bank.js';

const TRUTH = scaleTransform(1.35, -0.42);
const GRID = linspace(-3, 3, 25);

describe('the scale transformation', () => {
  it('rejects a non-positive or non-finite slope', () => {
    expect(() => scaleTransform(0, 0)).toThrow(/slope \(A\) must be positive/);
    expect(() => scaleTransform(-1, 0)).toThrow(/reverse or collapse/);
    expect(() => scaleTransform(Number.NaN, 0)).toThrow(/slope \(A\)/);
    expect(() => scaleTransform(1, Number.NaN)).toThrow(/intercept \(B\)/);
  });

  it('moves an ability linearly', () => {
    expect(transformAbility(TRUTH, 1)).toBeCloseTo(1.35 - 0.42, 14);
    expect(transformAbility(IDENTITY_TRANSFORM, 0.7)).toBe(0.7);
    expect(() => transformAbility(TRUTH, Number.NaN)).toThrow(/ability \(theta\)/);
  });

  it('inverts exactly', () => {
    const back = invertTransform(TRUTH);
    for (const theta of GRID) {
      expect(transformAbility(back, transformAbility(TRUTH, theta))).toBeCloseTo(theta, 12);
    }
    expect(composeTransforms(TRUTH, invertTransform(TRUTH)).slope).toBeCloseTo(1, 14);
    expect(composeTransforms(TRUTH, invertTransform(TRUTH)).intercept).toBeCloseTo(0, 14);
  });

  it('composes in application order', () => {
    const first = scaleTransform(1.2, 0.3);
    const second = scaleTransform(0.8, -0.5);
    const composed = composeTransforms(first, second);
    for (const theta of GRID) {
      expect(transformAbility(composed, theta)).toBeCloseTo(
        transformAbility(second, transformAbility(first, theta)),
        12,
      );
    }
  });

  it('leaves every response probability unchanged — the point of the whole exercise', () => {
    // A transformed item evaluated at a transformed ability must give exactly
    // the probability the original gave at the original ability. That identity
    // is what makes the metric arbitrary in the first place.
    const parameters = threePL(1.4, 0.6, 0.2);
    const moved = transformItemParameters(TRUTH, parameters);
    for (const theta of GRID) {
      expect(probabilityCorrect(moved, transformAbility(TRUTH, theta))).toBeCloseTo(
        probabilityCorrect(parameters, theta),
        12,
      );
    }
  });

  it('holds the same identity for polytomous items', () => {
    const parameters = graded(1.3, [-1.1, 0.2, 1.4]);
    const moved = transformPolytomousParameters(TRUTH, parameters);
    for (const theta of GRID) {
      expect(expectedScoreOf(makePolytomousItem('m', moved), transformAbility(TRUTH, theta)))
        .toBeCloseTo(expectedScoreOf(makePolytomousItem('o', parameters), theta), 12);
    }
  });

  it('carries the asymptotes across untouched', () => {
    // c and d are probabilities, not locations: the chance of guessing a
    // four-option item does not depend on where the origin was put.
    const moved = transformItemParameters(TRUTH, threePL(1.4, 0.6, 0.25));
    expect(moved.c).toBe(0.25);
    expect(moved.d).toBe(1);
  });

  it('preserves item information under the matching ability shift', () => {
    // Information is a density in theta, so it rescales by 1 / A^2 as the unit
    // of the scale changes.
    const item = makeItem('i', twoPL(1.4, 0.3));
    const moved = transformItem(TRUTH, item);
    for (const theta of GRID) {
      expect(informationOf(moved, transformAbility(TRUTH, theta))).toBeCloseTo(
        informationOf(item, theta) / (TRUTH.slope * TRUTH.slope),
        12,
      );
    }
  });

  it('round-trips a whole mixed bank to floating-point tolerance', () => {
    const bank = syntheticMixedBank({ size: 40, polytomousFraction: 0.25, seed: 5 });
    const back = transformBank(invertTransform(TRUTH), transformBank(TRUTH, bank));
    for (let index = 0; index < bank.length; index += 1) {
      const original = bank[index] as AnyItem;
      const returned = back[index] as AnyItem;
      expect(returned.id).toBe(original.id);
      expect(returned.parameters.a).toBeCloseTo(original.parameters.a, 12);
      const originalLocations = locationParameters([original]);
      const returnedLocations = locationParameters([returned]);
      expect(returnedLocations).toHaveLength(originalLocations.length);
      for (let k = 0; k < originalLocations.length; k += 1) {
        expect(returnedLocations[k]).toBeCloseTo(originalLocations[k] as number, 12);
      }
    }
  });

  it('keeps identity and blueprint metadata', () => {
    const item = makeItem('keeps', twoPL(1, 0), { domain: 'algebra', tags: ['anchor'] });
    const moved = transformItem(TRUTH, item);
    expect(moved.id).toBe('keeps');
    expect(moved.domain).toBe('algebra');
    expect(moved.tags).toEqual(['anchor']);
  });

  it('refuses to rescale a partial credit item', () => {
    const item = makePolytomousItem('pcm', partialCredit([-1, 0.5]));
    expect(() => transformItem(TRUTH, item)).toThrow(/fixes discrimination at 1|generalized/);
    // A pure shift leaves the family intact and is allowed.
    expect(() => transformItem(scaleTransform(1, 0.8), item)).not.toThrow();
  });

  it('formats for reporting', () => {
    expect(describeTransform(scaleTransform(1.0432, 0.2171))).toBe('theta* = 1.0432 theta + 0.2171');
    expect(describeTransform(scaleTransform(1, -0.5))).toBe('theta* = 1.0000 theta - 0.5000');
  });
});

describe('matching common items', () => {
  const a = makeItem('shared-1', twoPL(1.2, 0));
  const b = makeItem('shared-2', twoPL(1.0, 1));
  const onlySource = makeItem('source-only', twoPL(1.1, -1));
  const onlyTarget = makeItem('target-only', twoPL(0.9, 0.5));

  it('matches by identity and keeps source order', () => {
    const pairs = commonItems([a, onlySource, b], [b, onlyTarget, a]);
    expect(pairs.map((pair) => pair.id)).toEqual(['shared-1', 'shared-2']);
    expect(pairs[0]?.source).toBe(a);
    expect(pairs[0]?.target).toBe(a);
  });

  it('excludes items present in only one calibration', () => {
    expect(commonItems([onlySource], [onlyTarget])).toEqual([]);
  });

  it('rejects a duplicate id in the source bank', () => {
    expect(() => commonItems([a, a], [a])).toThrow(/duplicate item id/);
  });

  it('rejects an id calibrated as two different things', () => {
    const asPolytomous = makePolytomousItem('shared-1', graded(1.2, [-1, 1]));
    expect(() => commonItems([a], [asPolytomous])).toThrow(/dichotomous in one calibration/);

    const threeCategory = makePolytomousItem('p', graded(1, [-1, 1]));
    const fourCategory = makePolytomousItem('p', graded(1, [-1, 0, 1]));
    expect(() => commonItems([threeCategory], [fourCategory])).toThrow(/thresholds/);

    const asGraded = makePolytomousItem('q', graded(1, [-1, 1]));
    const asGpcm = makePolytomousItem('q', generalizedPartialCredit(1, [-1, 1]));
    expect(() => commonItems([asGraded], [asGpcm])).toThrow(/in one\s+calibration|graded/);
  });

  it('enforces a minimum anchor length', () => {
    expect(() => requireCommonItems([a], [a], 2)).toThrow(/at least 2 common items, found 1/);
    expect(requireCommonItems([a, b], [a, b], 2)).toHaveLength(2);
  });

  it('takes every threshold of a polytomous item as a location', () => {
    const item = makePolytomousItem('p', graded(1.2, [-1, 0.5, 1.5]));
    expect(locationParameters([item])).toEqual([-1, 0.5, 1.5]);
    expect(locationParameters([a])).toEqual([0]);
    expect(discriminationParameters([item, a])).toEqual([1.2, 1.2]);
  });
});

describe('exact recovery of a known transformation', () => {
  // The check that matters: when the target bank is an exact transformation of
  // the source, every method has a known right answer.
  const source = syntheticMixedBank({ size: 40, polytomousFraction: 0.25, seed: 5 });
  const target = transformBank(TRUTH, source);
  const pairs = commonItems(source, target);

  it('pairs the whole bank', () => {
    expect(pairs).toHaveLength(40);
  });

  it('recovers the coefficients under all four methods', () => {
    for (const result of linkAll(pairs)) {
      expect(result.transform.slope).toBeCloseTo(TRUTH.slope, 8);
      expect(result.transform.intercept).toBeCloseTo(TRUTH.intercept, 8);
      expect(result.converged).toBe(true);
      expect(result.commonItems).toBe(40);
    }
  });

  it('drives both criteria to zero at the true coefficients', () => {
    const rule = normalGridRule(0, 1, 41);
    expect(haebaraCriterion(pairs, TRUTH, rule)).toBeLessThan(1e-24);
    expect(stockingLordCriterion(pairs, TRUTH, rule)).toBeLessThan(1e-24);
  });

  it('recovers from a purely dichotomous anchor', () => {
    const binary = syntheticBank({ size: 30, seed: 9 });
    const moved = transformBank(TRUTH, binary);
    for (const result of linkAll(commonItems(binary, moved))) {
      expect(result.transform.slope).toBeCloseTo(TRUTH.slope, 8);
      expect(result.transform.intercept).toBeCloseTo(TRUTH.intercept, 8);
    }
  });

  it('recovers from a purely polytomous anchor', () => {
    const rubrics: AnyItem[] = [
      makePolytomousItem('e1', graded(1.2, [-1, 0, 1])),
      makePolytomousItem('e2', graded(1.6, [-0.5, 0.8, 1.9])),
      makePolytomousItem('e3', graded(0.9, [-2, -0.4, 1.2])),
    ];
    const moved = transformBank(TRUTH, rubrics);
    for (const result of linkAll(commonItems(rubrics, moved))) {
      expect(result.transform.slope).toBeCloseTo(TRUTH.slope, 8);
      expect(result.transform.intercept).toBeCloseTo(TRUTH.intercept, 8);
    }
  });

  it('agrees with the closed forms for the moment methods', () => {
    const sourceItems = pairs.map((pair) => pair.source);
    const targetItems = pairs.map((pair) => pair.target);

    const meanMeanSlope =
      mean(discriminationParameters(sourceItems)) / mean(discriminationParameters(targetItems));
    expect(meanMean(pairs).transform.slope).toBeCloseTo(meanMeanSlope, 14);

    const sourceLocations = locationParameters(sourceItems);
    const targetLocations = locationParameters(targetItems);
    const spread = (values: number[]): number => {
      const centre = mean(values);
      return Math.sqrt(mean(values.map((value) => (value - centre) ** 2)));
    };
    expect(meanSigma(pairs).transform.slope).toBeCloseTo(
      spread(targetLocations) / spread(sourceLocations),
      12,
    );
    expect(meanSigma(pairs).transform.intercept).toBeCloseTo(
      mean(targetLocations) - meanSigma(pairs).transform.slope * mean(sourceLocations),
      12,
    );
  });

  it('makes the linked bank reproduce the target characteristic curve', () => {
    const linked = transformBank(link(pairs, 'stocking-lord').transform, source);
    for (const theta of GRID) {
      expect(testCharacteristicCurve(linked, theta)).toBeCloseTo(
        testCharacteristicCurve(target, theta),
        8,
      );
    }
  });
});

describe('robustness of the criterion searches', () => {
  const source = syntheticBank({ size: 30, seed: 9 });
  const target = transformBank(TRUTH, source);
  const pairs = commonItems(source, target);

  it('recovers from starting points inside the spurious large-slope basin', () => {
    // A single-start search from either of the last two returned A near 11.5
    // against a true 1.35, converged and confident. Multi-start is what fixes it.
    for (const start of [
      scaleTransform(0.2, -5),
      scaleTransform(1, 0),
      scaleTransform(5, 6),
      scaleTransform(12, 4.5),
    ]) {
      for (const method of [haebara, stockingLord]) {
        const result = method(pairs, { start });
        expect(result.transform.slope).toBeCloseTo(TRUTH.slope, 6);
        expect(result.transform.intercept).toBeCloseTo(TRUTH.intercept, 6);
      }
    }
  });

  it('confirms the basin is a property of the objective, not the optimiser', () => {
    // Holding the intercept badly wrong, the criterion falls monotonically as
    // the slope grows past the truth — flattened curves beat misplaced ones.
    const rule = normalGridRule(0, 1, 41);
    const wrongIntercept = 4.53;
    const atTruth = haebaraCriterion(pairs, scaleTransform(TRUTH.slope, wrongIntercept), rule);
    const atDouble = haebaraCriterion(pairs, scaleTransform(2, wrongIntercept), rule);
    const atEight = haebaraCriterion(pairs, scaleTransform(8, wrongIntercept), rule);
    expect(atDouble).toBeLessThan(atTruth);
    expect(atEight).toBeLessThan(atDouble);
  });

  it('reaches the minimum of its own criterion', () => {
    const rule = normalGridRule(0, 1, 41);
    const solution = haebara(pairs).transform;
    const best = haebaraCriterion(pairs, solution, rule);
    for (const dSlope of [-0.05, 0.05]) {
      for (const dIntercept of [-0.05, 0.05]) {
        const nearby = scaleTransform(solution.slope + dSlope, solution.intercept + dIntercept);
        expect(haebaraCriterion(pairs, nearby, rule)).toBeGreaterThanOrEqual(best - 1e-15);
      }
    }
  });

  it('has each method beat the other on its own criterion', () => {
    const rule = normalGridRule(0, 1, 41);
    const h = haebara(pairs).transform;
    const s = stockingLord(pairs).transform;
    expect(haebaraCriterion(pairs, h, rule)).toBeLessThanOrEqual(
      haebaraCriterion(pairs, s, rule) + 1e-12,
    );
    expect(stockingLordCriterion(pairs, s, rule)).toBeLessThanOrEqual(
      stockingLordCriterion(pairs, h, rule) + 1e-12,
    );
  });
});

describe('linking under estimation noise', () => {
  const truth = scaleTransform(1.28, -0.35);
  const source = syntheticBank({ size: 30, seed: 9 });
  const exact = transformBank(truth, source);

  // The same model with perturbed parameters — the realistic case. Perturbing
  // the model itself instead (a 3PL anchor matched against 2PL estimates) is a
  // misspecification rather than noise, and it moves the criterion methods far
  // more than the moment methods, for reasons that have nothing to do with
  // linking.
  const rng = createRng(4);
  const noisy = exact.map((item) => {
    const parameters = item.parameters as { a: number; b: number; c: number };
    return makeItem(
      item.id,
      threePL(
        Math.max(0.2, parameters.a * (1 + 0.1 * rng.nextNormal())),
        parameters.b + 0.12 * rng.nextNormal(),
        parameters.c,
      ),
    );
  });
  const pairs = commonItems(source, noisy);

  it('keeps every method near the generating coefficients', () => {
    // The tolerance is set from the noise, not from a round number. Difficulties
    // are perturbed with a standard deviation of 0.12 across 30 anchor items, so
    // the intercept alone carries a standard error near 0.12 / sqrt(30) = 0.022
    // before the slope's own uncertainty is added. Errors of two to three times
    // that are ordinary sampling, and a test that failed on them would be
    // measuring the seed rather than the method.
    for (const result of linkAll(pairs)) {
      expect(Math.abs(result.transform.slope - truth.slope)).toBeLessThan(0.1);
      expect(Math.abs(result.transform.intercept - truth.intercept)).toBeLessThan(0.1);
    }
  });

  it('keeps the four methods close to one another', () => {
    const slopes = linkAll(pairs).map((result) => result.transform.slope);
    expect(Math.max(...slopes) - Math.min(...slopes)).toBeLessThan(0.1);
  });

  it('leaves a positive criterion, since no transformation fits exactly', () => {
    expect(haebara(pairs).criterion).toBeGreaterThan(0);
    expect(stockingLord(pairs).criterion).toBeGreaterThan(0);
  });

  it('fits the characteristic curve better than leaving the scales alone', () => {
    const rule = normalGridRule(0, 1, 41);
    expect(stockingLordCriterion(pairs, stockingLord(pairs).transform, rule)).toBeLessThan(
      stockingLordCriterion(pairs, IDENTITY_TRANSFORM, rule),
    );
    expect(haebaraCriterion(pairs, haebara(pairs).transform, rule)).toBeLessThan(
      haebaraCriterion(pairs, IDENTITY_TRANSFORM, rule),
    );
  });
});

describe('linking two independent calibrations', () => {
  // The operational case end to end: one anchor, two samples that differ in
  // ability, each calibrated on its own and therefore on its own metric.
  const rng = createRng(11);
  const anchor = Array.from({ length: 30 }, (_, index) =>
    makeItem(`anchor-${index}`, twoPL(0.8 + 0.9 * rng.next(), -2 + 4 * rng.next())),
  );
  const drawSample = (count: number, centre: number, spread: number, seed: number): number[] => {
    const source = createRng(seed);
    return Array.from({ length: count }, () => centre + spread * source.nextNormal());
  };

  const lower = calibrate(simulateMatrix(anchor, drawSample(2000, 0, 1, 101), 1), {
    model: '2pl',
  });
  const higher = calibrate(simulateMatrix(anchor, drawSample(2000, 0.8, 1.3, 202), 2), {
    model: '2pl',
  });

  it('calibrates both samples', () => {
    expect(lower.converged).toBe(true);
    expect(higher.converged).toBe(true);
  });

  it('finds the whole anchor common to both', () => {
    expect(commonItems(toItems(higher), toItems(lower))).toHaveLength(30);
  });

  it('produces coefficients the four methods agree on', () => {
    const pairs = commonItems(toItems(higher), toItems(lower));
    const results = linkAll(pairs);
    const slopes = results.map((result) => result.transform.slope);
    const intercepts = results.map((result) => result.transform.intercept);
    expect(Math.max(...slopes) - Math.min(...slopes)).toBeLessThan(0.1);
    expect(Math.max(...intercepts) - Math.min(...intercepts)).toBeLessThan(0.1);
    for (const result of results) expect(result.converged).toBe(true);
  });

  it('brings the two calibrations of the anchor onto one metric', () => {
    const pairs = commonItems(toItems(higher), toItems(lower));
    const rule = normalGridRule(0, 1, 41);
    const linked = stockingLord(pairs);
    // Linking must leave the two characteristic curves closer than they were.
    expect(stockingLordCriterion(pairs, linked.transform, rule)).toBeLessThan(
      stockingLordCriterion(pairs, IDENTITY_TRANSFORM, rule),
    );
  });
});

describe('linking errors', () => {
  const item = makeItem('only', twoPL(1, 0));

  it('requires at least one common item', () => {
    expect(() => meanMean([])).toThrow(/at least one common item/);
    expect(() => meanSigma([])).toThrow(/at least one common item/);
    expect(() => haebara([])).toThrow(/at least one common item/);
    expect(() => stockingLord([])).toThrow(/at least one common item/);
  });

  it('reports when the locations have no spread for mean/sigma', () => {
    const pairs = commonItems([item], [item]);
    expect(() => meanSigma(pairs)).toThrow(/no spread of locations/);
    // Mean/mean needs no spread and still answers.
    expect(meanMean(pairs).transform.slope).toBeCloseTo(1, 12);
  });

  it('still links a single-item anchor under the criterion methods', () => {
    // With one common item mean/sigma is undefined, so the criterion search
    // falls back to its other starting points rather than failing.
    const pairs = commonItems([item], [transformItem(scaleTransform(1, 0.4), item)]);
    const result = haebara(pairs);
    expect(result.transform.intercept).toBeCloseTo(0.4, 4);
  });

  it('refuses a partial credit anchor', () => {
    const pcm = makePolytomousItem('pcm', partialCredit([-1, 0.5]));
    const pairs = commonItems([pcm], [pcm]);
    expect(() => haebara(pairs)).toThrow(/partial credit/);
    expect(() => meanMean(pairs)).toThrow(/partial credit/);
  });

  it('rejects an unknown method name', () => {
    const pairs = commonItems([item], [item]);
    expect(() => link(pairs, 'nearest-neighbour' as never)).toThrow(/unknown linking method/);
  });
});
