import { describe, expect, it } from 'vitest';

import {
  createRng,
  exposureConcentration,
  exposureRatesFromIds,
  formatCounts,
  isPolytomous,
  ItemPool,
  maximumInformationSelector,
  precisionTarget,
  randomesque,
  simulateSession,
  syntheticMixedBank,
  syntheticPool,
  type AnyItem,
} from '../src/index.js';

function ratesOf(counts: readonly number[]): ReadonlyMap<string, number> {
  return new Map(counts.map((count, i) => [`i-${i}`, count]));
}

describe('exposureConcentration', () => {
  it('reports a Gini of exactly zero when every item is used equally', () => {
    // The closed form, and the reason the Gini is computed from the sorted
    // shares rather than by integrating the curve: a trapezoid integration
    // leaves a small positive residue where the exact definition gives zero.
    for (const size of [2, 5, 40, 301]) {
      const rates = ratesOf(Array.from({ length: size }, () => 0.25));
      expect(exposureConcentration(size, rates).gini).toBe(0);
    }
  });

  it('puts an even distribution exactly on the diagonal', () => {
    const size = 8;
    const result = exposureConcentration(size, ratesOf(Array.from({ length: size }, () => 0.5)));

    for (const point of result.curve) {
      expect(point.exposureShare).toBeCloseTo(point.bankFraction, 12);
    }
  });

  it('reaches (n - 1) / n when one item absorbs everything', () => {
    // The maximum a bank of finite size can reach — not one, which is the
    // continuous limit and is unreachable with a countable number of items.
    for (const size of [2, 4, 10, 100]) {
      const counts = Array.from({ length: size }, (_, i) => (i === 0 ? 1 : 0));
      const result = exposureConcentration(size, ratesOf(counts));

      expect(result.gini).toBeCloseTo((size - 1) / size, 12);
      expect(result.topShare).toBeCloseTo(1, 12);
      expect(result.unused).toBe(size - 1);
    }
  });

  it('is invariant to scaling every rate by the same factor', () => {
    // Concentration is about shares, so doubling every exposure count — twice
    // as many candidates, the same policy — must not move the coefficient.
    const counts = [0, 1, 1, 2, 3, 5, 8, 13];
    const base = exposureConcentration(counts.length, ratesOf(counts)).gini;

    for (const factor of [0.5, 2, 1000]) {
      const scaled = exposureConcentration(
        counts.length,
        ratesOf(counts.map((count) => count * factor)),
      ).gini;
      expect(scaled).toBeCloseTo(base, 12);
    }
  });

  it('matches a hand-computed Gini on a small distribution', () => {
    // Shares 1, 2, 3, 4 over a bank of four. With x ascending,
    //   G = 2 * (1*1 + 2*2 + 3*3 + 4*4) / (4 * 10) - 5/4
    //     = 2 * 30 / 40 - 1.25 = 1.5 - 1.25 = 0.25
    const result = exposureConcentration(4, ratesOf([1, 2, 3, 4]));

    expect(result.gini).toBeCloseTo(0.25, 12);
  });

  it('counts items absent from the map as never administered', () => {
    // An unused item is the most concentrating thing a bank can hold, so a
    // curve computed only over the items that appear would omit the evidence
    // that matters most.
    const partial = new Map([['i-0', 1]]);
    const result = exposureConcentration(10, partial);

    expect(result.unused).toBe(9);
    expect(result.gini).toBeCloseTo(0.9, 12);
  });

  it('treats an untouched bank as even rather than as maximally concentrated', () => {
    const result = exposureConcentration(50, new Map());

    expect(result.gini).toBe(0);
    expect(result.topShare).toBe(0);
    expect(result.topDecileShare).toBe(0);
    expect(result.unused).toBe(50);
  });

  it('runs the curve from the origin to the far corner, never decreasing', () => {
    const counts = [0, 0, 1, 1, 2, 4, 4, 9, 20, 31];
    const result = exposureConcentration(counts.length, ratesOf(counts));

    expect(result.curve[0]).toEqual({ bankFraction: 0, exposureShare: 0 });
    expect(result.curve.at(-1)?.bankFraction).toBeCloseTo(1, 12);
    expect(result.curve.at(-1)?.exposureShare).toBeCloseTo(1, 12);

    for (let i = 1; i < result.curve.length; i += 1) {
      const previous = result.curve[i - 1] as LorenzLike;
      const point = result.curve[i] as LorenzLike;
      expect(point.bankFraction).toBeGreaterThan(previous.bankFraction);
      expect(point.exposureShare).toBeGreaterThanOrEqual(previous.exposureShare);
      // A Lorenz curve is convex and never rises above the diagonal.
      expect(point.exposureShare).toBeLessThanOrEqual(point.bankFraction + 1e-12);
    }
  });

  it('reports the share taken by the most exposed tenth', () => {
    // Ten items, one of which took half of everything: the top decile is one
    // item and its share is a half.
    const result = exposureConcentration(10, ratesOf([1, 1, 1, 1, 1, 1, 1, 1, 1, 9]));

    expect(result.topDecileShare).toBeCloseTo(0.5, 12);
    expect(result.topShare).toBeCloseTo(0.5, 12);
  });

  it('orders two distributions with the same peak rate by their shape', () => {
    // The case the coefficient exists for. Both banks have the same peak rate
    // and the same unused count, and one spreads the remainder far more evenly.
    const concentrated = ratesOf([5, 5, 5, 0, 0, 0, 0, 0, 0, 0]);
    const spread = ratesOf([5, 4, 4, 3, 3, 3, 2, 2, 2, 1]);

    const a = exposureConcentration(10, concentrated);
    const b = exposureConcentration(10, spread);

    expect(a.topShare).toBeGreaterThan(b.topShare - 1e-9);
    expect(a.gini).toBeGreaterThan(b.gini);
  });

  it('rejects a bank size that cannot hold the rates it was given', () => {
    expect(() => exposureConcentration(2, ratesOf([1, 1, 1]))).toThrow(/exposure rates for a bank/);
    expect(() => exposureConcentration(0, new Map())).toThrow(/positive integer/);
    expect(() => exposureConcentration(2.5, new Map())).toThrow(/positive integer/);
  });

  it('rejects a negative or non-finite rate', () => {
    expect(() => exposureConcentration(3, ratesOf([1, -1, 0]))).toThrow(/non-negative/);
    expect(() => exposureConcentration(3, ratesOf([1, Number.NaN, 0]))).toThrow(/finite/);
  });
});

interface LorenzLike {
  readonly bankFraction: number;
  readonly exposureShare: number;
}

describe('exposureConcentration over simulated sessions', () => {
  it('finds unconstrained maximum information more concentrated than randomesque', () => {
    // The claim exposure control is bought for, measured on the shape of the
    // distribution rather than on its peak.
    const pool = syntheticPool({ size: 200, seed: 7 });
    const rng = createRng(11);
    const abilities = Array.from({ length: 120 }, () => rng.nextNormal());

    const administer = (policy: Parameters<typeof simulateSession>[0]): number => {
      const sessions = abilities.map(
        (theta, i) => simulateSession(policy, pool, theta, 5000 + i).itemIds,
      );
      return exposureConcentration(pool.size, exposureRatesFromIds(sessions)).gini;
    };

    const stopping = precisionTarget(0.35, { minimum: 5, maximum: 25 });
    const greedy = administer({
      name: 'max-information',
      selector: maximumInformationSelector(),
      stopping,
    });
    const randomised = administer({
      name: 'randomesque-5',
      selector: randomesque(maximumInformationSelector(), 5),
      stopping,
    });

    expect(greedy).toBeGreaterThan(randomised);
  });

  it('collapses onto the rubric items when a mixed bank is selected greedily', () => {
    // A rubric item carries several times the Fisher information of a single
    // dichotomous item at the same location, so an unconstrained maximum
    // information policy meeting a mixed bank does not balance the two formats
    // — it takes the rubrics until they run out.
    //
    // This is the cost of the mix, and it is not visible from the precision
    // side, where the mix looks like nothing but good news: the test gets
    // shorter because each item says more. The same fact read from the bank
    // side is that a fifth of the items absorb most of the administrations.
    const abilities = Array.from({ length: 150 }, (_, i) => -3 + (i / 149) * 6);
    const stopping = precisionTarget(0.3, { minimum: 5, maximum: 30 });

    const measure = (polytomousFraction: number) => {
      const bank = syntheticMixedBank({ size: 200, polytomousFraction, seed: 42 });
      const pool = new ItemPool(bank);
      const sessions = abilities.map(
        (theta, i) =>
          simulateSession(
            { name: 'max-information', selector: maximumInformationSelector(), stopping },
            pool,
            theta,
            4000 + i,
          ).itemIds,
      );

      const administered = sessions.flat();
      const rubric = administered.filter(
        (id) => isPolytomous(pool.byId(id) as AnyItem),
      ).length;
      return {
        gini: exposureConcentration(pool.size, exposureRatesFromIds(sessions)).gini,
        length: administered.length / sessions.length,
        rubricShareOfAdministrations: rubric / administered.length,
        rubricShareOfBank: formatCounts(bank).polytomous / bank.length,
      };
    };

    const dichotomous = measure(0);
    const mixed = measure(0.2);

    expect(dichotomous.rubricShareOfAdministrations).toBe(0);
    // A fifth of the bank taking well over half of what is administered.
    expect(mixed.rubricShareOfBank).toBeCloseTo(0.2, 6);
    expect(mixed.rubricShareOfAdministrations).toBeGreaterThan(0.5);
    // Concentrated onto fewer items, and over a shorter test.
    expect(mixed.gini).toBeGreaterThan(dichotomous.gini);
    expect(mixed.length).toBeLessThan(dichotomous.length);
  });

  it('is relieved by exposure control on a mixed bank, at the cost of length', () => {
    // The remedy is the one the dichotomous case already had. Randomesque
    // selection over the top five spreads administration across the bank; the
    // price is a longer test, since the item chosen is no longer the most
    // informative one available.
    const bank = syntheticMixedBank({ size: 200, polytomousFraction: 0.2, seed: 42 });
    const pool = new ItemPool(bank);
    const abilities = Array.from({ length: 150 }, (_, i) => -3 + (i / 149) * 6);
    const stopping = precisionTarget(0.3, { minimum: 5, maximum: 30 });

    const administer = (selector: Parameters<typeof simulateSession>[0]['selector']) => {
      const sessions = abilities.map(
        (theta, i) =>
          simulateSession({ name: 'p', selector, stopping }, pool, theta, 4000 + i).itemIds,
      );
      return {
        gini: exposureConcentration(pool.size, exposureRatesFromIds(sessions)).gini,
        length: sessions.reduce((total, ids) => total + ids.length, 0) / sessions.length,
      };
    };

    const greedy = administer(maximumInformationSelector());
    const spread = administer(randomesque(maximumInformationSelector(), 5));

    expect(spread.gini).toBeLessThan(greedy.gini);
    expect(spread.length).toBeGreaterThan(greedy.length);
  });
});
