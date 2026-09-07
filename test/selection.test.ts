import { describe, expect, it } from 'vitest';
import { linspace } from '../src/core/numeric.js';
import { simpson } from '../src/core/quadrature.js';
import { createRng, type Rng } from '../src/core/random.js';
import { makeItem, threePL, twoPL, type Item } from '../src/models/item.js';
import { itemInformation, type ScoredResponse } from '../src/models/response.js';
import {
  blueprint,
  contentBalanced,
  contentProportions,
} from '../src/selection/content.js';
import {
  exposureRates,
  randomesque,
  rankByInformationAt,
  sympsonHetter,
  unusedFraction,
} from '../src/selection/exposure.js';
import {
  kullbackLeiblerDivergence,
  kullbackLeiblerSelector,
  maximumInformationSelector,
} from '../src/selection/information.js';
import { rankByScore, type SelectionContext, type Selector } from '../src/selection/selector.js';
import { simulateResponse } from '../src/simulation/respondent.js';

function bank(count: number, a = 1.2): Item[] {
  return linspace(-3, 3, count).map((b, index) => makeItem(`i-${index}`, twoPL(a, b)));
}

function context(overrides: Partial<SelectionContext> & { candidates: readonly Item[] }): SelectionContext {
  return {
    theta: 0,
    responses: [],
    rng: createRng(1),
    ...overrides,
  };
}

/** Administer `length` items to a respondent of known ability, returning what was used. */
function runSession(
  selector: Selector,
  pool: readonly Item[],
  trueTheta: number,
  length: number,
  rng: Rng,
  theta = 0,
): Item[] {
  const remaining = [...pool];
  const responses: ScoredResponse[] = [];
  const administered: Item[] = [];
  for (let i = 0; i < length; i += 1) {
    const pick = selector.select({ candidates: remaining, theta, responses, rng });
    if (pick === null) break;
    administered.push(pick);
    responses.push({ item: pick, response: simulateResponse(pick, trueTheta, rng) });
    remaining.splice(
      remaining.findIndex((item) => item.id === pick.id),
      1,
    );
  }
  return administered;
}

describe('simpson', () => {
  it('integrates a cubic exactly', () => {
    // Simpson is exact through degree three.
    expect(simpson((x) => x * x * x - 2 * x + 1, 0, 2, 2)).toBeCloseTo(4 - 4 + 2, 12);
  });

  it('converges on a transcendental integral', () => {
    expect(simpson(Math.sin, 0, Math.PI, 512)).toBeCloseTo(2, 10);
    expect(simpson(Math.exp, 0, 1, 512)).toBeCloseTo(Math.E - 1, 12);
  });

  it('has fourth-order error: halving the step cuts the error about sixteenfold', () => {
    const coarse = Math.abs(simpson(Math.sin, 0, Math.PI, 16) - 2);
    const fine = Math.abs(simpson(Math.sin, 0, Math.PI, 32) - 2);
    expect(coarse / fine).toBeGreaterThan(12);
    expect(coarse / fine).toBeLessThan(20);
  });

  it('returns zero for a degenerate interval', () => {
    expect(simpson(Math.exp, 1, 1, 8)).toBe(0);
  });

  it('rejects an odd or too-small subinterval count', () => {
    expect(() => simpson(Math.exp, 0, 1, 3)).toThrow(RangeError);
    expect(() => simpson(Math.exp, 0, 1, 0)).toThrow(RangeError);
  });
});

describe('rankByScore', () => {
  it('orders best first and breaks ties by id', () => {
    const items = [makeItem('b', twoPL(1, 0)), makeItem('a', twoPL(1, 0)), makeItem('c', twoPL(2, 0))];
    const ranked = rankByScore(items, (item) => item.parameters.a);
    expect(ranked.map((entry) => entry.item.id)).toEqual(['c', 'a', 'b']);
  });
});

describe('maximumInformationSelector', () => {
  const selector = maximumInformationSelector();

  it('picks the item whose difficulty is closest to the estimate', () => {
    // With equal discriminations, 2PL information peaks at theta = b.
    const pool = bank(13);
    for (const theta of [-2, -0.5, 0, 1, 2.5]) {
      const pick = selector.select(context({ candidates: pool, theta }));
      expect(pick).not.toBeNull();
      const distances = pool.map((item) => Math.abs(item.parameters.b - theta));
      const best = Math.min(...distances);
      expect(Math.abs((pick as Item).parameters.b - theta)).toBeCloseTo(best, 12);
    }
  });

  it('prefers the more discriminating of two equally targeted items', () => {
    const pool = [makeItem('sharp', twoPL(2, 0)), makeItem('flat', twoPL(0.6, 0))];
    expect(selector.select(context({ candidates: pool }))?.id).toBe('sharp');
  });

  it('returns the item with maximum information, checked directly', () => {
    const pool = [
      makeItem('a', twoPL(1.9, 1.4)),
      makeItem('b', threePL(1.2, 0.1, 0.25)),
      makeItem('c', twoPL(0.8, -0.2)),
    ];
    const pick = selector.select(context({ candidates: pool, theta: 0.3 })) as Item;
    const best = Math.max(...pool.map((item) => itemInformation(item.parameters, 0.3)));
    expect(itemInformation(pick.parameters, 0.3)).toBeCloseTo(best, 14);
  });

  it('returns null on an empty candidate set', () => {
    expect(selector.select(context({ candidates: [] }))).toBeNull();
  });

  it('is deterministic', () => {
    const pool = bank(20);
    const first = selector.select(context({ candidates: pool, theta: 0.4 }));
    const second = selector.select(context({ candidates: pool, theta: 0.4 }));
    expect(first?.id).toBe(second?.id);
  });
});

describe('kullbackLeiblerDivergence', () => {
  it('is zero when the two abilities coincide', () => {
    expect(kullbackLeiblerDivergence(makeItem('a', twoPL(1.3, 0.2)), 0.5, 0.5)).toBeCloseTo(0, 14);
  });

  it('is strictly positive otherwise', () => {
    const item = makeItem('a', twoPL(1.3, 0.2));
    for (const theta of [-2, -0.5, 1, 2]) {
      expect(kullbackLeiblerDivergence(item, 0.5, theta)).toBeGreaterThan(0);
    }
  });

  it('grows as the two abilities separate', () => {
    const item = makeItem('a', twoPL(1.3, 0));
    const near = kullbackLeiblerDivergence(item, 0, 0.5);
    const far = kullbackLeiblerDivergence(item, 0, 2);
    expect(far).toBeGreaterThan(near);
  });

  it('is locally approximated by the Fisher information', () => {
    // KL(theta_hat || theta_hat + e) is approximately I(theta_hat) e^2 / 2.
    const item = makeItem('a', twoPL(1.4, 0.3));
    const epsilon = 1e-3;
    const kl = kullbackLeiblerDivergence(item, 0.3, 0.3 + epsilon);
    expect(kl).toBeCloseTo((itemInformation(item.parameters, 0.3) * epsilon * epsilon) / 2, 9);
  });
});

describe('kullbackLeiblerSelector', () => {
  it('agrees with maximum information once the window is narrow', () => {
    const pool = bank(25);
    const responses: ScoredResponse[] = Array.from({ length: 60 }, (_, i) => ({
      item: pool[i % pool.length] as Item,
      response: (i % 2) as 0 | 1,
    }));
    const kl = kullbackLeiblerSelector().select(context({ candidates: pool, theta: 0.7, responses }));
    const fisher = maximumInformationSelector().select(context({ candidates: pool, theta: 0.7 }));
    expect(kl?.id).toBe(fisher?.id);
  });

  it('disagrees with maximum information when the window is wide', () => {
    // At theta = 0 the flat, on-target item carries marginally more Fisher
    // information (0.2500) than the sharp off-target one (0.2330) — but across
    // a whole logit of plausible ability the sharp item separates far better.
    // Fisher information cannot see that, because it is a property of a point.
    const pool = [makeItem('flat-on-target', twoPL(1, 0)), makeItem('sharp-off-target', twoPL(3, 1.2))];
    expect(maximumInformationSelector().select(context({ candidates: pool, theta: 0 }))?.id).toBe(
      'flat-on-target',
    );
    const wide = kullbackLeiblerSelector({ delta: 1, shrinkWithStandardError: false });
    expect(wide.select(context({ candidates: pool, theta: 0 }))?.id).toBe('sharp-off-target');
  });

  it('agrees with maximum information as the window closes', () => {
    // The KL index is locally quadratic with curvature given by the Fisher
    // information, so a narrow window reproduces the Fisher ranking.
    const pool = [makeItem('flat-on-target', twoPL(1, 0)), makeItem('sharp-off-target', twoPL(3, 1.2))];
    const narrow = kullbackLeiblerSelector({ delta: 0.02, shrinkWithStandardError: false });
    expect(narrow.select(context({ candidates: pool, theta: 0 }))?.id).toBe('flat-on-target');
  });

  it('returns null on an empty candidate set', () => {
    expect(kullbackLeiblerSelector().select(context({ candidates: [] }))).toBeNull();
  });

  it('rejects a non-positive window', () => {
    expect(() => kullbackLeiblerSelector({ delta: 0 })).toThrow(RangeError);
  });
});

describe('randomesque', () => {
  const pool = bank(30);

  it('never selects outside the top k', () => {
    const selector = randomesque(maximumInformationSelector(), 4);
    const allowed = new Set(rankByInformationAt(pool, 0.5).slice(0, 4).map((item) => item.id));
    for (let seed = 0; seed < 300; seed += 1) {
      const pick = selector.select(context({ candidates: pool, theta: 0.5, rng: createRng(seed) }));
      expect(allowed.has((pick as Item).id)).toBe(true);
    }
  });

  it('spreads selection across the whole top k', () => {
    const selector = randomesque(maximumInformationSelector(), 5);
    const seen = new Set<string>();
    for (let seed = 0; seed < 300; seed += 1) {
      const pick = selector.select(context({ candidates: pool, theta: 0, rng: createRng(seed) }));
      seen.add((pick as Item).id);
    }
    expect(seen.size).toBe(5);
  });

  it('reduces to the base selector at k = 1', () => {
    const selector = randomesque(maximumInformationSelector(), 1);
    const base = maximumInformationSelector();
    for (const theta of [-1, 0, 1]) {
      expect(selector.select(context({ candidates: pool, theta }))?.id).toBe(
        base.select(context({ candidates: pool, theta }))?.id,
      );
    }
  });

  it('leaves less of the bank untouched than the unconstrained rule', () => {
    const plain = maximumInformationSelector();
    const spread = randomesque(plain, 6);
    const plainSessions: Item[][] = [];
    const spreadSessions: Item[][] = [];
    for (let candidate = 0; candidate < 120; candidate += 1) {
      const trueTheta = createRng(candidate + 1).nextNormal();
      plainSessions.push(runSession(plain, pool, trueTheta, 8, createRng(candidate + 500)));
      spreadSessions.push(runSession(spread, pool, trueTheta, 8, createRng(candidate + 500)));
    }
    expect(unusedFraction(pool, spreadSessions)).toBeLessThan(unusedFraction(pool, plainSessions));

    const plainPeak = Math.max(...exposureRates(plainSessions).values());
    const spreadPeak = Math.max(...exposureRates(spreadSessions).values());
    expect(spreadPeak).toBeLessThan(plainPeak);
  });

  it('is deterministic under a fixed seed', () => {
    const selector = randomesque(maximumInformationSelector(), 5);
    const first = selector.select(context({ candidates: pool, rng: createRng(77) }));
    const second = selector.select(context({ candidates: pool, rng: createRng(77) }));
    expect(first?.id).toBe(second?.id);
  });

  it('rejects a non-positive k', () => {
    expect(() => randomesque(maximumInformationSelector(), 0)).toThrow(RangeError);
  });
});

describe('sympsonHetter', () => {
  const pool = bank(20);

  it('reduces the exposure rate of a constrained item', () => {
    const plain = maximumInformationSelector();
    const hottest = rankByInformationAt(pool, 0).slice(0, 3).map((item) => item.id);
    const controlled = sympsonHetter(plain, {
      parameters: new Map(hottest.map((id) => [id, 0.2])),
    });

    const plainSessions: Item[][] = [];
    const controlledSessions: Item[][] = [];
    for (let candidate = 0; candidate < 200; candidate += 1) {
      const trueTheta = createRng(candidate + 1).nextNormal();
      plainSessions.push(runSession(plain, pool, trueTheta, 6, createRng(candidate + 900)));
      controlledSessions.push(runSession(controlled, pool, trueTheta, 6, createRng(candidate + 900)));
    }

    const before = exposureRates(plainSessions);
    const after = exposureRates(controlledSessions);
    for (const id of hottest) {
      expect(after.get(id) ?? 0).toBeLessThan(before.get(id) ?? 0);
    }
  });

  it('administers an unconstrained item unchanged', () => {
    const selector = sympsonHetter(maximumInformationSelector(), { parameters: new Map() });
    const base = maximumInformationSelector();
    expect(selector.select(context({ candidates: pool, theta: 0.6 }))?.id).toBe(
      base.select(context({ candidates: pool, theta: 0.6 }))?.id,
    );
  });

  it('administers something rather than nothing when every candidate is rejected', () => {
    const parameters = new Map(pool.map((item) => [item.id, 0]));
    const selector = sympsonHetter(maximumInformationSelector(), { parameters, maxAttempts: 3 });
    const pick = selector.select(context({ candidates: pool, theta: 0 }));
    expect(pick).not.toBeNull();
  });

  it('returns null on an empty candidate set', () => {
    const selector = sympsonHetter(maximumInformationSelector(), { parameters: new Map() });
    expect(selector.select(context({ candidates: [] }))).toBeNull();
  });

  it('rejects out-of-range parameters and attempt counts', () => {
    expect(() =>
      sympsonHetter(maximumInformationSelector(), { parameters: new Map([['a', 1.4]]) }),
    ).toThrow(/\[0, 1\]/);
    expect(() =>
      sympsonHetter(maximumInformationSelector(), { parameters: new Map(), maxAttempts: 0 }),
    ).toThrow(RangeError);
  });
});

describe('blueprint', () => {
  it('normalises weights', () => {
    const plan = blueprint({ arrays: 2, graphs: 1, dp: 1 });
    expect(plan.get('arrays')).toBeCloseTo(0.5, 12);
    expect(plan.get('graphs')).toBeCloseTo(0.25, 12);
  });

  it('rejects degenerate input', () => {
    expect(() => blueprint({})).toThrow(RangeError);
    expect(() => blueprint({ a: 0, b: 0 })).toThrow(RangeError);
    expect(() => blueprint({ a: -1 })).toThrow(RangeError);
  });
});

describe('contentBalanced', () => {
  const domains = ['arrays', 'graphs', 'dp'];
  const pool: Item[] = domains.flatMap((domain, d) =>
    linspace(-2.5, 2.5, 20).map((b, index) =>
      makeItem(`${domain}-${index}`, twoPL(1 + 0.1 * d, b), { domain }),
    ),
  );

  it('delivers the blueprint proportions', () => {
    const plan = blueprint({ arrays: 0.5, graphs: 0.25, dp: 0.25 });
    const selector = contentBalanced(maximumInformationSelector(), plan);
    const administered = runSession(selector, pool, 0.5, 24, createRng(4));
    const proportions = contentProportions(administered);
    expect(proportions.get('arrays')).toBeCloseTo(0.5, 1);
    expect(proportions.get('graphs')).toBeCloseTo(0.25, 1);
    expect(proportions.get('dp')).toBeCloseTo(0.25, 1);
  });

  it('picks the most informative item within the chosen domain', () => {
    const plan = blueprint({ arrays: 1 });
    const selector = contentBalanced(maximumInformationSelector(), plan);
    const pick = selector.select(context({ candidates: pool, theta: 1.2 })) as Item;
    expect(pick.domain).toBe('arrays');
    const withinDomain = pool.filter((item) => item.domain === 'arrays');
    const best = Math.max(...withinDomain.map((item) => itemInformation(item.parameters, 1.2)));
    expect(itemInformation(pick.parameters, 1.2)).toBeCloseTo(best, 14);
  });

  it('honours a lopsided blueprint', () => {
    const plan = blueprint({ arrays: 0.9, graphs: 0.1 });
    const selector = contentBalanced(maximumInformationSelector(), plan);
    const administered = runSession(selector, pool, 0, 20, createRng(6));
    const proportions = contentProportions(administered);
    expect(proportions.get('arrays') ?? 0).toBeGreaterThan(0.8);
    expect(proportions.get('dp')).toBeUndefined();
  });

  it('falls back to the whole candidate set once blueprint domains are exhausted', () => {
    const plan = blueprint({ missing: 1 });
    const selector = contentBalanced(maximumInformationSelector(), plan);
    const pick = selector.select(context({ candidates: pool, theta: 0 }));
    expect(pick).not.toBeNull();
  });

  it('returns null on an empty candidate set', () => {
    const selector = contentBalanced(maximumInformationSelector(), blueprint({ arrays: 1 }));
    expect(selector.select(context({ candidates: [] }))).toBeNull();
  });

  it('composes with exposure control', () => {
    const plan = blueprint({ arrays: 0.5, graphs: 0.5 });
    const selector = contentBalanced(randomesque(maximumInformationSelector(), 4), plan);
    const administered = runSession(selector, pool, 0.2, 16, createRng(9));
    const proportions = contentProportions(administered);
    expect(proportions.get('arrays')).toBeCloseTo(0.5, 1);
    expect(proportions.get('graphs')).toBeCloseTo(0.5, 1);
    expect(selector.name).toContain('randomesque');
  });

  it('rejects an empty blueprint', () => {
    expect(() => contentBalanced(maximumInformationSelector(), new Map())).toThrow(RangeError);
  });
});

describe('exposure accounting', () => {
  const pool = bank(6);

  it('counts a repeated item once per session', () => {
    const rates = exposureRates([[pool[0] as Item, pool[0] as Item], [pool[1] as Item]]);
    expect(rates.get('i-0')).toBeCloseTo(0.5, 12);
    expect(rates.get('i-1')).toBeCloseTo(0.5, 12);
  });

  it('reports empty results for no sessions', () => {
    expect(exposureRates([]).size).toBe(0);
    expect(unusedFraction([], [])).toBe(0);
  });

  it('reports the fraction of the bank never administered', () => {
    expect(unusedFraction(pool, [[pool[0] as Item, pool[1] as Item]])).toBeCloseTo(4 / 6, 12);
    expect(unusedFraction(pool, [pool])).toBe(0);
  });
});
