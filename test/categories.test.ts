import { describe, expect, it } from 'vitest';

import {
  categoryProbabilitiesOf,
  expectedScoreOf,
  graded,
  linspace,
  makeItem,
  makePolytomousItem,
  maximumTestScore,
  partialCredit,
  probabilityCorrect,
  testCharacteristicCurve,
  twoPL,
} from '../src/index.js';
import { modalCategories, rampColour } from '../web/app/views/categories.js';

const GRID = linspace(-4, 4, 201);

function curvesOf(item: Parameters<typeof categoryProbabilitiesOf>[0]): number[][] {
  return GRID.map((theta) => categoryProbabilitiesOf(item, theta));
}

describe('rampColour', () => {
  it('walks the ramp end to end whatever the category count', () => {
    // The darkest step always means the top category. If assignment started at
    // the low end instead, a three-category item's top score would wear the
    // colour a six-category item uses for its middle, and a reader comparing
    // two items in one session would read that as a difference in the items.
    for (const count of [2, 3, 4, 5, 6, 7]) {
      expect(rampColour(0, count)).toBe('var(--ordinal-1)');
      expect(rampColour(count - 1, count)).toBe('var(--ordinal-6)');
    }
  });

  it('never assigns a lower category a later step than a higher one', () => {
    const step = (colour: string): number => Number(colour.replace(/\D/g, ''));
    for (const count of [2, 3, 4, 5, 6, 7, 9]) {
      for (let k = 1; k < count; k += 1) {
        expect(step(rampColour(k, count))).toBeGreaterThanOrEqual(
          step(rampColour(k - 1, count)),
        );
      }
    }
  });

  it('stays inside the ramp for a degenerate count', () => {
    expect(rampColour(0, 1)).toBe('var(--ordinal-6)');
    expect(rampColour(0, 0)).toBe('var(--ordinal-6)');
  });
});

describe('modalCategories', () => {
  it('finds every category modal on a well-spread graded item', () => {
    // Thresholds a full logit apart: each level owns a band of the scale, which
    // is what a rubric is supposed to do.
    const item = makePolytomousItem('spread', graded(1.2, [-1.5, -0.5, 0.5, 1.5]));
    expect(modalCategories(curvesOf(item))).toEqual(new Set([0, 1, 2, 3, 4]));
  });

  it('finds a middle category modal nowhere when two thresholds collapse', () => {
    // Thresholds two and three almost on top of each other. Category 2 is the
    // band between them, and that band has closed: there is no ability at which
    // scoring exactly 2 is the most likely outcome, so the level is doing no
    // discriminating work and should be merged with a neighbour.
    const item = makePolytomousItem('collapsed', graded(1.4, [-1.2, 0.4, 0.45, 1.6]));
    const modal = modalCategories(curvesOf(item));
    expect(modal.has(2)).toBe(false);
    expect(modal.has(1)).toBe(true);
    expect(modal.has(3)).toBe(true);
  });

  it('reports both categories of a dichotomous item as modal', () => {
    const item = makeItem('mc', twoPL(1.3, 0.2));
    expect(modalCategories(curvesOf(item))).toEqual(new Set([0, 1]));
  });

  it('breaks a tie towards the lower category', () => {
    // A level only ever *tied* for most likely is not separating a band of
    // ability from the one below it, so it does not count as modal.
    expect(modalCategories([[0.5, 0.5]])).toEqual(new Set([0]));
    expect(modalCategories([[0.25, 0.25, 0.25, 0.25]])).toEqual(new Set([0]));
  });

  it('handles a partial credit item with a reversed step', () => {
    // A reversed step is legitimate in the partial credit family — and it is
    // exactly the case that produces a category modal nowhere.
    const item = makePolytomousItem('reversed', partialCredit([0.8, -0.6, 1.4]));
    const modal = modalCategories(curvesOf(item));
    expect(modal.size).toBeLessThan(4);
    expect(modal.has(0)).toBe(true);
    expect(modal.has(3)).toBe(true);
  });
});

describe('the expected score the category panel draws', () => {
  it('is the item characteristic curve for a dichotomous item', () => {
    // The panel plots expectedScoreOf / maximumScoreOf on the probability axis.
    // On a dichotomous item the maximum is 1, so that fraction is exactly the
    // probability of a correct answer — which is what makes it honest to put it
    // on an axis labelled in percent.
    const item = makeItem('mc', twoPL(1.15, -0.3));
    for (const theta of GRID) {
      expect(expectedScoreOf(item, theta) / 1).toBeCloseTo(
        probabilityCorrect(item.parameters, theta),
        12,
      );
    }
  });

  it('stays a proportion on a rubric item, at both ends of the scale', () => {
    const item = makePolytomousItem('rubric', graded(1.6, [-1, 0, 1]));
    for (const theta of GRID) {
      const share = expectedScoreOf(item, theta) / 3;
      expect(share).toBeGreaterThanOrEqual(0);
      expect(share).toBeLessThanOrEqual(1);
    }
    expect(expectedScoreOf(item, -20) / 3).toBeLessThan(0.01);
    expect(expectedScoreOf(item, 20) / 3).toBeGreaterThan(0.99);
  });
});

describe('the test characteristic curve the score panel draws', () => {
  const form = [
    makeItem('mc-1', twoPL(1.2, -0.5)),
    makeItem('mc-2', twoPL(0.9, 0.3)),
    makePolytomousItem('cr-1', graded(1.4, [-0.8, 0.2, 1.1])),
    makePolytomousItem('cr-2', graded(1.1, [0, 1])),
  ];

  it('runs from zero to the form maximum over the drawn domain', () => {
    // The panel's y axis is [0, maximumTestScore] and is never fitted to the
    // curve. These two assertions are what make that axis the right one: the
    // curve really does use the whole of it.
    const maximum = maximumTestScore(form);
    expect(maximum).toBe(1 + 1 + 3 + 2);
    expect(testCharacteristicCurve(form, -20)).toBeLessThan(0.05 * maximum);
    expect(testCharacteristicCurve(form, 20)).toBeGreaterThan(0.95 * maximum);
  });

  it('is strictly increasing across every sampled point', () => {
    // Monotonicity is what lets the panel treat the curve as a lookup from an
    // ability to a score and back. A non-monotone stretch would mean two
    // abilities share an expected score, and the reading off the chart would be
    // ambiguous exactly where the reader most wants it.
    let previous = testCharacteristicCurve(form, GRID[0] as number);
    for (const theta of GRID.slice(1)) {
      const value = testCharacteristicCurve(form, theta);
      expect(value).toBeGreaterThan(previous);
      previous = value;
    }
  });

  it('is the sum of the item expected scores, which is how the residual is read', () => {
    // The panel subtracts the observed score from this curve at the estimate
    // and calls the difference a residual. That is only meaningful because the
    // curve is the sum of the per-item expectations the responses were drawn
    // from.
    for (const theta of [-2, -0.5, 0, 0.75, 2]) {
      const summed = form.reduce((total, item) => total + expectedScoreOf(item, theta), 0);
      expect(testCharacteristicCurve(form, theta)).toBeCloseTo(summed, 12);
    }
  });

  it('is empty-form safe at zero, which is the panel’s first render', () => {
    expect(maximumTestScore([])).toBe(0);
    expect(testCharacteristicCurve([], 0)).toBe(0);
  });
});
