import { linspace, requireFinite } from '../core/numeric.js';
import { createRng, normalDeviate, type Rng } from '../core/random.js';

/**
 * A population of examinees, described by the distribution their abilities are
 * drawn from.
 *
 * A simulation study is only as informative as the population it assumes.
 * Sampling everyone from a standard normal flatters any policy that concentrates
 * its bank near the middle of the scale, because almost nobody arrives at the
 * tails to expose the gap there. So the harness carries the distribution as a
 * first-class, named object: the report can say which population produced the
 * numbers, and swapping in a uniform or grid population to probe the edges is a
 * one-line change rather than a rewrite of the study.
 */
export interface AbilityDistribution {
  /** Stable identifier, reproduced in study reports. */
  readonly name: string;
  /** Draw one ability. */
  draw(rng: Rng): number;
}

/** Abilities drawn from a normal distribution — the usual operational assumption. */
export function normalPopulation(mean = 0, sd = 1): AbilityDistribution {
  requireFinite(mean, 'population mean');
  requireFinite(sd, 'population standard deviation');
  if (sd <= 0) {
    throw new RangeError(`normalPopulation: standard deviation must be positive, received ${sd}`);
  }
  return {
    name: `normal(${mean}, ${sd})`,
    draw: (rng) => normalDeviate(rng, mean, sd),
  };
}

/**
 * Abilities drawn uniformly from `[lower, upper]`.
 *
 * The right population for a bank-coverage study: it puts as many examinees in
 * the tails as in the middle, so a region the bank measures badly shows up in
 * the conditional report instead of being averaged away by a thin tail.
 */
export function uniformPopulation(lower: number, upper: number): AbilityDistribution {
  requireFinite(lower, 'population lower bound');
  requireFinite(upper, 'population upper bound');
  if (!(lower < upper)) {
    throw new RangeError(
      `uniformPopulation: lower (${lower}) must be strictly below upper (${upper})`,
    );
  }
  return {
    name: `uniform(${lower}, ${upper})`,
    draw: (rng) => lower + (upper - lower) * rng.next(),
  };
}

/**
 * A fixed set of abilities, cycled in order.
 *
 * Not a distribution in any real sense — it consumes no randomness at all — but
 * it is the population a conditional study wants: an exactly equal number of
 * examinees at each of a set of ability points, so every bin of the conditional
 * report has the same standard error and the bins can be compared directly.
 */
export function gridPopulation(values: readonly number[]): AbilityDistribution {
  if (values.length === 0) throw new RangeError('gridPopulation: at least one value is required');
  for (const value of values) requireFinite(value, 'grid ability');
  const points = Object.freeze([...values]);
  let cursor = 0;
  return {
    name: `grid(${points.length} points from ${points[0]} to ${points[points.length - 1]})`,
    draw: () => {
      const value = points[cursor % points.length] as number;
      cursor += 1;
      return value;
    },
  };
}

/** An evenly spaced grid population covering `[lower, upper]`. */
export function evenGridPopulation(lower: number, upper: number, count: number): AbilityDistribution {
  return gridPopulation(linspace(lower, upper, count));
}

/**
 * Draw `count` abilities from a population.
 *
 * Takes a seed rather than a generator so a caller can name a population and a
 * seed and get the same examinees back on every run, without having to thread a
 * generator through whatever produced the study configuration.
 */
export function drawPopulation(
  distribution: AbilityDistribution,
  count: number,
  seed = 1,
): number[] {
  if (!Number.isInteger(count) || count < 1) {
    throw new RangeError(`drawPopulation: count must be a positive integer, received ${count}`);
  }
  const rng = createRng(seed);
  const abilities: number[] = new Array<number>(count);
  for (let i = 0; i < count; i += 1) abilities[i] = distribution.draw(rng);
  return abilities;
}
