import { simpson } from '../core/quadrature.js';
import type { Item } from '../models/item.js';
import { itemInformation, probabilityCorrect } from '../models/response.js';
import { rankByScore, type SelectionContext, type Selector } from './selector.js';

/**
 * Maximum Fisher information selection.
 *
 * Administers the item whose answer would most reduce the variance of the
 * ability estimate, *assuming the current estimate is correct*. That assumption
 * is the whole weakness of the rule: after two items the estimate is barely
 * better than a guess, and the rule cheerfully spends the bank's most
 * discriminating items chasing it.
 */
export function maximumInformationSelector(): Selector {
  return {
    name: 'maximum-information',
    select(context: SelectionContext): Item | null {
      if (context.candidates.length === 0) return null;
      const ranked = rankByScore(context.candidates, (item) =>
        itemInformation(item.parameters, context.theta),
      );
      return ranked[0]?.item ?? null;
    },
  };
}

/**
 * The Kullback–Leibler information an item carries about distinguishing the
 * current ability estimate from a competing ability.
 *
 * `KL_i(theta_hat || theta) = P_i(theta_hat) ln[P_i(theta_hat)/P_i(theta)]
 *                           + Q_i(theta_hat) ln[Q_i(theta_hat)/Q_i(theta)]`
 *
 * Unlike Fisher information this is a global measure: it asks how well the item
 * separates two specified abilities, not how steep the likelihood is at a point.
 */
export function kullbackLeiblerDivergence(item: Item, thetaHat: number, theta: number): number {
  const pHat = probabilityCorrect(item.parameters, thetaHat);
  const qHat = 1 - pHat;
  const p = probabilityCorrect(item.parameters, theta);
  const q = 1 - p;
  const floor = 1e-300;
  return (
    pHat * Math.log(Math.max(pHat, floor) / Math.max(p, floor)) +
    qHat * Math.log(Math.max(qHat, floor) / Math.max(q, floor))
  );
}

export interface KullbackLeiblerOptions {
  /**
   * Half-width of the integration window around the current estimate.
   *
   * A fixed width is the simple choice; shrinking it as the standard error
   * shrinks is the refinement, and `shrinkWithStandardError` enables it.
   * Default 1.
   */
  readonly delta?: number;
  /**
   * Scale the window by `1 / sqrt(n)` in the number of items answered, so the
   * index narrows towards Fisher information as evidence accumulates. Default
   * true.
   */
  readonly shrinkWithStandardError?: boolean;
  /** Simpson subintervals across the window. Default 32. */
  readonly intervals?: number;
}

/**
 * Kullback–Leibler information selection.
 *
 * Integrates the KL divergence over a window around the current estimate rather
 * than evaluating information at a single point. Early in a test, when the
 * estimate could be off by a logit, an item that separates the whole plausible
 * interval is worth more than one that is razor-sharp at a number nobody
 * believes yet. As responses accumulate the window narrows and the rule
 * converges on maximum information.
 */
export function kullbackLeiblerSelector(options: KullbackLeiblerOptions = {}): Selector {
  const delta = options.delta ?? 1;
  const shrink = options.shrinkWithStandardError ?? true;
  const intervals = options.intervals ?? 32;
  if (delta <= 0) {
    throw new RangeError(`kullbackLeiblerSelector: delta must be positive, received ${delta}`);
  }

  return {
    name: 'kullback-leibler',
    select(context: SelectionContext): Item | null {
      if (context.candidates.length === 0) return null;
      const answered = context.responses.length;
      const width = shrink && answered > 0 ? delta / Math.sqrt(answered) : delta;
      const lo = context.theta - width;
      const hi = context.theta + width;
      const ranked = rankByScore(context.candidates, (item) =>
        simpson((theta) => kullbackLeiblerDivergence(item, context.theta, theta), lo, hi, intervals),
      );
      return ranked[0]?.item ?? null;
    },
  };
}
