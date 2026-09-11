import { createRng, type Rng } from '../core/random.js';
import { makeItem, threePL, type Item } from '../models/item.js';
import { makePolytomousItem, type AnyItem } from '../models/mixed.js';
import {
  generalizedPartialCredit,
  graded,
  partialCredit,
  type PolytomousModel,
} from '../models/polytomous.js';
import { ItemPool } from '../session/pool.js';

export interface SyntheticBankOptions {
  /** Number of items. Default 300. */
  readonly size?: number;
  /** Domains to spread items across. Default three interview-prep topics. */
  readonly domains?: readonly string[];
  /** Mean and spread of the lognormal discrimination distribution. */
  readonly discrimination?: { readonly logMean?: number; readonly logSd?: number };
  /** Standard deviation of the normal difficulty distribution. Default 1.2. */
  readonly difficultySd?: number;
  /** Lower asymptote, i.e. one over the number of options. Default 0.25. */
  readonly guessing?: number;
  /** Seed. Default 20260101. */
  readonly seed?: number;
}

/**
 * Generate a synthetic item bank with realistic parameter distributions.
 *
 * Discriminations are lognormal and difficulties normal, which is what
 * calibrated banks actually look like: a long right tail of a few very
 * discriminating items, and difficulties concentrated near the population mean
 * with thin tails. A bank of uniformly spaced, identically discriminating items
 * — the obvious thing to generate — would make every selection policy look
 * equally good, because there would be nothing to choose between.
 *
 * The guessing parameter defaults to 0.25, the chance level for a
 * four-option multiple-choice item.
 */
export function syntheticBank(options: SyntheticBankOptions = {}): Item[] {
  const size = options.size ?? 300;
  if (!Number.isInteger(size) || size < 1) {
    throw new RangeError(`syntheticBank: size must be a positive integer, received ${size}`);
  }
  const domains = options.domains ?? ['arrays', 'graphs', 'dynamic-programming'];
  if (domains.length === 0) throw new RangeError('syntheticBank: at least one domain is required');
  const logMean = options.discrimination?.logMean ?? Math.log(1.1);
  const logSd = options.discrimination?.logSd ?? 0.35;
  const difficultySd = options.difficultySd ?? 1.2;
  const guessing = options.guessing ?? 0.25;
  const rng: Rng = createRng(options.seed ?? 20260101);

  const items: Item[] = [];
  for (let index = 0; index < size; index += 1) {
    const a = Math.min(Math.exp(logMean + logSd * rng.nextNormal()), 4);
    const b = difficultySd * rng.nextNormal();
    const domain = domains[index % domains.length] as string;
    items.push(makeItem(`${domain}-${String(index).padStart(4, '0')}`, threePL(a, b, guessing), { domain }));
  }
  return items;
}

/** The same bank, wrapped in a pool. */
export function syntheticPool(options: SyntheticBankOptions = {}): ItemPool {
  return new ItemPool(syntheticBank(options));
}

export interface MixedBankOptions extends SyntheticBankOptions {
  /**
   * Fraction of the bank that is polytomous, in [0, 1]. Default 0.2 — roughly
   * the constructed-response share of a typical mixed form.
   */
  readonly polytomousFraction?: number;
  /** Categories a polytomous item scores into, drawn uniformly. Default 3 to 5. */
  readonly categories?: { readonly min?: number; readonly max?: number };
  /**
   * Which polytomous family to generate. Default `graded`.
   *
   * `partial-credit` fixes discrimination at 1 by definition, so a bank
   * generated in that family ignores the discrimination distribution.
   */
  readonly polytomousModel?: PolytomousModel;
  /**
   * Spacing between adjacent thresholds, in the item's own metric — that is,
   * `a` times the gap in logits. Default 2.0.
   *
   * Not a raw logit gap, because the raw gap does not decide anything on its
   * own: a middle category of a graded item is modal somewhere exactly when
   * `a * gap` exceeds `2 ln 2`, whatever `a` is. Setting this below about 1.39
   * generates rubrics whose middle levels are never the most likely outcome for
   * anybody — which is a legitimate thing to want to generate, and the reason
   * this is an option rather than a constant.
   */
  readonly thresholdSpacing?: number;
  /**
   * Jitter applied to each threshold, in the same metric as the spacing.
   * Default 0.35.
   *
   * Scaled for the same reason. A fixed jitter in raw logits is small beside
   * the gap a weakly discriminating item needs and large beside the gap a sharp
   * one needs, so it would flatten the variation on one end of the bank and
   * swamp it on the other.
   */
  readonly thresholdJitter?: number;
  /**
   * Ceiling on the total spread from the lowest threshold to the highest, in
   * logits. Default 5.
   *
   * The spacing above is expressed in the item's metric, so a weakly
   * discriminating item needs a wide raw gap to separate its levels — and a
   * five-category item at a discrimination of 0.6 would want its thresholds
   * more than ten logits apart, which puts its end categories outside any
   * ability a test will ever measure. That is not a bank anybody calibrated.
   *
   * Where this cap binds, the spacing shrinks and some of the item's levels
   * stop being modal. That is the honest outcome rather than a failure of the
   * generator: an item that discriminates weakly genuinely cannot separate five
   * ordered levels across a four-logit range, and a bank that pretended
   * otherwise would make every rubric look better than rubrics are.
   */
  readonly thresholdSpan?: number;
}

/**
 * Generate a bank mixing dichotomous and polytomous items.
 *
 * The polytomous items are placed on the same difficulty distribution as the
 * dichotomous ones, with their thresholds spread around that location. That
 * spread is the parameter that matters: thresholds bunched together make an
 * item behave almost dichotomously, since the middle categories are never
 * modal, while thresholds spread too far leave dead bands where the item
 * informs about nothing.
 *
 * The spacing is measured in the item's own metric — `a` times the gap — and
 * not in raw logits, because that is the quantity which decides the outcome.
 * A middle category of a graded item is modal somewhere exactly when
 * `a * (b_k - b_{k-1})` exceeds `2 ln 2`, a threshold that does not depend on
 * `a` at all, so the raw gap an item needs runs from about two logits at a
 * discrimination of 0.7 down to half a logit at 3. Spreading every item at one
 * flat logit — which this did until the modal-category report caught it —
 * leaves a collapsed level in every item below a discrimination of 1.39, which
 * on a lognormal centred at 1.1 is most of the bank.
 *
 * The default of 2.0 sits about 1.44 times the critical value: comfortably
 * clear for a graded item, close enough to it that a rubric with a genuinely
 * weak level is still within reach of the jitter, which is scaled the same way
 * for the same reason.
 *
 * Graded thresholds are sorted before construction. They are drawn around a
 * location and would otherwise arrive unordered, which the graded model rejects
 * — correctly, since unordered cumulative boundaries are not a bank you could
 * calibrate. The partial credit families keep the draw order, because a
 * reversed step there is a legitimate item and generating one occasionally is
 * the point.
 */
export function syntheticMixedBank(options: MixedBankOptions = {}): AnyItem[] {
  const size = options.size ?? 300;
  const fraction = options.polytomousFraction ?? 0.2;
  if (!(fraction >= 0 && fraction <= 1)) {
    throw new RangeError(
      `syntheticMixedBank: polytomousFraction must lie in [0, 1], received ${fraction}`,
    );
  }
  const minCategories = options.categories?.min ?? 3;
  const maxCategories = options.categories?.max ?? 5;
  if (!Number.isInteger(minCategories) || minCategories < 2) {
    throw new RangeError(
      `syntheticMixedBank: minimum categories must be an integer of at least 2, ` +
        `received ${minCategories}`,
    );
  }
  if (!Number.isInteger(maxCategories) || maxCategories < minCategories) {
    throw new RangeError(
      `syntheticMixedBank: maximum categories must be an integer of at least the ` +
        `minimum (${minCategories}), received ${maxCategories}`,
    );
  }

  const spacing = options.thresholdSpacing ?? 2;
  if (!(spacing > 0) || !Number.isFinite(spacing)) {
    throw new RangeError(
      `syntheticMixedBank: thresholdSpacing must be a positive, finite number, received ${spacing}`,
    );
  }
  const jitter = options.thresholdJitter ?? 0.35;
  if (!(jitter >= 0) || !Number.isFinite(jitter)) {
    throw new RangeError(
      `syntheticMixedBank: thresholdJitter must be a non-negative, finite number, ` +
        `received ${jitter}`,
    );
  }
  const span = options.thresholdSpan ?? 5;
  if (!(span > 0) || !Number.isFinite(span)) {
    throw new RangeError(
      `syntheticMixedBank: thresholdSpan must be a positive, finite number, received ${span}`,
    );
  }

  const model = options.polytomousModel ?? 'graded';
  const domains = options.domains ?? ['arrays', 'graphs', 'dynamic-programming'];
  const logMean = options.discrimination?.logMean ?? Math.log(1.1);
  const logSd = options.discrimination?.logSd ?? 0.35;
  const difficultySd = options.difficultySd ?? 1.2;
  const guessing = options.guessing ?? 0.25;
  const rng: Rng = createRng(options.seed ?? 20260101);

  const polytomousCount = Math.round(size * fraction);
  const items: AnyItem[] = [];
  let polytomousPlaced = 0;
  for (let index = 0; index < size; index += 1) {
    const domain = domains[index % domains.length] as string;
    const a = Math.min(Math.exp(logMean + logSd * rng.nextNormal()), 4);
    const location = difficultySd * rng.nextNormal();

    // Deal the polytomous items evenly across the bank rather than clustering
    // them at one end, so that a content-balanced or exposure-controlled policy
    // meets both formats in every domain. The comparison is against the
    // proportion of the bank passed so far, which places exactly
    // `polytomousCount` of them however the two numbers divide.
    const isPolytomousSlot =
      polytomousPlaced < polytomousCount &&
      (index + 1) * polytomousCount >= (polytomousPlaced + 1) * size;

    if (!isPolytomousSlot) {
      items.push(
        makeItem(`${domain}-${String(index).padStart(4, '0')}`, threePL(a, location, guessing), {
          domain,
        }),
      );
      continue;
    }

    const categories =
      minCategories + Math.floor(rng.next() * (maxCategories - minCategories + 1));
    const steps = categories - 1;
    // The partial credit model fixes discrimination at one by definition, so
    // its own metric is the raw logit scale and the spacing converts to itself.
    const discrimination = model === 'partial-credit' ? 1 : a;
    // The gap the item's metric asks for, narrowed if that would spread the
    // thresholds further than a test could ever reach.
    const wanted = spacing / discrimination;
    const gap = steps > 1 ? Math.min(wanted, span / (steps - 1)) : wanted;
    // The jitter narrows with the gap, so a capped item does not have its
    // ordering scrambled by a jitter sized for the gap it did not get.
    const spread = (jitter / discrimination) * (gap / wanted);
    const thresholds: number[] = [];
    for (let k = 0; k < steps; k += 1) {
      // Spread around the location, in the metric that decides whether the
      // middle categories are ever modal.
      thresholds.push(location + (k - (steps - 1) / 2) * gap + spread * rng.nextNormal());
    }
    if (model === 'graded') thresholds.sort((left, right) => left - right);

    const parameters =
      model === 'graded'
        ? graded(a, thresholds)
        : model === 'partial-credit'
          ? partialCredit(thresholds)
          : generalizedPartialCredit(a, thresholds);

    items.push(
      makePolytomousItem(`${domain}-cr-${String(index).padStart(4, '0')}`, parameters, { domain }),
    );
    polytomousPlaced += 1;
  }
  return items;
}

/** The same mixed bank, wrapped in a pool. */
export function syntheticMixedPool(options: MixedBankOptions = {}): ItemPool {
  return new ItemPool(syntheticMixedBank(options));
}
