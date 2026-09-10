import { requireFinite } from '../core/numeric.js';
import { makeItem, validateItemParameters, type Item, type ItemParameters } from '../models/item.js';
import { isPolytomous, makePolytomousItem, type AnyItem } from '../models/mixed.js';
import { validatePolytomousParameters, type PolytomousParameters } from '../models/polytomous.js';

/**
 * A linear transformation between two ability metrics.
 *
 * Item response theory determines the ability scale only up to a linear
 * transformation: multiply every ability by `A` and add `B`, divide every
 * discrimination by `A` and transform every difficulty the same way as ability,
 * and every response probability in the model is unchanged. Two calibrations of
 * the same bank therefore agree about everything except where they put the
 * origin and how wide they make the unit, and a linking coefficient pair is
 * exactly that disagreement written down.
 *
 * The convention here is that the transformation carries a value *from* the
 * source scale *to* the target scale:
 *
 * ```
 * theta* = A theta + B
 * a*     = a / A
 * b*     = A b + B
 * ```
 */
export interface ScaleTransform {
  /** Slope. Must be positive: a negative slope would reverse the ability order. */
  readonly slope: number;
  /** Intercept. */
  readonly intercept: number;
}

/** Build a validated transformation. */
export function scaleTransform(slope: number, intercept: number): ScaleTransform {
  requireFinite(slope, 'slope (A)');
  requireFinite(intercept, 'intercept (B)');
  if (slope <= 0) {
    throw new RangeError(
      `slope (A) must be positive, received ${slope}; ` +
        'a non-positive slope would reverse or collapse the ability order',
    );
  }
  return Object.freeze({ slope, intercept });
}

/** The identity transformation, which leaves every scale unchanged. */
export const IDENTITY_TRANSFORM: ScaleTransform = scaleTransform(1, 0);

/** Apply a transformation to an ability. */
export function transformAbility(transform: ScaleTransform, theta: number): number {
  requireFinite(theta, 'ability (theta)');
  return transform.slope * theta + transform.intercept;
}

/**
 * The inverse transformation, carrying values back to the source scale.
 *
 * `theta = (theta* - B) / A`, so the inverse is `(1 / A, -B / A)`.
 */
export function invertTransform(transform: ScaleTransform): ScaleTransform {
  return scaleTransform(1 / transform.slope, -transform.intercept / transform.slope);
}

/**
 * Compose two transformations: apply `first`, then `second`.
 *
 * Useful when a bank has been linked through an intermediate calibration, which
 * is the ordinary situation for a bank maintained over several years.
 */
export function composeTransforms(
  first: ScaleTransform,
  second: ScaleTransform,
): ScaleTransform {
  return scaleTransform(
    second.slope * first.slope,
    second.slope * first.intercept + second.intercept,
  );
}

/**
 * Transform a dichotomous parameter set onto the target scale.
 *
 * The asymptotes `c` and `d` are probabilities, not locations on the ability
 * scale, so they are carried across untouched. Transforming them would be a
 * category error: the chance of guessing a four-option item correctly does not
 * depend on where anyone chose to put the origin.
 */
export function transformItemParameters(
  transform: ScaleTransform,
  parameters: ItemParameters,
): ItemParameters {
  return validateItemParameters({
    a: parameters.a / transform.slope,
    b: transform.slope * parameters.b + transform.intercept,
    c: parameters.c,
    d: parameters.d,
    metric: parameters.metric,
  });
}

/**
 * Transform a polytomous parameter set onto the target scale.
 *
 * Every threshold is a location on the ability scale, whether it is a
 * cumulative boundary or an adjacent step, so all of them transform exactly as
 * a difficulty does. The ordering the graded model requires survives, because a
 * positive slope is monotone.
 */
export function transformPolytomousParameters(
  transform: ScaleTransform,
  parameters: PolytomousParameters,
): PolytomousParameters {
  return validatePolytomousParameters({
    a: parameters.a / transform.slope,
    thresholds: parameters.thresholds.map(
      (threshold) => transform.slope * threshold + transform.intercept,
    ),
    model: parameters.model,
    metric: parameters.metric,
  });
}

/**
 * Transform an item of either format, preserving its identity and metadata.
 *
 * The partial credit model fixes discrimination at 1, so an item in that family
 * cannot be rescaled without leaving the family. Rather than silently promote it
 * to the generalized model — which would change what the item *is* while
 * appearing to only move it — this throws unless the slope is 1. Linking a
 * partial credit bank means linking with `partial-credit` items calibrated on a
 * common metric, or converting the family deliberately first.
 */
export function transformItem(transform: ScaleTransform, item: AnyItem): AnyItem {
  const meta = {
    ...(item.domain === undefined ? {} : { domain: item.domain }),
    ...(item.tags === undefined ? {} : { tags: item.tags }),
  };
  if (!isPolytomous(item)) {
    return makeItem(item.id, transformItemParameters(transform, item.parameters), meta);
  }
  if (item.parameters.model === 'partial-credit' && transform.slope !== 1) {
    throw new RangeError(
      `cannot rescale partial credit item "${item.id}" by a slope of ${transform.slope}: ` +
        'the model fixes discrimination at 1, so rescaling would move it into the ' +
        'generalized partial credit family',
    );
  }
  return makePolytomousItem(
    item.id,
    transformPolytomousParameters(transform, item.parameters),
    meta,
  );
}

/** Transform a whole bank onto the target scale. */
export function transformBank(
  transform: ScaleTransform,
  items: readonly AnyItem[],
): AnyItem[] {
  return items.map((item) => transformItem(transform, item));
}

/** Format a transformation for reporting: `theta* = 1.043 theta + 0.217`. */
export function describeTransform(transform: ScaleTransform): string {
  const sign = transform.intercept < 0 ? '-' : '+';
  return (
    `theta* = ${transform.slope.toFixed(4)} theta ` +
    `${sign} ${Math.abs(transform.intercept).toFixed(4)}`
  );
}

/** A dichotomous item, narrowed. Used where a caller knows the format. */
export function asDichotomous(item: AnyItem): Item {
  if (isPolytomous(item)) {
    throw new RangeError(`item "${item.id}" is polytomous`);
  }
  return item;
}
