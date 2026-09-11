/**
 * The ordinal ramp: colour for a set whose order carries meaning.
 *
 * An item's response categories are ordered — a 3 is a better answer than a 2 —
 * so they are not N identities to be told apart but one scale to be read along,
 * and they take a single hue at stepped lightness rather than N slots from the
 * series palette. N distinct hues would say these curves are unrelated, which
 * is the one thing about them that is false.
 *
 * The steps themselves live in the stylesheet, once per theme, because the ramp
 * is re-stepped for each surface rather than flipped. This module only decides
 * which step a category gets.
 */

const RAMP = [
  'var(--ordinal-1)',
  'var(--ordinal-2)',
  'var(--ordinal-3)',
  'var(--ordinal-4)',
  'var(--ordinal-5)',
  'var(--ordinal-6)',
] as const;

/** The steps of the ramp, lowest category first. */
export const ORDINAL_RAMP: readonly string[] = RAMP;

/**
 * The colour for category `k` of an item with `count` categories.
 *
 * The ramp is walked end to end whatever the category count, so the first step
 * always means the bottom category and the last always means the top.
 * Assigning from one end and stopping early instead would make a
 * three-category item's top score wear the colour a six-category item uses for
 * its middle, and a reader comparing two items in one session would read that
 * as a difference between the items rather than between their rubrics.
 */
export function rampColour(k: number, count: number): string {
  if (count <= 1) return RAMP[RAMP.length - 1] as string;
  const position = (k / (count - 1)) * (RAMP.length - 1);
  return RAMP[Math.round(position)] as string;
}
