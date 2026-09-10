/** Left-align `text` in a field `width` wide. */
export function pad(text: string, width: number): string {
  return text.length >= width ? text : text + ' '.repeat(width - text.length);
}

/** Right-align `text` in a field `width` wide. */
export function padStart(text: string, width: number): string {
  return text.length >= width ? text : ' '.repeat(width - text.length) + text;
}

/**
 * Render a scored category for a transcript column.
 *
 * A dichotomous item reads as right or wrong, which is what it means. A
 * polytomous item reads as the level awarded out of the level available: "2/3"
 * says more than "partial" and fits the same column width.
 */
export function scoreLabel(category: number, maximumScore: number): string {
  if (maximumScore === 1) return category === 1 ? 'right' : 'wrong';
  return `${category}/${maximumScore}`;
}
