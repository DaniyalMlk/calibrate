/** The typographic minus sign, which aligns with digits where a hyphen does not. */
const MINUS = '−';

/**
 * Fixed-decimal formatting, with a real minus sign and no negative zero.
 *
 * A posterior mean of -1e-17 is zero, and `toFixed` renders it "-0.00". Nothing
 * is more corrosive to trust in a number than a sign it has not earned.
 */
export function fixed(value: number, digits = 2): string {
  if (!Number.isFinite(value)) {
    return '—';
  }
  const rounded = Number(value.toFixed(digits));
  const text = Math.abs(rounded).toFixed(digits);
  return rounded < 0 ? `${MINUS}${text}` : text;
}

/** Fixed-decimal formatting with an explicit sign, for differences. */
export function signed(value: number, digits = 2): string {
  if (!Number.isFinite(value)) {
    return '—';
  }
  const rounded = Number(value.toFixed(digits));
  return `${rounded < 0 ? MINUS : '+'}${Math.abs(rounded).toFixed(digits)}`;
}

/** A closed interval, as it would be written in a report. */
export function interval(lower: number, upper: number, digits = 2): string {
  return `${fixed(lower, digits)} to ${fixed(upper, digits)}`;
}

/** Percentage with no decimals. */
export function percent(fraction: number, digits = 0): string {
  return Number.isFinite(fraction) ? `${(fraction * 100).toFixed(digits)}%` : '—';
}

/**
 * Name the model a parameter set really is, as a reader would write it.
 *
 * The engine stores every dichotomous item as a 4PL, so the model is a property
 * of the parameter values rather than of a tag on the item, and this reads it
 * back off the values. A polytomous item is named by its thresholds instead:
 * there is no single difficulty to report, and writing the threshold mean where
 * a reader expects `b` would be a number they could not find in the item.
 */
export function describeItemParameters(
  parameters:
    | { readonly a: number; readonly b: number; readonly c: number; readonly d: number }
    | { readonly a: number; readonly thresholds: readonly number[] },
): string {
  if ('thresholds' in parameters) {
    const thresholds = parameters.thresholds.map((value) => signed(value, 2)).join(', ');
    return `a = ${fixed(parameters.a, 2)}   b = [${thresholds}]`;
  }
  const parts = [`a = ${fixed(parameters.a, 2)}`, `b = ${signed(parameters.b, 2)}`];
  if (parameters.c > 0) {
    parts.push(`c = ${fixed(parameters.c, 2)}`);
  }
  if (parameters.d < 1) {
    parts.push(`d = ${fixed(parameters.d, 2)}`);
  }
  return parts.join('   ');
}
