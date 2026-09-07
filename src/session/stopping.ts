import type { AbilityEstimate } from '../estimation/mle.js';

/** What the session knows when it asks whether to stop. */
export interface StoppingContext {
  /** How many items have been administered and answered. */
  readonly administered: number;
  /** The current ability estimate. */
  readonly estimate: AbilityEstimate;
  /** How many eligible items are left in the pool. */
  readonly remaining: number;
}

/** Why a session ended. */
export interface StopReason {
  /** Which rule fired. */
  readonly rule: string;
  /** Human-readable detail, suitable for a transcript. */
  readonly detail: string;
}

/** A stopping rule: returns a reason to stop, or `null` to continue. */
export interface StoppingRule {
  readonly name: string;
  shouldStop(context: StoppingContext): StopReason | null;
}

/** Stop after exactly `length` items. */
export function fixedLength(length: number): StoppingRule {
  if (!Number.isInteger(length) || length < 1) {
    throw new RangeError(`fixedLength: length must be a positive integer, received ${length}`);
  }
  return {
    name: `fixed-length(${length})`,
    shouldStop({ administered }): StopReason | null {
      return administered >= length
        ? { rule: `fixed-length(${length})`, detail: `administered ${administered} of ${length} items` }
        : null;
    },
  };
}

/**
 * Stop once the standard error falls below `target`.
 *
 * On its own this rule is dangerous: two lucky answers can produce a small
 * standard error on a pattern that carries almost no real information about the
 * candidate. Compose it with `withMinimumLength` — which is why that guard
 * exists as a wrapper rather than an option.
 */
export function standardErrorBelow(target: number): StoppingRule {
  if (!(target > 0)) {
    throw new RangeError(`standardErrorBelow: target must be positive, received ${target}`);
  }
  return {
    name: `standard-error-below(${target})`,
    shouldStop({ estimate }): StopReason | null {
      return estimate.standardError <= target
        ? {
            rule: `standard-error-below(${target})`,
            detail: `standard error ${estimate.standardError.toFixed(4)} reached target ${target}`,
          }
        : null;
    },
  };
}

/** Stop once `limit` items have been administered, whatever else is true. */
export function maximumItems(limit: number): StoppingRule {
  if (!Number.isInteger(limit) || limit < 1) {
    throw new RangeError(`maximumItems: limit must be a positive integer, received ${limit}`);
  }
  return {
    name: `maximum-items(${limit})`,
    shouldStop({ administered }): StopReason | null {
      return administered >= limit
        ? { rule: `maximum-items(${limit})`, detail: `reached the ${limit}-item ceiling` }
        : null;
    },
  };
}

/** Suppress an inner rule until at least `minimum` items have been administered. */
export function withMinimumLength(minimum: number, inner: StoppingRule): StoppingRule {
  if (!Number.isInteger(minimum) || minimum < 0) {
    throw new RangeError(
      `withMinimumLength: minimum must be a non-negative integer, received ${minimum}`,
    );
  }
  return {
    name: `min-length(${minimum}, ${inner.name})`,
    shouldStop(context): StopReason | null {
      if (context.administered < minimum) return null;
      return inner.shouldStop(context);
    },
  };
}

/** Stop as soon as any of the given rules fires. */
export function anyOf(...rules: readonly StoppingRule[]): StoppingRule {
  if (rules.length === 0) throw new RangeError('anyOf: at least one rule is required');
  return {
    name: `any-of(${rules.map((r) => r.name).join(', ')})`,
    shouldStop(context): StopReason | null {
      for (const rule of rules) {
        const reason = rule.shouldStop(context);
        if (reason !== null) return reason;
      }
      return null;
    },
  };
}

/** Stop only when every given rule fires; reports the last one to do so. */
export function allOf(...rules: readonly StoppingRule[]): StoppingRule {
  if (rules.length === 0) throw new RangeError('allOf: at least one rule is required');
  const name = `all-of(${rules.map((r) => r.name).join(', ')})`;
  return {
    name,
    shouldStop(context): StopReason | null {
      const details: string[] = [];
      for (const rule of rules) {
        const reason = rule.shouldStop(context);
        if (reason === null) return null;
        details.push(reason.detail);
      }
      return { rule: name, detail: details.join('; ') };
    },
  };
}

/**
 * The conventional operational configuration: stop at the target precision, but
 * never before `minimum` items and never after `maximum`.
 */
export function precisionTarget(
  target: number,
  { minimum = 5, maximum = 40 }: { minimum?: number; maximum?: number } = {},
): StoppingRule {
  if (minimum > maximum) {
    throw new RangeError(
      `precisionTarget: minimum (${minimum}) must not exceed maximum (${maximum})`,
    );
  }
  return anyOf(withMinimumLength(minimum, standardErrorBelow(target)), maximumItems(maximum));
}
