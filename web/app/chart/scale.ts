export interface Scale {
  /** Map a data value onto a pixel coordinate. */
  readonly to: (value: number) => number;
  /** Map a pixel coordinate back onto a data value. */
  readonly from: (pixel: number) => number;
  readonly domain: readonly [number, number];
  readonly range: readonly [number, number];
}

/** A linear scale between a data domain and a pixel range. */
export function linearScale(
  domain: readonly [number, number],
  range: readonly [number, number],
): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  const factor = span === 0 ? 0 : (r1 - r0) / span;

  return {
    to: (value: number) => r0 + (value - d0) * factor,
    from: (pixel: number) => (factor === 0 ? d0 : d0 + (pixel - r0) / factor),
    domain,
    range,
  };
}

/** Multipliers a reader would have chosen, in order of preference. */
const MULTIPLIERS: readonly { readonly value: number; readonly preferred: boolean }[] = [
  { value: 1, preferred: true },
  { value: 2, preferred: true },
  { value: 2.5, preferred: false },
  { value: 5, preferred: true },
];

function ticksAt(domain: readonly [number, number], step: number): number[] {
  const [lo, hi] = domain;
  const first = Math.ceil(lo / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let i = 0; ; i += 1) {
    const value = first + i * step;
    if (value > hi + step * 1e-9) {
      break;
    }
    // Re-round to kill the accumulated binary error that turns
    // 0.30000000000000004 into an axis label.
    ticks.push(Number((Math.round(value / step) * step).toFixed(10)));
  }
  return ticks;
}

/**
 * Tick values at a "nice" step covering a domain.
 *
 * Steps come from 1, 2, 2.5 and 5 times a power of ten, so ticks land on the
 * numbers a reader would have chosen — 0, 0.5, 1.0 rather than 0, 0.43, 0.86.
 *
 * The step is picked by whose tick count lands closest to the target, rather
 * than by rounding the ideal spacing up to the next nice value. Rounding up
 * fails badly at the edges: over [-4, 4] asking for four ticks gives an ideal
 * spacing of 2.67, which rounds up to 5, whose only multiple inside the domain
 * is zero — an axis with one label on it. Comparing counts instead picks 2 and
 * yields the five labels anyone would have drawn by hand.
 */
export function niceTicks(domain: readonly [number, number], count = 5): number[] {
  const [lo, hi] = domain;
  if (!(hi > lo) || count < 2) {
    return [lo];
  }

  const rough = (hi - lo) / (count - 1);
  const exponent = Math.floor(Math.log10(rough));

  let best: number[] = [];
  let bestScore = Infinity;
  for (const power of [exponent - 1, exponent, exponent + 1]) {
    const magnitude = 10 ** power;
    for (const { value, preferred } of MULTIPLIERS) {
      const step = value * magnitude;
      const ticks = ticksAt(domain, step);
      if (ticks.length < 2) {
        continue;
      }
      // Distance from the target count, then a nudge against 2.5 and a nudge
      // towards fewer ticks, so ties resolve the way a person would resolve them.
      const score = Math.abs(ticks.length - count) + (preferred ? 0 : 0.25) + ticks.length * 1e-4;
      if (score < bestScore) {
        bestScore = score;
        best = ticks;
      }
    }
  }

  return best.length >= 2 ? best : ticksAt(domain, (hi - lo) / (count - 1));
}

/** Clamp a value into a closed interval. */
export function clampTo(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value;
}

/**
 * Decimal places needed to write a tick step exactly.
 *
 * Not a guess from the magnitude: a step of 0.25 needs two places, and one
 * place rounds it to 0.3 — which puts the label "0.3" beside a gridline that
 * is at 0.25 and beside a reference line that really is at 0.30. An axis that
 * misreports where its own gridlines are undermines every number on the plot.
 */
export function tickDecimals(step: number): number {
  const magnitude = Math.abs(step);
  if (!Number.isFinite(magnitude) || magnitude === 0) {
    return 2;
  }
  for (let decimals = 0; decimals <= 6; decimals += 1) {
    if (Math.abs(Number(magnitude.toFixed(decimals)) - magnitude) < magnitude * 1e-9) {
      return decimals;
    }
  }
  return 6;
}
