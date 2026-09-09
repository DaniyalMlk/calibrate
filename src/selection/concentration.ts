/** One point on a Lorenz curve over item exposure. */
export interface LorenzPoint {
  /** Fraction of the bank, counting from the least exposed item. */
  readonly bankFraction: number;
  /** Share of all administrations those items received. */
  readonly exposureShare: number;
}

export interface ExposureConcentration {
  /**
   * The Lorenz curve, from `(0, 0)` to `(1, 1)`.
   *
   * Point `i` says that the least-exposed `bankFraction` of the bank absorbed
   * `exposureShare` of everything administered. The diagonal is perfect
   * evenness; the further the curve sags below it, the more concentrated
   * administration is on a few items.
   */
  readonly curve: readonly LorenzPoint[];
  /**
   * Gini coefficient of the exposure distribution: twice the area between the
   * diagonal and the Lorenz curve.
   *
   * Zero when every item is administered equally often. Its maximum is
   * `(n - 1) / n`, reached when one item absorbs every administration — not
   * one, because a bank of finite size cannot concentrate perfectly.
   */
  readonly gini: number;
  /** The largest share any single item took of all administrations. */
  readonly topShare: number;
  /**
   * Share of all administrations taken by the most-exposed tenth of the bank.
   *
   * The number a bank owner asks for by name: if a tenth of the bank carries
   * most of the testing, that tenth is what a harvesting effort would target.
   */
  readonly topDecileShare: number;
  /** Items that were never administered. */
  readonly unused: number;
}

/**
 * Concentration of item exposure across a bank.
 *
 * The peak exposure rate, the overlap rate and the unused fraction each say
 * something true about bank security and none of them describes the shape of
 * the distribution. A bank where thirty items carry sixty percent of all
 * administrations and one where a hundred and twenty items carry the same sixty
 * percent can report the same peak rate and the same unused fraction; only the
 * shape separates them, and the shape is what says how much of the bank an
 * attacker would have to harvest to reconstruct most of the testing.
 *
 * Items in the bank that never appear in `rates` count as zero-exposure items,
 * which is what makes `bankSize` a required argument rather than something
 * inferred from the map: an unused item is the most concentrating thing a bank
 * can contain, and a curve computed only over the items that were used would
 * omit exactly the evidence that matters.
 */
export function exposureConcentration(
  bankSize: number,
  rates: ReadonlyMap<string, number>,
): ExposureConcentration {
  if (!Number.isInteger(bankSize) || bankSize < 1) {
    throw new RangeError(
      `exposureConcentration: bankSize must be a positive integer, received ${bankSize}`,
    );
  }
  if (rates.size > bankSize) {
    throw new RangeError(
      `exposureConcentration: ${rates.size} exposure rates for a bank of ${bankSize} items`,
    );
  }

  const values: number[] = [];
  for (const rate of rates.values()) {
    if (!Number.isFinite(rate) || rate < 0) {
      throw new RangeError(`exposureConcentration: exposure rates must be finite and non-negative, received ${rate}`);
    }
    values.push(rate);
  }
  while (values.length < bankSize) {
    values.push(0);
  }
  values.sort((a, b) => a - b);

  const total = values.reduce((sum, value) => sum + value, 0);
  const unused = values.filter((value) => value === 0).length;

  if (total === 0) {
    // Nothing was administered. Every item has an equal share of nothing, which
    // is even rather than concentrated — reporting a Gini of one here would say
    // an untouched bank is maximally compromised.
    return {
      curve: [
        { bankFraction: 0, exposureShare: 0 },
        { bankFraction: 1, exposureShare: 1 },
      ],
      gini: 0,
      topShare: 0,
      topDecileShare: 0,
      unused: bankSize,
    };
  }

  const curve: LorenzPoint[] = [{ bankFraction: 0, exposureShare: 0 }];
  let cumulative = 0;
  for (let i = 0; i < bankSize; i += 1) {
    cumulative += values[i] as number;
    curve.push({ bankFraction: (i + 1) / bankSize, exposureShare: cumulative / total });
  }

  // Gini from the sorted shares directly, rather than by integrating the curve:
  //
  //   G = (2 * sum_i i * x_i) / (n * sum_i x_i) - (n + 1) / n,  i from 1
  //
  // with `x` ascending. This is the exact discrete definition, so an even
  // distribution returns a clean zero instead of the small positive residue a
  // trapezoid integration of the curve would leave behind.
  let weighted = 0;
  for (let i = 0; i < bankSize; i += 1) {
    weighted += (i + 1) * (values[i] as number);
  }
  const gini = Math.max(
    0,
    (2 * weighted) / (bankSize * total) - (bankSize + 1) / bankSize,
  );

  const decileCount = Math.max(1, Math.ceil(bankSize / 10));
  let topDecile = 0;
  for (let i = bankSize - decileCount; i < bankSize; i += 1) {
    topDecile += values[i] as number;
  }

  return {
    curve,
    gini,
    topShare: (values[bankSize - 1] as number) / total,
    topDecileShare: topDecile / total,
    unused,
  };
}
