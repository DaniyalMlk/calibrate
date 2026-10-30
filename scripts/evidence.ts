/**
 * The figures the README's evidence table quotes, recomputed from the library.
 *
 * Every row in "What is checked, and against what" comes from here, and this
 * runs in CI. If a figure moves past the bound its row claims, the process
 * exits non-zero and the table cannot quietly go stale.
 *
 * Two rules decide how a row is written, and they are not interchangeable:
 *
 * - A row that is an exact-arithmetic identity says **zero**. `a^2 / 4` is the
 *   peak information of a 2PL item by construction, not approximately, so
 *   publishing "small" there would hide a real regression inside a tolerance.
 * - A row that is floating-point residue publishes a **bound**, never the
 *   digits. The last digits of a residue are not portable — a different
 *   interpreter, a different order of summation, and they change — so a table
 *   quoting them would be wrong somewhere without anything being broken.
 *
 * The bounds were chosen by measuring what they refuse, which is the only way
 * to know a tolerance is doing anything. Slipping one difficulty by 1e-9 moves
 * the Raju row by 1e-9, a hundred times its 1e-11 bound; slipping a chi-square
 * statistic by 1e-12 moves that row by 3e-14, past its 1e-14. The exact rows
 * refuse any difference at all above one unit in the last place: evaluating
 * the information 1e-7 off the peak already fails, and only a displacement
 * near 1e-9 — which changes the value by less than a double can represent —
 * goes unnoticed.
 */

import { linspace, variance } from '../src/core/numeric.js';
import { chiSquareUpperTail } from '../src/core/gamma.js';
import { normalGaussHermiteRule } from '../src/core/quadrature.js';
import { createRng } from '../src/core/random.js';
import {
  makeItem,
  metricScale,
  onePL,
  rasch,
  twoPL,
  type Item,
  type Metric,
} from '../src/models/item.js';
import { categoryProbabilities, graded, partialCredit } from '../src/models/polytomous.js';
import { categoryProbabilitiesOf, type AnyItem } from '../src/models/mixed.js';
import {
  informationPeak,
  itemInformation,
  probabilityCorrect,
} from '../src/models/response.js';
import { estimateMle } from '../src/estimation/mle.js';
import { personFit } from '../src/estimation/person.js';
import { simulateResponses } from '../src/simulation/respondent.js';
import { scoreDistribution } from '../src/scoring/summed.js';
import { integrateArea, rajuArea } from '../src/dif/area.js';
import { commonItems } from '../src/linking/common.js';
import { haebara, meanMean, meanSigma, stockingLord } from '../src/linking/methods.js';
import { scaleTransform, transformItem } from '../src/linking/scale.js';
import { simulateMatrix } from '../src/simulation/respondent.js';
import { calibrate } from '../src/calibration/jmle.js';
import { marginalCalibrate } from '../src/calibration/mml.js';

export interface Check {
  /** The row label, which is also the left-hand cell of the README table. */
  readonly label: string;
  /** The rendered right-hand cell. */
  readonly result: string;
  /** False when the figure has moved past what the row claims. */
  readonly ok: boolean;
  /** What went wrong, when it did. */
  readonly complaint?: string;
}

/** A row whose claim is that a residue stays under a bound. */
function bounded(label: string, worst: number, bound: number, over: string): Check {
  const ok = Number.isFinite(worst) && worst <= bound;
  return {
    label,
    result: `below ${format(bound)}, ${over}`,
    ok,
    ...(ok ? {} : { complaint: `worst was ${worst}, over the ${format(bound)} the row claims` }),
  };
}

/** A row whose claim is that the difference is exactly zero. */
function exact(label: string, worst: number, over: string): Check {
  const ok = worst === 0;
  return {
    label,
    result: `0, ${over}`,
    ok,
    ...(ok ? {} : { complaint: `expected exact agreement, worst difference was ${worst}` }),
  };
}

function format(value: number): string {
  const exponent = Math.round(Math.log10(value));
  return `1e-${-exponent}`;
}

const GRID = linspace(-4, 4, 81);

/**
 * The peak of a 2PL item's information is `(D a)^2 / 4`, exactly.
 *
 * Over both metrics, because `D` is the whole content of the claim: a row
 * measured only on the logistic metric, where `D` is one, would read the same
 * whether the information function applied the scaling or ignored it.
 */
function peakInformation(): Check {
  let worst = 0;
  let sets = 0;
  for (const metric of ['logistic', 'normal'] as const) {
    const d = metricScale(metric);
    for (const a of linspace(0.2, 3, 57)) {
      for (const b of linspace(-3, 3, 13)) {
        const item = twoPL(a, b, metric);
        worst = Math.max(
          worst,
          Math.abs(itemInformation(item, informationPeak(item)) - (d * a) ** 2 / 4),
        );
        sets += 1;
      }
    }
  }
  return exact(
    'Peak information of a 2PL item against the closed form `(D a)^2 / 4`, on both metrics',
    worst,
    `over ${sets.toLocaleString('en-GB')} parameter sets`,
  );
}

/** Category probabilities are a distribution, so they sum to one. */
function categoriesSumToOne(): Check {
  const items = [
    graded(1.2, [-1.5, -0.2, 0.9]),
    graded(0.6, [-2, 2]),
    graded(2.1, [-0.4, -0.3, -0.2, 1.7]),
    partialCredit([-0.5, 0.8]),
    partialCredit([1.2, -0.3]),
  ];
  let worst = 0;
  let points = 0;
  for (const item of items) {
    for (const theta of GRID) {
      const total = categoryProbabilities(item, theta).reduce((a, b) => a + b, 0);
      worst = Math.max(worst, Math.abs(total - 1));
      points += 1;
    }
  }
  return bounded(
    'Category probabilities summing to one, graded and partial credit',
    worst,
    1e-14,
    `over ${points.toLocaleString('en-GB')} evaluations`,
  );
}

/**
 * A two-category graded item *is* the 2PL, and a two-category partial credit
 * item *is* the Rasch model.
 *
 * The README called this agreement to fourteen digits. It is not: the
 * polytomous code path reduces to the same arithmetic, so the difference is
 * exactly zero, and a row saying "fourteen digits" would accept a real
 * divergence appearing in the fifteenth.
 */
function twoCategoryReduction(): Check {
  let worst = 0;
  let points = 0;
  for (const a of linspace(0.3, 2.5, 23)) {
    for (const b of linspace(-2, 2, 17)) {
      const gradedItem = graded(a, [b]);
      const dichotomous = twoPL(a, b);
      for (const theta of GRID) {
        const category = categoryProbabilities(gradedItem, theta)[1] as number;
        worst = Math.max(worst, Math.abs(category - probabilityCorrect(dichotomous, theta)));
        points += 1;
      }
    }
  }
  for (const b of linspace(-2, 2, 17)) {
    const stepItem = partialCredit([b]);
    const raschItem = rasch(b);
    for (const theta of GRID) {
      const category = categoryProbabilities(stepItem, theta)[1] as number;
      worst = Math.max(worst, Math.abs(category - probabilityCorrect(raschItem, theta)));
      points += 1;
    }
  }
  return exact(
    'A two-category graded item against the 2PL, and two-category partial credit against Rasch',
    worst,
    `over ${points.toLocaleString('en-GB')} evaluations`,
  );
}

/** Enumerate every response pattern and bucket it by total score. */
function enumerateTotals(items: readonly AnyItem[], theta: number): number[] {
  const rows = items.map((item) => categoryProbabilitiesOf(item, theta));
  const maximum = rows.reduce((total, row) => total + row.length - 1, 0);
  const out = new Array<number>(maximum + 1).fill(0);
  const walk = (index: number, score: number, probability: number): void => {
    if (index === rows.length) {
      out[score] = (out[score] as number) + probability;
      return;
    }
    const row = rows[index] as readonly number[];
    for (let category = 0; category < row.length; category += 1) {
      walk(index + 1, score + category, probability * (row[category] as number));
    }
  };
  walk(0, 0, 1);
  return out;
}

/**
 * The recursion over items against enumerating every pattern.
 *
 * The recursion is linear in the number of items and the enumeration is
 * exponential, which is the whole reason the recursion exists — so the
 * enumeration is the reference, at a form size where it is still affordable.
 */
function scoreDistributionAgainstEnumeration(): Check {
  const form: AnyItem[] = linspace(-2, 2, 15).map((b, index) =>
    makeItem(`i${index}`, twoPL(0.6 + index * 0.1, b)),
  );
  let worst = 0;
  for (const theta of [-2, -1, -0.25, 0, 0.5, 1, 2]) {
    const byRecursion = scoreDistribution(form, theta);
    const byEnumeration = enumerateTotals(form, theta);
    for (let total = 0; total < byRecursion.length; total += 1) {
      worst = Math.max(
        worst,
        Math.abs((byRecursion[total] as number) - (byEnumeration[total] as number)),
      );
    }
  }
  return bounded(
    'The total-score distribution by recursion against enumerating every pattern',
    worst,
    1e-14,
    `over ${(2 ** 15).toLocaleString('en-GB')} patterns at 7 abilities`,
  );
}

/** Every pattern of a given length, as 0/1 arrays. */
function everyPattern(length: number): number[][] {
  const out: number[][] = [];
  for (let mask = 0; mask < 1 << length; mask += 1) {
    const row: number[] = [];
    for (let index = 0; index < length; index += 1) row.push((mask >> index) & 1);
    out.push(row);
  }
  return out;
}

/** The widest disagreement between two patterns sharing a total score. */
function sufficiencySpread(form: readonly Item[], tolerance?: number): number {
  const byTotal = new Map<number, number[]>();
  const options = tolerance === undefined ? {} : { tolerance };
  for (const pattern of everyPattern(form.length)) {
    const total = pattern.reduce((a, b) => a + b, 0);
    if (total === 0 || total === form.length) continue;
    const responses = form.map((item, index) => ({
      item,
      response: pattern[index] as 0 | 1,
    }));
    const estimates = byTotal.get(total) ?? [];
    estimates.push(estimateMle(responses, options).theta);
    byTotal.set(total, estimates);
  }
  let worst = 0;
  for (const estimates of byTotal.values()) {
    worst = Math.max(worst, Math.max(...estimates) - Math.min(...estimates));
  }
  return worst;
}

/**
 * Under Rasch the total score is sufficient for ability; under the 2PL it is not.
 *
 * Both halves are the claim. A check that only confirmed the Rasch side would
 * pass just as happily against an estimator that ignored which items were
 * answered correctly, so the 2PL figure is what gives the first one meaning.
 */
function sufficiency(): Check {
  const length = 10;
  const difficulties = linspace(-2, 2, length);
  const raschForm = difficulties.map((b, index) => makeItem(`r${index}`, onePL(b)));
  const twoPlForm = difficulties.map((b, index) =>
    makeItem(`t${index}`, twoPL(0.5 + index * 0.18, b)),
  );

  // Sufficiency is exact in the model: two patterns with the same total score
  // have the same likelihood, so they have the same maximiser. What is left
  // over is the root-finder's stopping rule and nothing else, and the way to
  // show that is to move the stopping rule and watch the residue follow it
  // down. A single figure under a bound could not tell the two apart.
  const atDefault = sufficiencySpread(raschForm);
  const atTight = sufficiencySpread(raschForm, 1e-14);
  const twoPl = sufficiencySpread(twoPlForm);

  const ok = atDefault <= 1e-7 && atTight <= 1e-14 && twoPl > 1;
  return {
    label: 'Two patterns with the same total score, scored under Rasch and under the 2PL',
    result:
      `the same ability over all ${(2 ** length).toLocaleString('en-GB')} patterns, to the ` +
      `root-finder's tolerance and no further: below 1e-7 at its default of 1e-8, below ` +
      `1e-14 when that is tightened to 1e-14. Under the 2PL the same comparison spans ` +
      `${twoPl.toFixed(2)} logits`,
    ok,
    ...(ok
      ? {}
      : {
          complaint:
            `default ${atDefault}, tightened ${atTight}, 2PL ${twoPl}` +
            ' — the residue should track the tolerance, and the 2PL should not be sufficient',
        }),
  };
}

/** All four linking methods against a transformation that is known exactly. */
function linkingRecovery(): Check {
  const bank: Item[] = Array.from({ length: 30 }, (_, index) =>
    makeItem(`k${index}`, twoPL(0.5 + (index % 7) * 0.2, -2.2 + index * 0.15)),
  );
  const known = scaleTransform(1.35, -0.42);
  const moved = bank.map((item) => transformItem(known, item) as Item);
  const pairs = commonItems(bank, moved);
  let worst = 0;
  for (const method of [meanMean, meanSigma, haebara, stockingLord]) {
    const found = method(pairs);
    if (!found.converged) return { label: 'linking', result: '', ok: false, complaint: 'a method did not converge' };
    worst = Math.max(
      worst,
      Math.abs(found.transform.slope - known.slope),
      Math.abs(found.transform.intercept - known.intercept),
    );
  }
  return bounded(
    'Four linking methods recovering a transformation applied on purpose (A 1.35, B -0.42)',
    worst,
    1e-14,
    'in both coefficients, 30 common items',
  );
}

/**
 * The four methods on an anchor whose two calibrations chose different metrics.
 *
 * The same thirty curves written twice — once with `D = 1.702` carried on the
 * item and once with it multiplied into the discrimination — so the two
 * calibrations agree about every probability and the transformation between
 * them is the identity. The criterion methods see only curves and cannot be
 * fooled by the convention; the moment methods read parameters, and mean/mean
 * reads the one parameter the convention changes. It is the row that decides
 * whether the metric is inside the ratio or outside it: with the raw
 * discriminations averaged instead, mean/mean reports a slope of 1/1.702.
 */
function crossMetricLinking(): Check {
  const known = scaleTransform(1, 0);
  const onNormal: Item[] = Array.from({ length: 30 }, (_, index) =>
    makeItem(`x${index}`, twoPL(0.5 + (index % 7) * 0.2, -2.2 + index * 0.15, 'normal')),
  );
  const onLogistic = onNormal.map((item) =>
    makeItem(
      item.id,
      twoPL(metricScale('normal') * item.parameters.a, item.parameters.b, 'logistic'),
    ),
  );
  const pairs = commonItems(onNormal, onLogistic);
  let worst = 0;
  for (const method of [meanMean, meanSigma, haebara, stockingLord]) {
    const found = method(pairs);
    if (!found.converged) {
      return {
        label: 'cross-metric linking',
        result: '',
        ok: false,
        complaint: 'a method did not converge',
      };
    }
    worst = Math.max(
      worst,
      Math.abs(found.transform.slope - known.slope),
      Math.abs(found.transform.intercept - known.intercept),
    );
  }
  return bounded(
    'The same four methods on an anchor calibrated once on each metric, where the ' +
      'transformation is the identity',
    worst,
    1e-14,
    'in both coefficients, 30 common items',
  );
}

/**
 * Raju's areas in closed form against numerical integration of the same curves.
 *
 * The closed form is the thing being checked; the quadrature is slow, obvious
 * and independent of it, which is what makes it a reference rather than a
 * second opinion from the same code.
 */
function areaAgainstQuadrature(): Check {
  // Four of these pairs are written on one metric and four span two, because
  // the closed form reads the slope of each exponent and a pair that agrees
  // about the convention cannot tell whether it read that slope once for the
  // pair or once per item. The quadrature reads each item's own metric and
  // knows nothing about the closed form either way.
  const cases: readonly (readonly [number, number, Metric, number, number, Metric])[] = [
    [1.2, -0.5, 'logistic', 1.2, 0.3, 'logistic'],
    [0.8, 0.1, 'logistic', 1.6, -0.4, 'logistic'],
    [1.0, -1, 'logistic', 1.4, 1.1, 'logistic'],
    [1.5, 0.2, 'logistic', 0.9, 0.2, 'logistic'],
    [0.7, -1.4, 'logistic', 0.7, -1.4, 'logistic'],
    [0.6, -0.4, 'normal', 1.9, 0.5, 'logistic'],
    [1.4, 0.3, 'logistic', 0.5, -0.8, 'normal'],
    [0.8, 0.2, 'normal', 1.1, -0.6, 'normal'],
    [1.3, -0.25, 'logistic', 1.3 / metricScale('normal'), 0.75, 'normal'],
  ];
  let worst = 0;
  for (const [aReference, bReference, first, aFocal, bFocal, second] of cases) {
    const reference = makeItem('x', twoPL(aReference, bReference, first));
    const focal = makeItem('x', twoPL(aFocal, bFocal, second));
    const closed = rajuArea(reference, focal);
    const numeric = integrateArea(reference, focal);
    worst = Math.max(
      worst,
      Math.abs(closed.signed - numeric.signed),
      Math.abs(closed.unsigned - numeric.unsigned),
    );
  }
  return bounded(
    "Raju's signed and unsigned areas in closed form against quadrature",
    worst,
    1e-11,
    `over ${cases.length} item pairs, crossing and not, on one metric and across two`,
  );
}

/** The chi-square tail against published critical values. */
function chiSquareAgainstTables(): Check {
  const published: readonly (readonly [number, number, number])[] = [
    [3.841458820694124, 1, 0.05],
    [5.991464547107979, 2, 0.05],
    [11.070497693516351, 5, 0.05],
    [18.307038053275146, 10, 0.05],
    [6.634896601021214, 1, 0.01],
    [23.209251158954356, 10, 0.01],
    [10.827566170662733, 1, 0.001],
  ];
  let worst = 0;
  for (const [statistic, degrees, tail] of published) {
    worst = Math.max(worst, Math.abs(chiSquareUpperTail(statistic, degrees) - tail));
  }
  return bounded(
    'The chi-square upper tail at published critical values, 1 to 10 degrees of freedom',
    worst,
    1e-14,
    `over ${published.length} points from 0.001 to 0.05`,
  );
}

/**
 * The structural bias of joint estimation, against its own closed form.
 *
 * Joint maximum likelihood estimates every person parameter alongside every
 * item parameter, and the person parameters do not stop arriving as the sample
 * grows — which is why the bias does not vanish with more data. For the Rasch
 * model the leading term stretches the difficulty scale by `L / (L - 1)` in a
 * test of `L` items, so this is a measurement against a number known in
 * advance rather than against "less than the other one".
 */
function jointBias(): readonly Check[] {
  const rule = normalGaussHermiteRule(0, 1, 41);
  const seeds = [101, 202, 303, 404, 505];
  const rows: Check[] = [];

  for (const length of [10, 20]) {
    const difficulties = linspace(-2, 2, 30).slice(0, length);
    const bank = difficulties.map((b, index) => makeItem(`i-${index}`, onePL(b)));
    let joint = 0;
    let marginal = 0;
    for (const seed of seeds) {
      const rng = createRng(seed);
      const abilities = Array.from({ length: 2000 }, () => rng.nextNormal());
      const matrix = simulateMatrix(bank, abilities, seed + 7);
      joint += spreadRatio(
        calibrate(matrix, { model: 'rasch', biasCorrection: false }),
        bank,
        difficulties,
      );
      marginal += spreadRatio(marginalCalibrate(matrix, { rule }), bank, difficulties);
    }
    joint /= seeds.length;
    marginal /= seeds.length;
    const predicted = length / (length - 1);
    const gap = Math.abs(joint - predicted);
    // The claim has two halves. The joint estimator's inflation matches the
    // number predicted in advance, and the marginal estimator carries no such
    // term — which means it sits nearer to 1 than the joint estimator does,
    // not nearer to 1 than the joint estimator sits to its own prediction.
    // Asserting the second thing by mistake fails against a correct result.
    const ok = gap < 0.01 && Math.abs(marginal - 1) < Math.abs(joint - 1) / 4;
    rows.push({
      label:
        `Joint estimation's scale inflation on a ${length}-item test, ` +
        'against the closed form `L / (L - 1)`',
      result:
        `${joint.toFixed(4)} measured against ${predicted.toFixed(4)} predicted; ` +
        `the marginal estimator reads ${marginal.toFixed(4)}`,
      ok,
      ...(ok
        ? {}
        : {
            complaint:
              `joint ${joint} against predicted ${predicted} (gap ${gap}), ` +
              `marginal ${marginal}`,
          }),
    });
  }
  return rows;
}

/** How far a set of estimates stretches the scale, against the truth. */
function spreadRatio(
  result: { readonly items: readonly { readonly id: string; readonly difficulty: number }[] },
  bank: readonly Item[],
  truth: readonly number[],
): number {
  const estimates = result.items.map((item) => item.difficulty);
  const aligned = result.items.map(
    (item) => truth[bank.findIndex((candidate) => candidate.id === item.id)] as number,
  );
  return Math.sqrt(variance(estimates) / variance(aligned));
}


/**
 * The null distribution of the person-fit statistic, before and after the
 * correction for an estimated ability.
 *
 * Three numbers, because the claim has three parts and checking only the last
 * would pass against a correction applied to a statistic that never needed
 * one: `lz` is standard normal at a known ability, too narrow once the ability
 * is estimated from the same responses, and brought back by `lzStar`.
 */
function personFitCalibration(): Check {
  const form: Item[] = linspace(-2, 2, 40).map((b, index) =>
    makeItem(`p${index}`, twoPL(0.7 + ((index * 7) % 40) / 40, b)),
  );
  const rng = createRng(20260101);
  const atTruth: number[] = [];
  const atEstimate: number[] = [];
  const corrected: number[] = [];

  for (let person = 0; person < 3000; person += 1) {
    const theta = rng.nextNormal();
    const responses = simulateResponses(form, theta, rng);
    const estimate = estimateMle(responses);
    if (!estimate.converged) continue;
    atTruth.push(personFit(responses, theta).lz);
    const fit = personFit(responses, estimate.theta);
    atEstimate.push(fit.lz);
    corrected.push(fit.lzStar as number);
  }

  const sd = (values: readonly number[]): number => Math.sqrt(variance(values));
  const known = sd(atTruth);
  const estimated = sd(atEstimate);
  const fixed = sd(corrected);
  const ok =
    Math.abs(known - 1) < 0.05 && estimated < 0.95 && Math.abs(fixed - 1) < Math.abs(estimated - 1);

  return {
    label: 'The person-fit statistic on a 40-item form, at a known ability and an estimated one',
    result:
      `standard deviation ${known.toFixed(2)} when the ability is known, ` +
      `${estimated.toFixed(2)} when it is estimated from the same responses, and ` +
      `${fixed.toFixed(2)} once corrected; over ${atTruth.length.toLocaleString('en-GB')} ` +
      'model-conforming candidates',
    ok,
    ...(ok
      ? {}
      : {
          complaint:
            `known ${known}, estimated ${estimated}, corrected ${fixed} — the middle one ` +
            'should be visibly below one and the last should be nearer to it',
        }),
  };
}

/** Every row of the evidence table, in the order the README prints them. */
export function measure(): readonly Check[] {
  return [
    peakInformation(),
    twoCategoryReduction(),
    categoriesSumToOne(),
    scoreDistributionAgainstEnumeration(),
    sufficiency(),
    linkingRecovery(),
    crossMetricLinking(),
    areaAgainstQuadrature(),
    chiSquareAgainstTables(),
    ...jointBias(),
    personFitCalibration(),
  ];
}

/** The rows as the README prints them. */
export function renderTable(checks: readonly Check[]): string {
  const lines = ['| Check | Result |', '| --- | --- |'];
  for (const check of checks) lines.push(`| ${check.label} | ${check.result} |`);
  return lines.join('\n');
}

/**
 * Print the table and say whether it still holds.
 *
 * Kept separate from the entry point in `evidence-main.ts` so the tests can
 * import `measure` and compare it against the README without a process exiting
 * underneath them.
 */
export function report(): number {
  const checks = measure();
  process.stdout.write(`${renderTable(checks)}\n`);
  const failed = checks.filter((check) => !check.ok);
  if (failed.length > 0) {
    process.stderr.write('\n');
    for (const check of failed) {
      process.stderr.write(`FAILED: ${check.label}\n  ${check.complaint ?? ''}\n`);
    }
    return 1;
  }
  process.stdout.write(`\n${checks.length} checks, all within what the table claims\n`);
  return 0;
}
