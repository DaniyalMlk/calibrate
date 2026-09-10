import { calibrate, toItems } from '../calibration/jmle.js';
import { mean, variance } from '../core/numeric.js';
import { normalGridRule } from '../core/quadrature.js';
import { createRng } from '../core/random.js';
import { commonItems, locationParameters } from '../linking/common.js';
import { linkAll, stockingLordCriterion, type LinkingResult } from '../linking/methods.js';
import { IDENTITY_TRANSFORM, transformAbility, transformBank } from '../linking/scale.js';
import { makeItem, twoPL, type Item } from '../models/item.js';
import { testCharacteristicCurve } from '../models/mixed.js';
import { simulateMatrix } from '../simulation/respondent.js';
import { pad, padStart } from './format.js';
import { parseOptions, type OptionSpecs } from './options.js';

export const LINK_OPTIONS = {
  anchor: { kind: 'integer', fallback: 30, describe: 'Number of common items shared by both forms' },
  sample: { kind: 'integer', fallback: 2000, describe: 'Respondents per calibration sample' },
  shift: {
    kind: 'number',
    fallback: 0.8,
    describe: 'How much abler the second sample is, in logits',
  },
  spread: {
    kind: 'number',
    fallback: 1.3,
    describe: 'Ability spread of the second sample, relative to the first',
  },
  seed: { kind: 'integer', fallback: 20260101, describe: 'Seed for the whole run' },
} as const satisfies OptionSpecs;

/**
 * Calibrate one anchor on two different samples and link the two scales.
 *
 * The point the command is built to show is that the two calibrations are not
 * in disagreement about the items — they are the same items — but about where
 * to put the origin and how wide to make the unit. Nothing is wrong with either
 * scale until you try to compare a score on one with a score on the other.
 */
export function runLink(argv: readonly string[]): void {
  const options = parseOptions(argv, LINK_OPTIONS);
  if (options.anchor < 2) {
    throw new RangeError(`anchor must be at least 2 items, received ${options.anchor}`);
  }
  if (options.spread <= 0) {
    throw new RangeError(`spread must be positive, received ${options.spread}`);
  }

  const parameters = createRng(options.seed);
  // Dichotomous throughout: simulateMatrix generates 0/1 response data.
  const anchor: Item[] = Array.from({ length: options.anchor }, (_, index) =>
    makeItem(
      `anchor-${String(index).padStart(3, '0')}`,
      twoPL(0.8 + 0.9 * parameters.next(), -2 + 4 * parameters.next()),
    ),
  );

  const drawSample = (centre: number, spread: number, seed: number): number[] => {
    const rng = createRng(seed);
    return Array.from({ length: options.sample }, () => centre + spread * rng.nextNormal());
  };
  const reference = drawSample(0, 1, options.seed + 1);
  const second = drawSample(options.shift, options.spread, options.seed + 2);

  process.stdout.write(
    `Anchor: ${anchor.length} items, calibrated twice\n` +
      `Reference sample: ${options.sample} respondents, ability ~ N(0, 1)\n` +
      `Second sample:    ${options.sample} respondents, ability ~ ` +
      `N(${options.shift.toFixed(2)}, ${options.spread.toFixed(2)}^2)\n\n`,
  );

  const referenceFit = calibrate(simulateMatrix(anchor, reference, options.seed + 3), {
    model: '2pl',
  });
  const secondFit = calibrate(simulateMatrix(anchor, second, options.seed + 4), { model: '2pl' });
  if (!referenceFit.converged || !secondFit.converged) {
    process.stdout.write('warning: at least one calibration did not converge\n\n');
  }

  const referenceBank = toItems(referenceFit);
  const secondBank = toItems(secondFit);

  const referenceLocations = locationParameters(referenceBank);
  const secondLocations = locationParameters(secondBank);
  process.stdout.write(
    'Each calibration fixes the metric to its own sample, so the two disagree\n' +
      'about the anchor even though it is the same anchor:\n\n' +
      `  reference scale: mean difficulty ${mean(referenceLocations).toFixed(3)}, ` +
      `sd ${Math.sqrt(variance(referenceLocations)).toFixed(3)}\n` +
      `  second scale:    mean difficulty ${mean(secondLocations).toFixed(3)}, ` +
      `sd ${Math.sqrt(variance(secondLocations)).toFixed(3)}\n\n`,
  );

  // Link the second calibration onto the reference scale.
  const pairs = commonItems(secondBank, referenceBank);
  const results: LinkingResult[] = linkAll(pairs);

  const header =
    pad('method', 16) + padStart('A', 9) + padStart('B', 9) + padStart('criterion', 13) + '\n';
  process.stdout.write(header);
  process.stdout.write('-'.repeat(header.length + 2) + '\n');
  for (const result of results) {
    process.stdout.write(
      pad(result.method, 16) +
        padStart(result.transform.slope.toFixed(4), 9) +
        padStart(result.transform.intercept.toFixed(4), 9) +
        padStart(result.criterion === 0 ? '-' : result.criterion.toExponential(2), 13) +
        '\n',
    );
  }

  const chosen = results.find((result) => result.method === 'stocking-lord') as LinkingResult;
  const rule = normalGridRule(0, 1, 41);
  const before = stockingLordCriterion(pairs, IDENTITY_TRANSFORM, rule);
  const after = stockingLordCriterion(pairs, chosen.transform, rule);
  const linked = transformBank(chosen.transform, secondBank);

  // How far apart the two forms' characteristic curves are, in score points,
  // before and after linking — the practical consequence of the whole exercise.
  let worstBefore = 0;
  let worstAfter = 0;
  for (const theta of rule.nodes) {
    const referenceScore = testCharacteristicCurve(referenceBank, theta);
    worstBefore = Math.max(
      worstBefore,
      Math.abs(referenceScore - testCharacteristicCurve(secondBank, theta)),
    );
    worstAfter = Math.max(
      worstAfter,
      Math.abs(referenceScore - testCharacteristicCurve(linked, theta)),
    );
  }

  process.stdout.write(
    `\nTaking Stocking-Lord: a candidate at ability 1.00 on the second scale sits at ` +
      `${transformAbility(chosen.transform, 1).toFixed(3)} on the reference scale.\n` +
      `Test characteristic curves differed by up to ${worstBefore.toFixed(2)} score points ` +
      `before linking and ${worstAfter.toFixed(2)} after.\n` +
      `Stocking-Lord criterion fell from ${before.toExponential(2)} to ${after.toExponential(2)}.\n`,
  );
}
