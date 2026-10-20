import { normalUpperTail } from '../core/gamma.js';
import { mean, variance } from '../core/numeric.js';
import { estimateMle } from '../estimation/mle.js';
import { personFit, scanPersons } from '../estimation/person.js';
import { syntheticBank } from '../simulation/bank.js';
import {
  detectionRates,
  simulateSitting,
  type Aberrance,
  type DetectionRates,
} from '../simulation/aberrance.js';
import { pad, padStart } from './format.js';
import { parseOptions, type OptionSpecs } from './options.js';

export const MISFIT_OPTIONS = {
  candidates: { kind: 'integer', fallback: 1500, describe: 'Candidates in the simulated sitting' },
  length: { kind: 'integer', fallback: 40, describe: 'Items on the form' },
  share: {
    kind: 'number',
    fallback: 0.2,
    describe: 'Share of candidates who depart from the model',
  },
  extent: { kind: 'number', fallback: 0.2, describe: 'Share of the form each departure covers' },
  alpha: { kind: 'number', fallback: 0.05, describe: 'Nominal one-sided rejection rate' },
  seed: { kind: 'integer', fallback: 20260101, describe: 'Seed for the whole run' },
} as const satisfies OptionSpecs;

interface Row {
  readonly label: string;
  readonly kind: Aberrance;
  /** Sort the form from easiest to hardest before applying the departure. */
  readonly ordered?: boolean;
}

/**
 * The departures to report, and one of them twice.
 *
 * Guessing runs to the end of the form, so how detectable it is depends on
 * what is at the end — which is a property of the form, not of the candidate.
 * A bank in no particular order puts the guessed items at an average
 * difficulty; a form ordered easiest-to-hardest puts them on the items the
 * candidate was least likely to get right anyway, where one answer in four is
 * close to what the model already expects. Reporting both is the difference
 * between asserting that and showing it.
 */
const ROWS: readonly Row[] = [
  { label: 'none', kind: 'none' },
  { label: 'preknowledge', kind: 'preknowledge' },
  { label: 'careless', kind: 'careless' },
  { label: 'guessing', kind: 'guessing' },
  { label: 'guessing, ordered', kind: 'guessing', ordered: true },
];

/**
 * The one-sided normal critical value for a nominal rate.
 *
 * By bisection on `normalUpperTail`, which the DIF code already needed and
 * which is accurate to the regularized incomplete gamma it is built on. The
 * tail is monotone, so bisection is enough and there is no second
 * approximation of the normal distribution anywhere in the package.
 */
function criticalValue(alpha: number): number {
  let low = -8;
  let high = 0;
  for (let step = 0; step < 200; step += 1) {
    const middle = (low + high) / 2;
    if (normalUpperTail(-middle) < alpha) low = middle;
    else high = middle;
  }
  return (low + high) / 2;
}

function percent(value: number): string {
  return Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : '--';
}

/**
 * Scan a simulated sitting for candidates who did not answer as the model says.
 *
 * Runs each departure separately rather than mixing them into one sitting,
 * because the detection rate is a property of where on the form the misfit
 * sits, and a single figure over a mixture would hide that.
 */
export function runMisfit(argv: readonly string[]): void {
  const options = parseOptions(argv, MISFIT_OPTIONS);
  if (!(options.share >= 0 && options.share <= 1)) {
    throw new Error('option "--share" must be between 0 and 1');
  }
  if (!(options.extent > 0 && options.extent <= 1)) {
    throw new Error('option "--extent" must be greater than 0 and at most 1');
  }
  if (!(options.alpha > 0 && options.alpha < 0.5)) {
    throw new Error('option "--alpha" must be between 0 and 0.5');
  }

  const form = syntheticBank({ size: options.length, seed: options.seed, guessing: 0 });
  const threshold = criticalValue(options.alpha);

  process.stdout.write(
    `Form: ${form.length} items. Sitting: ${options.candidates} candidates, ` +
      `${percent(options.share)} of whom depart from the model over ` +
      `${percent(options.extent)} of the form.\n` +
      `Flagging lz* below ${threshold.toFixed(4)}, the one-sided ` +
      `${percent(options.alpha)} point.\n\n`,
  );

  process.stdout.write(
    `  ${pad('departure', 19)}${padStart('found', 8)}${padStart('false+', 9)}` +
      `${padStart('mean lz', 10)}${padStart('sd lz', 8)}${padStart('mean lz*', 10)}` +
      `${padStart('sd lz*', 8)}\n`,
  );

  for (const [index, row] of ROWS.entries()) {
    const { kind } = row;
    const share = kind === 'none' ? 0 : options.share;
    const sitting = simulateSitting({
      form: row.ordered === true
        ? [...form].sort((left, right) => left.parameters.b - right.parameters.b)
        : form,
      candidates: options.candidates,
      aberrance: { kind, extent: options.extent },
      share,
      seed: options.seed + index,
    });

    const scored = sitting.candidates.map((candidate) => {
      const theta = estimateMle(candidate.responses).theta;
      return { id: candidate.id, responses: candidate.responses, theta };
    });

    // The spreads are reported over the candidates nothing was done to, since
    // that is where the null distribution is supposed to hold.
    const honest = scored.filter((candidate) => !sitting.affected.has(candidate.id));
    const plain: number[] = [];
    const corrected: number[] = [];
    for (const candidate of honest) {
      const fit = personFit(candidate.responses, candidate.theta);
      plain.push(fit.lz);
      if (fit.lzStar !== null) corrected.push(fit.lzStar);
    }

    const flagged = scanPersons(scored, { threshold });
    const rates: DetectionRates = detectionRates(
      sitting,
      flagged.map((row) => row.id),
    );

    process.stdout.write(
      `  ${pad(row.label, 19)}${padStart(kind === 'none' ? '--' : percent(rates.sensitivity), 8)}` +
        `${padStart(percent(rates.falsePositiveRate), 9)}` +
        `${padStart(mean(plain).toFixed(3), 10)}` +
        `${padStart(Math.sqrt(variance(plain)).toFixed(3), 8)}` +
        `${padStart(mean(corrected).toFixed(3), 10)}` +
        `${padStart(Math.sqrt(variance(corrected)).toFixed(3), 8)}\n`,
    );
  }

  process.stdout.write(
    '\nThe `none` row is the calibration check: nothing was done to anyone, so the\n' +
      'false positive column should sit near the nominal rate, and the two spreads\n' +
      'show what the correction is for — lz comes out narrower than one because the\n' +
      'ability was estimated from the same responses being judged.\n\n' +
      'The two guessing rows are the same behaviour on the same bank, differing only\n' +
      'in the order the items are administered in. Guessing runs to the end of the\n' +
      'form, so what it costs depends on what is at the end: in a bank in no\n' +
      'particular order the guessed items sit at an average difficulty and the\n' +
      'residuals are large, while a form ordered easiest-to-hardest puts them where\n' +
      'the candidate was unlikely to succeed anyway, so one answer in four is close\n' +
      'to what the model already expects and there is little left for a residual to\n' +
      'find. The detection rate is a property of the form as much as the statistic.\n',
  );
}
