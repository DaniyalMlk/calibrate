import {
  bankHealth,
  exposureChiSquare,
  exposureConcentration,
  exposureRatesFromIds,
  exposureVariance,
  overlapRate,
  precisionTarget,
  simulateSession,
  unusedFraction,
  type BankHealth,
  type ExposureConcentration,
} from '../../../src/index.js';
import { attachCrosshair, Plot } from '../chart/plot.js';
import { bandPath, crisp, html, polyline, svg } from '../chart/svg.js';
import { fixed, percent, signed } from '../format.js';
import { selectorFor, THETA_DOMAIN, type Store, type ViewModel } from '../state.js';

const COVERAGE = 'var(--series-1)';
const CONCENTRATION = 'var(--series-5)';
const EVEN = 'var(--de-emphasis)';

/** How many simulated candidates the exposure run administers. */
const CANDIDATES = 250;

interface ExposureRun {
  readonly candidates: number;
  readonly meanLength: number;
  readonly peak: number;
  readonly overlap: number;
  readonly unused: number;
  readonly chiSquare: number;
  readonly concentration: ExposureConcentration;
}

/**
 * Where the bank can measure, and which of its items do the work.
 *
 * Two questions a bank owner has to answer together. Coverage says which
 * abilities the bank could serve if it were fully available; exposure says how
 * much of the bank a testing programme actually reaches. A bank can look
 * healthy on the first and be trivially harvestable on the second.
 */
export function mountBank(root: HTMLElement, store: Store): void {
  const heading = html('div');
  heading.append(
    html('h2', { className: 'panel__title', text: 'Bank coverage and exposure' }),
    html('p', {
      className: 'panel__note',
      text:
        'The standard error the whole bank could reach at each ability, and how administration ' +
        'spreads across its items over a simulated population.',
    }),
  );

  const run = html('button', {
    className: 'button button--primary',
    text: `Run ${CANDIDATES} candidates`,
    attributes: { type: 'button' },
  });

  const head = html('div', { className: 'panel__head' });
  head.append(heading, run);

  const coverageLegend = html('div', { className: 'legend' });
  coverageLegend.append(
    legendItem('Standard error, whole bank', COVERAGE, 'line'),
    legendItem('Below target', COVERAGE, 'swatch'),
  );

  const coverageChart = html('div', { className: 'chart' });
  const coverageTip = html('div', { className: 'tooltip', attributes: { role: 'status' } });
  coverageTip.hidden = true;
  coverageChart.append(coverageTip);
  const coverageCaption = html('p', { className: 'chart__caption' });

  const exposureHeading = html('h3', {
    className: 'panel__title',
    text: 'Exposure concentration',
  });
  exposureHeading.style.fontSize = '15px';
  const exposureNote = html('p', { className: 'panel__note' });

  const exposureLegend = html('div', { className: 'legend' });
  exposureLegend.append(
    legendItem('Observed exposure', CONCENTRATION, 'line'),
    legendItem('Perfectly even', EVEN, 'line'),
  );

  const exposureChart = html('div', { className: 'chart' });
  const exposureTip = html('div', { className: 'tooltip', attributes: { role: 'status' } });
  exposureTip.hidden = true;
  exposureChart.append(exposureTip);

  const readout = html('div', { className: 'readout' });

  const exposureSection = html('div', { className: 'panel__body' });
  exposureSection.append(
    exposureHeading,
    exposureNote,
    exposureLegend,
    exposureChart,
    readout,
  );

  root.append(head, coverageLegend, coverageChart, coverageCaption, exposureSection);

  let width = 0;
  let model: ViewModel | null = null;
  let exposure: ExposureRun | null = null;

  run.addEventListener('click', () => {
    if (model === null) {
      return;
    }
    // Hold the previous render rather than flashing a skeleton, then compute.
    exposureSection.classList.add('is-stale');
    run.disabled = true;
    // Yield once so the browser paints the held state before the loop blocks.
    window.setTimeout(() => {
      if (model !== null) {
        exposure = administer(model);
      }
      exposureSection.classList.remove('is-stale');
      run.disabled = false;
      if (model !== null && width > 0) {
        draw(model);
      }
    }, 0);
  });

  const draw = (current: ViewModel): void => {
    drawCoverage(current);
    drawExposure();
  };

  const drawCoverage = (current: ViewModel): void => {
    const health = bankHealth(current.bank, {
      range: [THETA_DOMAIN[0], THETA_DOMAIN[1]],
      points: 121,
      target: current.configuration.target,
    });

    const thetas = health.points.map((point) => point.theta);
    const errors = health.points.map((point) => point.standardError);
    const finite = errors.filter((value) => Number.isFinite(value));
    const ceiling = Math.min(2.5, Math.max(...finite, current.configuration.target * 2) * 1.15);

    const plot = new Plot({
      width,
      height: 210,
      xDomain: THETA_DOMAIN,
      yDomain: [0, ceiling],
      yTicks: 4,
      xTickFormat: (value) => (value === 0 ? '0' : signed(value, 0)),
      xLabel: 'ability θ, logits',
    });

    // The region below the target: filled where the bank measures well enough,
    // which makes the gaps read as the absence of fill rather than as an
    // annotation the reader has to decode.
    const targetY = plot.y.to(current.configuration.target);
    const covered = thetas.map((theta, i) =>
      Number.isFinite(errors[i]) && (errors[i] as number) <= current.configuration.target
        ? plot.x.to(theta)
        : Number.NaN,
    );
    const runs: number[][] = [];
    let currentRun: number[] = [];
    for (const x of covered) {
      if (Number.isFinite(x)) {
        currentRun.push(x);
      } else if (currentRun.length > 0) {
        runs.push(currentRun);
        currentRun = [];
      }
    }
    if (currentRun.length > 0) {
      runs.push(currentRun);
    }
    for (const stretch of runs) {
      if (stretch.length < 2) {
        continue;
      }
      plot.dataLayer.append(
        svg('path', {
          d: bandPath(
            stretch,
            stretch.map(() => targetY),
            stretch.map(() => plot.plotBottom),
          ),
          fill: COVERAGE,
          'fill-opacity': 'var(--band-outer)',
          stroke: 'none',
        }),
      );
    }

    plot.dataLayer.append(
      svg('path', {
        d: polyline(
          thetas.map((theta, i) => {
            const value = errors[i] as number;
            // Off the top of the scale breaks the line; clamping would draw a
            // flat run that reads as a constant standard error.
            return [plot.x.to(theta), value > ceiling ? Number.NaN : plot.y.to(value)];
          }),
        ),
        fill: 'none',
        stroke: COVERAGE,
        'stroke-width': 2,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
      }),
    );

    plot.referenceY(
      current.configuration.target,
      `SE target ${fixed(current.configuration.target)}`,
      'target',
    );

    replaceSvg(coverageChart, plot.root);
    attachCrosshair(coverageChart, plot, coverageTip, thetas, (index) => ({
      head: `θ = ${signed(thetas[index] as number)}`,
      rows: [
        {
          key: 'standard error',
          colour: COVERAGE,
          value: fixed(errors[index] as number, 3),
        },
        {
          key: 'items nearby',
          colour: 'var(--text-muted)',
          value: String(health.points[index]?.itemsNearby ?? 0),
        },
      ],
    }));

    coverageCaption.textContent = describeCoverage(health, current);
  };

  const drawExposure = (): void => {
    if (exposure === null) {
      exposureNote.textContent =
        `Nothing administered yet beyond this session. Run ${CANDIDATES} simulated candidates ` +
        'through the current policy to see how administration spreads across the bank.';
      exposureLegend.hidden = true;
      exposureChart.hidden = true;
      readout.replaceChildren();
      return;
    }

    exposureLegend.hidden = false;
    exposureChart.hidden = false;
    const { concentration } = exposure;
    exposureNote.textContent =
      `A Lorenz curve over item exposure: the point at ${percent(0.5)} of the bank says what ` +
      'share of all administrations the least-used half absorbed. The diagonal is perfectly ' +
      'even use; the further the curve sags below it, the more the testing rests on a few items.';

    const plot = new Plot({
      width,
      height: 230,
      xDomain: [0, 1],
      yDomain: [0, 1],
      yTicks: 5,
      xTicks: 5,
      xTickFormat: (value) => percent(value),
      yTickFormat: (value) => percent(value),
      xLabel: 'share of the bank, least exposed first',
    });

    // The diagonal is the reference, drawn in de-emphasis grey rather than a
    // series colour: it is what even use would look like, not a second measured
    // series.
    plot.dataLayer.append(
      svg('line', {
        x1: crisp(plot.x.to(0)),
        y1: crisp(plot.y.to(0)),
        x2: crisp(plot.x.to(1)),
        y2: crisp(plot.y.to(1)),
        stroke: EVEN,
        'stroke-width': 1.25,
      }),
    );

    const points = concentration.curve;
    plot.dataLayer.append(
      svg('path', {
        d: polyline(
          points.map((point) => [plot.x.to(point.bankFraction), plot.y.to(point.exposureShare)]),
        ),
        fill: 'none',
        stroke: CONCENTRATION,
        'stroke-width': 2,
        'stroke-linejoin': 'round',
      }),
    );

    replaceSvg(exposureChart, plot.root);
    attachCrosshair(
      exposureChart,
      plot,
      exposureTip,
      points.map((point) => point.bankFraction),
      (index) => {
        const point = points[index] as (typeof points)[number];
        return {
          head: `${percent(point.bankFraction, 1)} of the bank`,
          rows: [
            {
              key: 'exposure share',
              colour: CONCENTRATION,
              value: percent(point.exposureShare, 1),
            },
            { key: 'if even', colour: EVEN, value: percent(point.bankFraction, 1) },
          ],
        };
      },
    );

    readout.replaceChildren(
      cell('Gini', fixed(concentration.gini, 3)),
      cell('Peak rate', fixed(exposure.peak, 3)),
      cell('Top decile', percent(concentration.topDecileShare)),
      cell('Overlap', fixed(exposure.overlap, 3)),
      cell('Unused', percent(exposure.unused)),
      cell('Mean length', fixed(exposure.meanLength, 1)),
      cell('Exposure χ²', fixed(exposure.chiSquare, 1)),
      cell('Candidates', String(exposure.candidates)),
    );
  };

  store.subscribe((next) => {
    const policyChanged = model !== null && model.configuration.policy !== next.configuration.policy;
    const bankChanged = model !== null && model.bank !== next.bank;
    model = next;
    if (policyChanged || bankChanged) {
      // The exposure run describes a policy and a bank. Keeping a run made
      // under one policy on screen while the control row says another would be
      // the one failure a reader cannot see.
      exposure = null;
    }
    if (width > 0) {
      draw(next);
    }
  });

  const observer = new ResizeObserver((entries) => {
    const measured = Math.floor(entries[0]?.contentRect.width ?? 0);
    if (measured <= 0 || Math.abs(measured - width) < 2) {
      return;
    }
    width = measured;
    if (model !== null) {
      draw(model);
    }
  });
  observer.observe(coverageChart);
}

/**
 * Administer a simulated population through the current policy.
 *
 * Each candidate is seeded from their index rather than from a shared stream,
 * so the population is identical whichever policy is selected and the
 * comparison between two runs is a comparison of the policies.
 */
function administer(model: ViewModel): ExposureRun {
  const { configuration } = model;
  const policy = {
    name: configuration.policy,
    selector: selectorFor(configuration.policy),
    stopping: precisionTarget(configuration.target, { minimum: 5, maximum: configuration.maximum }),
  };

  const abilities: number[] = [];
  for (let i = 0; i < CANDIDATES; i += 1) {
    // An even spread of quantiles rather than a random draw: it covers the
    // ability range at a fixed sample size instead of leaving the tails to luck.
    abilities.push(THETA_DOMAIN[0] + ((i + 0.5) / CANDIDATES) * (THETA_DOMAIN[1] - THETA_DOMAIN[0]));
  }

  const sessions = abilities.map(
    (theta, i) => simulateSession(policy, model.pool, theta, 90000 + i).itemIds,
  );
  const rates = exposureRatesFromIds(sessions);
  const bankSize = model.bank.length;
  const meanLength = sessions.reduce((sum, ids) => sum + ids.length, 0) / sessions.length;
  const variance = exposureVariance(bankSize, rates);

  let peak = 0;
  for (const rate of rates.values()) {
    peak = Math.max(peak, rate);
  }

  return {
    candidates: CANDIDATES,
    meanLength,
    peak,
    overlap: overlapRate(bankSize, meanLength, variance),
    unused: unusedFraction(
      model.bank,
      sessions.map((ids) => ids.map((id) => model.byId.get(id)).filter(isItem)),
    ),
    chiSquare: exposureChiSquare(bankSize, meanLength, rates),
    concentration: exposureConcentration(bankSize, rates),
  };
}

function isItem<T>(value: T | undefined): value is T {
  return value !== undefined;
}

function describeCoverage(health: BankHealth, model: ViewModel): string {
  const target = fixed(model.configuration.target);

  let coverage: string;
  if (health.gaps.length === 0) {
    coverage =
      `All ${health.items} items together measure the whole range from ` +
      `${signed(health.range[0], 0)} to ${signed(health.range[1], 0)} to the target of ` +
      `${target}. Best precision ${fixed(1 / Math.sqrt(health.peak), 3)} at the information peak.`;
  } else {
    const worst = health.gaps.reduce((a, b) =>
      b.worstStandardError > a.worstStandardError ? b : a,
    );
    coverage =
      `${percent(health.covered)} of the range reaches the target of ${target}. ` +
      `${health.gaps.length} ${health.gaps.length === 1 ? 'gap' : 'gaps'}; the worst runs from θ ` +
      `${signed(worst.from)} to ${signed(worst.to)}, where the whole bank can only reach a ` +
      `standard error of ${fixed(worst.worstStandardError, 2)}. The next items to write belong ` +
      `there.`;
  }

  return `${coverage} ${describeMix(health)}`;
}

/**
 * The format mix, as a sentence rather than a pair of counts.
 *
 * The share of the *score* is the fact worth stating, not the share of the
 * items. Three hundred items of which sixty are four-category rubrics is a bank
 * where a fifth of the questions carry nearly half the points, and a reader
 * told only "60 rubric items" has to do that arithmetic themselves before they
 * can see what the coverage above rests on.
 */
function describeMix(health: BankHealth): string {
  const { dichotomous, polytomous, maximumScore } = health.formats;
  if (polytomous === 0) {
    return `Every item is dichotomous, for ${maximumScore} points in all.`;
  }
  if (dichotomous === 0) {
    return `Every item is rubric-scored, for ${maximumScore} points across ${polytomous} items.`;
  }
  // Each dichotomous item is worth exactly one point, so the rest of the
  // maximum is what the rubrics carry.
  const rubricPoints = maximumScore - dichotomous;
  return (
    `${polytomous} of the ${health.items} items are rubric-scored: ` +
    `${percent(polytomous / health.items)} of the bank carrying ` +
    `${percent(rubricPoints / maximumScore)} of its ${maximumScore} points.`
  );
}

function replaceSvg(container: HTMLElement, next: SVGSVGElement): void {
  const existing = container.querySelector('svg');
  if (existing === null) {
    container.append(next);
  } else {
    existing.replaceWith(next);
  }
}

function cell(label: string, value: string): HTMLElement {
  const wrap = html('div');
  wrap.append(
    html('div', { className: 'readout__label', text: label }),
    html('div', { className: 'readout__value', text: value }),
  );
  return wrap;
}

function legendItem(label: string, colour: string, kind: 'line' | 'swatch'): HTMLElement {
  const item = html('span', { className: 'legend__item' });
  const mark = html('span', { className: kind === 'line' ? 'legend__line' : 'legend__swatch' });
  mark.style.background = colour;
  if (kind === 'swatch') {
    mark.style.opacity = '0.45';
  }
  item.append(mark, html('span', { text: label }));
  return item;
}
