import {
  categoryProbabilitiesOf,
  expectedScoreOf,
  informationOf,
  isPolytomous,
  itemLocation,
  linspace,
  maximumScoreOf,
  type AnyItem,
} from '../../../src/index.js';
import { attachCrosshair, Plot } from '../chart/plot.js';
import { haloText, html, polyline, svg } from '../chart/svg.js';
import { fixed, percent, signed } from '../format.js';
import { THETA_DOMAIN, type Store, type ViewModel } from '../state.js';

const INFORMATION = 'var(--series-2)';
const EXPECTED = 'var(--series-3)';

/** Ability values every curve in this panel is sampled at. */
const GRID = linspace(THETA_DOMAIN[0], THETA_DOMAIN[1], 201);

/** The ordinal ramp, in order. Index is the category, never the series slot. */
const RAMP = [
  'var(--ordinal-1)',
  'var(--ordinal-2)',
  'var(--ordinal-3)',
  'var(--ordinal-4)',
  'var(--ordinal-5)',
  'var(--ordinal-6)',
] as const;

/**
 * The colour for category `k` of an item with `count` categories.
 *
 * The ramp is walked end to end whatever the category count, so the darkest
 * step always means the top category. Assigning from the low end instead would
 * make a three-category item's top score wear the colour a six-category item
 * uses for its middle, and a reader comparing two items would read that as a
 * difference in the items.
 */
export function rampColour(k: number, count: number): string {
  if (count <= 1) return RAMP[RAMP.length - 1] as string;
  const position = (k / (count - 1)) * (RAMP.length - 1);
  return RAMP[Math.round(position)] as string;
}

/**
 * Which categories are the most likely outcome at some ability on the grid.
 *
 * The complement is the finding this panel exists to surface: a category that
 * is modal nowhere is a rubric level the model never expects to be anybody's
 * most likely score, which is what two thresholds collapsing onto each other
 * looks like from the outside. Such a level is doing no discriminating work and
 * should be merged with its neighbour.
 *
 * Ties go to the lower category, which is the conservative reading: a level
 * that is only ever tied for most likely is not distinguishing a band of
 * ability from the one below it.
 */
export function modalCategories(
  probabilities: readonly (readonly number[])[],
): ReadonlySet<number> {
  const modal = new Set<number>();
  for (const row of probabilities) {
    let best = 0;
    for (let k = 1; k < row.length; k += 1) {
      if ((row[k] as number) > (row[best] as number)) best = k;
    }
    modal.add(best);
  }
  return modal;
}

/**
 * What one item's response model actually looks like.
 *
 * A dichotomous item is described well enough by its parameters: one curve,
 * one difficulty, and a reader who knows the model can picture it from
 * `a = 1.2  b = +0.4`. A rubric item cannot be pictured that way.
 * `a = 2.49  b = [−0.36, +1.01, +1.56]` is the whole item, but nobody reads a
 * threshold vector and sees whether the middle categories are ever modal —
 * which is the question that decides whether the rubric is doing what its
 * author intended. Thresholds bunched together make an item behave almost
 * dichotomously; spread too far they leave bands where it distinguishes
 * nothing. Both are obvious in the curves and invisible in the numbers.
 *
 * A dichotomous item draws here too, as the two-category case rather than as a
 * refusal. Most items in a mixed bank are dichotomous, and a panel that blanked
 * on them would be empty most of the time.
 */
export function mountCategories(root: HTMLElement, store: Store): void {
  const heading = html('div');
  heading.append(
    html('h2', { className: 'panel__title', text: 'Category response curves' }),
    html('p', {
      className: 'panel__note',
      text:
        'For the item awaiting a response: the probability of each category across the ability ' +
        'range, the expected score as a share of the maximum, and what the item contributes to ' +
        'precision below. A category that is modal nowhere is a rubric level the model never ' +
        'expects to be the most likely outcome for anyone.',
    }),
  );

  const badge = html('span', { className: 'badge badge--neutral' });
  const head = html('div', { className: 'panel__head' });
  head.append(heading, badge);

  const legend = html('div', { className: 'legend' });

  const curveChart = html('div', { className: 'chart' });
  const curveTip = html('div', { className: 'tooltip', attributes: { role: 'status' } });
  curveTip.hidden = true;
  curveChart.append(curveTip);

  const lowerChart = html('div', { className: 'chart' });
  const lowerTip = html('div', { className: 'tooltip', attributes: { role: 'status' } });
  lowerTip.hidden = true;
  lowerChart.append(lowerTip);

  const caption = html('p', { className: 'chart__caption' });
  const charts = html('div');
  charts.append(curveChart, lowerChart);

  root.append(head, legend, charts, caption);

  let width = 0;
  let model: ViewModel | null = null;

  const draw = (current: ViewModel): void => {
    const item = current.current;
    if (item === null) {
      badge.textContent = 'No item pending';
      legend.replaceChildren();
      curveChart.replaceChildren(curveTip);
      lowerChart.replaceChildren(lowerTip);
      caption.textContent =
        'The session has stopped, so no item is awaiting a response. Start a new session to ' +
        'see the next item’s response model.';
      return;
    }

    const top = maximumScoreOf(item);
    const count = top + 1;
    badge.textContent = isPolytomous(item) ? `${count} categories` : 'dichotomous';

    legend.replaceChildren();
    for (let k = 0; k < count; k += 1) {
      legend.append(
        legendItem(
          count === 2 ? (k === 0 ? 'Incorrect' : 'Correct') : `${k} point${k === 1 ? '' : 's'}`,
          rampColour(k, count),
        ),
      );
    }
    legend.append(
      legendItem('Expected score, share of maximum', EXPECTED),
      legendItem('Item information', INFORMATION),
    );

    // Probability is a proportion: the axis runs 0 to 1 always, never fitted to
    // the curves. A category whose peak is 0.31 has to *look* like a category
    // that is never more likely than not, and an axis rescaled to its maximum
    // would draw it as a confident hump.
    const curvePlot = new Plot({
      width,
      height: 230,
      xDomain: THETA_DOMAIN,
      yDomain: [0, 1],
      yTicks: 5,
      yTickFormat: (value) => percent(value),
      xTickFormat: (value) => (value === 0 ? '0' : signed(value, 0)),
      hideXLabels: true,
      insets: { bottom: 20 },
    });

    const probabilities = GRID.map((theta) => categoryProbabilitiesOf(item, theta));

    for (let k = 0; k < count; k += 1) {
      const colour = rampColour(k, count);
      curvePlot.dataLayer.append(
        svg('path', {
          d: polyline(
            GRID.map((theta, i) => [
              curvePlot.x.to(theta),
              curvePlot.y.to((probabilities[i] as number[])[k] as number),
            ]),
          ),
          fill: 'none',
          stroke: colour,
          'stroke-width': 2,
          'stroke-linejoin': 'round',
          'stroke-linecap': 'round',
        }),
      );

      // A direct label at the curve's own maximum. These curves cross
      // constantly, so a legend alone would make the reader trace a line back
      // to a swatch; the peak is both the least crowded point on each curve and
      // where the eye already is.
      let peak = 0;
      for (let i = 1; i < GRID.length; i += 1) {
        if (
          ((probabilities[i] as number[])[k] as number) >
          ((probabilities[peak] as number[])[k] as number)
        ) {
          peak = i;
        }
      }
      const height = (probabilities[peak] as number[])[k] as number;
      // Below a tenth the peak is a shoulder rather than a hump, and a label
      // pinned to it lands on top of whichever curve is actually dominant
      // there. The legend carries those categories instead.
      if (height >= 0.1) {
        // The end categories saturate at 1, so their peak sits on the top
        // gridline and a label placed above it is drawn outside the viewBox
        // and clipped. Flip below the curve when there is no room above.
        const above = curvePlot.y.to(height) - 8;
        const clear = above >= curvePlot.plotTop + 4;
        curvePlot.overlay.append(
          haloText(
            curvePlot.x.to(GRID[peak] as number),
            clear ? above : curvePlot.y.to(height) + 15,
            count === 2 ? (k === 0 ? '✗' : '✓') : String(k),
            { 'text-anchor': 'middle', fill: colour, 'font-weight': 600, 'font-size': 12 },
          ),
        );
      }
    }

    curvePlot.referenceX(
      current.snapshot.theta,
      `current θ ${signed(current.snapshot.theta)}`,
      'measured',
    );

    // The expected score belongs on the probability axis above, not on the
    // information axis below.
    //
    // Drawing it under the information curve was the first attempt, and it was
    // the dual-axis mistake wearing a disguise: two quantities in different
    // units sharing one vertical scale, where only one of them is in the units
    // the axis is labelled with. A reader would have read the dashed line's
    // height off an axis that does not describe it.
    //
    // As a fraction of the item's maximum, the expected score *is* a
    // proportion, so it belongs on the 0-to-100% axis honestly — and on a
    // dichotomous item it coincides exactly with the probability of a correct
    // answer, which is the item characteristic curve the reader already knows.
    const expected = GRID.map((theta) => expectedScoreOf(item, theta));
    curvePlot.dataLayer.append(
      svg('path', {
        d: polyline(
          GRID.map((theta, i) => [
            curvePlot.x.to(theta),
            curvePlot.y.to((expected[i] as number) / top),
          ]),
        ),
        fill: 'none',
        stroke: EXPECTED,
        'stroke-width': 2,
        'stroke-dasharray': '5 3',
        'stroke-linejoin': 'round',
      }),
    );

    // Below: what the item contributes to precision, alone on an axis in its
    // own units.
    const information = GRID.map((theta) => informationOf(item, theta));
    const infoCeiling = Math.max(...information, 0.25) * 1.15;

    const lowerPlot = new Plot({
      width,
      height: 170,
      xDomain: THETA_DOMAIN,
      yDomain: [0, infoCeiling],
      yTicks: 4,
      xTickFormat: (value) => (value === 0 ? '0' : signed(value, 0)),
      xLabel: 'ability θ, logits',
    });

    lowerPlot.dataLayer.append(
      svg('path', {
        d: polyline(
          GRID.map((theta, i) => [
            lowerPlot.x.to(theta),
            lowerPlot.y.to(information[i] as number),
          ]),
        ),
        fill: 'none',
        stroke: INFORMATION,
        'stroke-width': 2,
        'stroke-linejoin': 'round',
      }),
    );

    replaceSvg(curveChart, curvePlot.root);
    replaceSvg(lowerChart, lowerPlot.root);

    attachCrosshair(curveChart, curvePlot, curveTip, GRID, (index) => ({
      head: `θ = ${signed(GRID[index] as number)}`,
      rows: [
        ...Array.from({ length: count }, (_, k) => ({
          key: count === 2 ? (k === 0 ? 'incorrect' : 'correct') : `${k} of ${top}`,
          colour: rampColour(k, count),
          value: percent((probabilities[index] as number[])[k] as number, 1),
        })),
        {
          key: 'expected score',
          colour: EXPECTED,
          value: `${fixed(expected[index] as number, 2)} of ${top}`,
        },
      ],
    }));

    attachCrosshair(lowerChart, lowerPlot, lowerTip, GRID, (index) => ({
      head: `θ = ${signed(GRID[index] as number)}`,
      rows: [
        {
          key: 'item information',
          colour: INFORMATION,
          value: fixed(information[index] as number, 3),
        },
      ],
    }));

    caption.textContent = describe(item, probabilities, information);
  };

  store.subscribe((next) => {
    model = next;
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
  observer.observe(curveChart);
}

/**
 * The one sentence the curves are for.
 *
 * A category that is modal nowhere is the finding: it is a level on the rubric
 * that the model never expects to be the most likely outcome for anybody, which
 * usually means two thresholds have collapsed onto each other and the level
 * should be merged with its neighbour. Naming it in words means a reader gets
 * the finding without having to trace five curves to notice an absence.
 */
function describe(
  item: AnyItem,
  probabilities: readonly (readonly number[])[],
  information: readonly number[],
): string {
  const count = probabilities[0]?.length ?? 2;
  const peakIndex = information.reduce(
    (best, value, i) => (value > (information[best] as number) ? i : best),
    0,
  );
  const peakTheta = GRID[peakIndex] as number;
  const where = `Most informative at θ ${signed(peakTheta)}, where it contributes ${fixed(
    information[peakIndex] as number,
    2,
  )}.`;

  if (!isPolytomous(item)) {
    return (
      `A dichotomous item: two categories, crossing at its difficulty of ` +
      `${signed(itemLocation(item))}. ${where}`
    );
  }

  const modal = modalCategories(probabilities);
  const never = Array.from({ length: count }, (_, k) => k).filter((k) => !modal.has(k));
  if (never.length === 0) {
    return (
      `Every one of the ${count} categories is the most likely outcome somewhere on the ` +
      `scale, so each rubric level separates a band of ability from its neighbours. ${where}`
    );
  }
  return (
    `${never.length === 1 ? 'Category' : 'Categories'} ${never.join(', ')} ` +
    `${never.length === 1 ? 'is' : 'are'} modal nowhere: no ability makes ` +
    `${never.length === 1 ? 'that level' : 'those levels'} the most likely score, which is ` +
    `what adjacent thresholds collapsing onto each other looks like. ${where}`
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

function legendItem(label: string, colour: string): HTMLElement {
  const item = html('span', { className: 'legend__item' });
  const mark = html('span', { className: 'legend__line' });
  mark.style.background = colour;
  item.append(mark, html('span', { text: label }));
  return item;
}
