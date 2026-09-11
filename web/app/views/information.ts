import {
  isPolytomous,
  itemLocation,
  linspace,
  standardError,
  testInformationOf,
  type AnyItem,
} from '../../../src/index.js';
import { attachCrosshair, Plot } from '../chart/plot.js';
import { crisp, html, polyline, svg } from '../chart/svg.js';
import { fixed, signed } from '../format.js';
import { THETA_DOMAIN, type Store, type ViewModel } from '../state.js';

const INFORMATION = 'var(--series-2)';
const ERROR_SERIES = 'var(--series-1)';
const CONTEXT = 'var(--de-emphasis)';
/**
 * The format channel, held to one hue across every panel that marks a
 * rubric-scored item. Format is one fact, so it gets one colour; a panel that
 * chose its own would make the reader learn the encoding twice.
 */
export const RUBRIC = 'var(--series-4)';

/** Ability values the curves are sampled at. Shared by both panels. */
const GRID = linspace(THETA_DOMAIN[0], THETA_DOMAIN[1], 161);

/**
 * Test information and the standard error of measurement.
 *
 * Two stacked plots on one shared ability axis, not one plot with two vertical
 * axes. Information and standard error are reciprocally related — the error is
 * one over the square root of the information — so drawing them against two
 * independently scaled axes produces a crossing point that is an artefact of
 * the two ranges chosen and means nothing at all. Stacking them keeps every
 * comparison horizontal, which is the only direction in which these two
 * quantities are comparable.
 */
export function mountInformation(root: HTMLElement, store: Store): void {
  const heading = html('div');
  heading.append(
    html('h2', { className: 'panel__title', text: 'Information and precision' }),
    html('p', {
      className: 'panel__note',
      text:
        'How much the items administered so far can tell you, and at which abilities. The ' +
        'ticks along the baseline mark where each item sits, which is what gives the sum its ' +
        'shape \u2014 taller ticks are rubric-scored items, which carry several times the ' +
        'information of a single dichotomous one. The standard error below is the ' +
        'information\u2019s reciprocal square root.',
    }),
  );

  const tableToggle = html('button', {
    className: 'button button--ghost',
    text: 'Table',
    attributes: { type: 'button', 'aria-pressed': 'false' },
  });

  const head = html('div', { className: 'panel__head' });
  head.append(heading, tableToggle);

  const legend = html('div', { className: 'legend' });
  legend.append(
    legendItem('Test information', INFORMATION),
    legendItem('Standard error', ERROR_SERIES),
    legendItem('Item location', CONTEXT),
    legendItem('Rubric item', RUBRIC),
  );

  const charts = html('div');
  const infoChart = html('div', { className: 'chart' });
  const errorChart = html('div', { className: 'chart' });
  const infoTip = html('div', { className: 'tooltip', attributes: { role: 'status' } });
  const errorTip = html('div', { className: 'tooltip', attributes: { role: 'status' } });
  infoTip.hidden = true;
  errorTip.hidden = true;
  infoChart.append(infoTip);
  errorChart.append(errorTip);
  charts.append(infoChart, errorChart);

  const caption = html('p', { className: 'chart__caption' });
  const tableHost = html('div', { className: 'table-wrap' });
  tableHost.hidden = true;

  let showTable = false;
  tableToggle.addEventListener('click', () => {
    showTable = !showTable;
    tableToggle.setAttribute('aria-pressed', String(showTable));
    tableToggle.textContent = showTable ? 'Chart' : 'Table';
    charts.hidden = showTable;
    tableHost.hidden = !showTable;
  });

  root.append(head, legend, charts, caption, tableHost);

  let width = 0;
  let model: ViewModel | null = null;

  const draw = (current: ViewModel): void => {
    const administered = itemsAdministered(current);
    const information = GRID.map((theta) => testInformationOf(administered, theta));
    const errors = information.map((value) => (value > 0 ? standardError(value) : Number.NaN));

    // Information starts at zero. A magnitude on a truncated baseline overstates
    // every difference on the plot, and here the value at zero is meaningful:
    // it is what the test cannot measure at all.
    const peak = Math.max(...information, 0.5);
    const infoPlot = new Plot({
      width,
      height: 210,
      xDomain: THETA_DOMAIN,
      yDomain: [0, peak * 1.12],
      yTicks: 4,
      xTickFormat: (value) => (value === 0 ? '0' : signed(value, 0)),
      hideXLabels: true,
      insets: { bottom: 20 },
    });

    if (administered.length === 0) {
      infoPlot.dimFrame(0.4);
    }

    // A rug of item locations along the baseline, rather than each item's own
    // information curve.
    //
    // The curves were the first attempt and they carry nothing: twenty-five
    // items peaking near 0.7 each, summed to a test information of 11, are a
    // flat grey smudge along the axis once both are on the scale the sum needs.
    // Where the items sit is the fact that explains the shape of the sum, and a
    // tick per item at its location says it in one channel, at full contrast,
    // without competing with the curve it explains.
    //
    // A rubric item gets a taller tick at the mean of its thresholds. Its
    // location is a summary of several thresholds rather than a single
    // difficulty, and it contributes several times the information of a
    // dichotomous item at that point, so a tick of the same height would
    // under-report it — but the height is the only thing that differs, because
    // position on this axis means the same thing for both.
    for (const item of administered) {
      const x = crisp(infoPlot.x.to(itemLocation(item)));
      if (x < infoPlot.plotLeft || x > infoPlot.plotRight) {
        continue;
      }
      const rubric = isPolytomous(item);
      infoPlot.dataLayer.append(
        svg('line', {
          x1: x,
          x2: x,
          y1: infoPlot.plotBottom,
          y2: infoPlot.plotBottom - (rubric ? 16 : 9),
          stroke: rubric ? RUBRIC : CONTEXT,
          'stroke-width': rubric ? 2 : 1.5,
          'stroke-linecap': 'round',
          opacity: rubric ? 0.95 : 0.85,
        }),
      );
    }

    infoPlot.dataLayer.append(
      svg('path', {
        d: polyline(
          GRID.map((theta, i) => [infoPlot.x.to(theta), infoPlot.y.to(information[i] as number)]),
        ),
        fill: 'none',
        stroke: INFORMATION,
        'stroke-width': 2,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
      }),
    );

    if (administered.length > 0) {
      infoPlot.referenceX(
        current.snapshot.theta,
        `current θ ${signed(current.snapshot.theta)}`,
        'measured',
      );
    }

    // The error panel shares the ability domain and the x scale exactly.
    const finite = errors.filter((value) => Number.isFinite(value));
    const errorCeiling = Math.min(2.5, Math.max(...finite, 1) * 1.1);
    const errorPlot = new Plot({
      width,
      height: 190,
      xDomain: THETA_DOMAIN,
      yDomain: [0, errorCeiling],
      yTicks: 4,
      xTickFormat: (value) => (value === 0 ? '0' : signed(value, 0)),
      xLabel: 'ability θ, logits',
    });

    if (administered.length === 0) {
      errorPlot.dimFrame(0.4);
    }

    // Values above the ceiling break the line rather than being clamped to it.
    // Clamping drew a flat run along the top of the panel, which reads as "the
    // standard error is constant out here" when it means "it is off the scale".
    errorPlot.dataLayer.append(
      svg('path', {
        d: polyline(
          GRID.map((theta, i) => {
            const value = errors[i] as number;
            return [
              errorPlot.x.to(theta),
              value > errorCeiling ? Number.NaN : errorPlot.y.to(value),
            ];
          }),
        ),
        fill: 'none',
        stroke: ERROR_SERIES,
        'stroke-width': 2,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
      }),
    );

    errorPlot.referenceY(
      current.configuration.target,
      `SE target ${fixed(current.configuration.target)}`,
      'target',
    );

    replaceSvg(infoChart, infoPlot.root);
    replaceSvg(errorChart, errorPlot.root);

    attachCrosshair(infoChart, infoPlot, infoTip, GRID, (index) => ({
      head: `θ = ${signed(GRID[index] as number)}`,
      rows: [
        {
          key: 'test information',
          colour: INFORMATION,
          value: fixed(information[index] as number, 3),
        },
        {
          key: 'standard error',
          colour: ERROR_SERIES,
          value: fixed(errors[index] as number, 3),
        },
      ],
    }));
    attachCrosshair(errorChart, errorPlot, errorTip, GRID, (index) => ({
      head: `θ = ${signed(GRID[index] as number)}`,
      rows: [
        {
          key: 'standard error',
          colour: ERROR_SERIES,
          value: fixed(errors[index] as number, 3),
        },
        {
          key: 'test information',
          colour: INFORMATION,
          value: fixed(information[index] as number, 3),
        },
      ],
    }));

    // Where the administered items actually measure to target: the interval of
    // abilities whose standard error clears the session's own threshold.
    const withinTarget = GRID.filter(
      (_, i) => Number.isFinite(errors[i]) && (errors[i] as number) <= current.configuration.target,
    );
    caption.textContent =
      administered.length === 0
        ? 'No items administered yet. Both panels show the coordinate space the curves will fill.'
        : withinTarget.length === 0
          ? `Peak information ${fixed(peak, 2)} at θ ${signed(
              GRID[information.indexOf(peak)] ?? 0,
            )}. No ability is yet measured to the target of ${fixed(
              current.configuration.target,
            )}; the best standard error anywhere is ${fixed(Math.min(...finite), 3)}.`
          : `Peak information ${fixed(peak, 2)}. These ${administered.length} items measure to ` +
            `the target of ${fixed(current.configuration.target)} between θ ${signed(
              withinTarget[0] as number,
            )} and ${signed(withinTarget.at(-1) as number)}.`;

    tableHost.replaceChildren(informationTable(information, errors, current));
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
  observer.observe(infoChart);
}

function replaceSvg(container: HTMLElement, next: SVGSVGElement): void {
  const existing = container.querySelector('svg');
  if (existing === null) {
    container.append(next);
  } else {
    existing.replaceWith(next);
  }
}

function itemsAdministered(model: ViewModel): readonly AnyItem[] {
  return model.responses.map((response) => response.item);
}

function legendItem(label: string, colour: string): HTMLElement {
  const item = html('span', { className: 'legend__item' });
  const mark = html('span', { className: 'legend__line' });
  mark.style.background = colour;
  item.append(mark, html('span', { text: label }));
  return item;
}

/**
 * The table twin, sampled at whole and half logits rather than at all 161 grid
 * points. A reader checking a number wants the row for θ = 1, not the row for
 * θ = 0.95.
 */
function informationTable(
  information: readonly number[],
  errors: readonly number[],
  model: ViewModel,
): HTMLElement {
  const table = html('table');
  const head = html('thead');
  const headRow = html('tr');
  headRow.append(
    html('th', { className: 'num', text: 'θ' }),
    html('th', { className: 'num', text: 'information' }),
    html('th', { className: 'num', text: 'SE' }),
    html('th', { text: 'meets target' }),
  );
  head.append(headRow);

  const body = html('tbody');
  for (let theta = THETA_DOMAIN[0]; theta <= THETA_DOMAIN[1] + 1e-9; theta += 0.5) {
    let nearest = 0;
    for (let i = 1; i < GRID.length; i += 1) {
      if (Math.abs((GRID[i] as number) - theta) < Math.abs((GRID[nearest] as number) - theta)) {
        nearest = i;
      }
    }
    const error = errors[nearest] as number;
    const meets = Number.isFinite(error) && error <= model.configuration.target;
    const row = html('tr');
    row.append(
      html('td', { className: 'num', text: signed(theta, 1) }),
      html('td', { className: 'num', text: fixed(information[nearest] as number, 3) }),
      html('td', { className: 'num', text: fixed(error, 3) }),
      html('td', { text: model.responses.length === 0 ? '—' : meets ? 'yes' : 'no' }),
    );
    body.append(row);
  }

  table.append(head, body);
  return table;
}
