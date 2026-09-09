import { credibleInterval, type Posterior } from '../../../src/index.js';
import { attachCrosshair, Plot } from '../chart/plot.js';
import { bandPath, html, polyline, svg } from '../chart/svg.js';
import { fixed, interval, percent, signed } from '../format.js';
import { THETA_DOMAIN, type Store, type ViewModel } from '../state.js';

const SERIES = 'var(--series-1)';
const HEIGHT = 260;

/** Duration of the redraw tween. Under 300ms, per the interaction budget. */
const TWEEN_MS = 200;

function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Exponential ease-out: cubic-bezier(0.32, 0.72, 0, 1), close enough for a tween. */
function easeOut(t: number): number {
  return 1 - (1 - t) ** 3;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

interface Frame {
  readonly density: readonly number[];
  readonly peak: number;
  readonly bounds: {
    readonly inner: readonly [number, number];
    readonly middle: readonly [number, number];
    readonly outer: readonly [number, number];
  };
  readonly mean: number;
}

function frameOf(model: ViewModel): Frame {
  const { posterior, intervals } = model;
  return {
    density: posterior.density,
    peak: Math.max(...posterior.density),
    bounds: {
      inner: [intervals.inner.lower, intervals.inner.upper],
      middle: [intervals.middle.lower, intervals.middle.upper],
      outer: [intervals.outer.lower, intervals.outer.upper],
    },
    mean: posterior.mean,
  };
}

function blend(from: Frame, to: Frame, t: number): Frame {
  const density = to.density.map((value, i) => lerp((from.density[i] as number) ?? value, value, t));
  const pair = (
    a: readonly [number, number],
    b: readonly [number, number],
  ): readonly [number, number] => [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];

  return {
    density,
    peak: lerp(from.peak, to.peak, t),
    bounds: {
      inner: pair(from.bounds.inner, to.bounds.inner),
      middle: pair(from.bounds.middle, to.bounds.middle),
      outer: pair(from.bounds.outer, to.bounds.outer),
    },
    mean: lerp(from.mean, to.mean, t),
  };
}

/**
 * The area under a density curve between two ability values.
 *
 * The endpoints are interpolated onto exactly `lo` and `hi` rather than snapped
 * to the nearest grid point, so a band edge sits where the credible interval
 * actually is. Snapping would make the drawn band disagree with the interval
 * printed beneath it by up to one grid step, which on a 321-point grid is 0.025
 * logits — small, and exactly the kind of small disagreement that makes a
 * reader stop trusting the rest of the numbers.
 */
function areaBetween(
  grid: readonly number[],
  density: readonly number[],
  lo: number,
  hi: number,
  toX: (theta: number) => number,
  toY: (value: number) => number,
  baseline: number,
): string {
  const at = (theta: number): number => {
    if (theta <= (grid[0] as number)) {
      return density[0] as number;
    }
    if (theta >= (grid[grid.length - 1] as number)) {
      return density[density.length - 1] as number;
    }
    let hiIndex = 1;
    while (hiIndex < grid.length - 1 && (grid[hiIndex] as number) < theta) {
      hiIndex += 1;
    }
    const loIndex = hiIndex - 1;
    const span = (grid[hiIndex] as number) - (grid[loIndex] as number);
    const weight = span <= 0 ? 0 : (theta - (grid[loIndex] as number)) / span;
    return lerp(density[loIndex] as number, density[hiIndex] as number, weight);
  };

  const xs: number[] = [toX(lo)];
  const tops: number[] = [toY(at(lo))];
  for (const [i, theta] of grid.entries()) {
    if (theta > lo && theta < hi) {
      xs.push(toX(theta));
      tops.push(toY(density[i] as number));
    }
  }
  xs.push(toX(hi));
  tops.push(toY(at(hi)));

  return bandPath(xs, tops, xs.map(() => baseline));
}

export function mountPosterior(root: HTMLElement, store: Store): void {
  const heading = html('div');
  heading.append(
    html('h2', { className: 'panel__title', text: 'Ability posterior' }),
    html('p', {
      className: 'panel__note',
      text:
        'The full posterior over ability, not just its mean. Shaded from the inside out: the ' +
        'central 50, 80 and 95 percent credible intervals, each an equal-tailed pair of ' +
        'posterior quantiles.',
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
    legendItem('Posterior density', SERIES, 'line'),
    legendItem('50 / 80 / 95% credible', SERIES, 'swatch'),
    legendItem('Simulated true ability', 'var(--text-muted)', 'line'),
  );

  const chart = html('div', { className: 'chart' });
  const tooltip = html('div', { className: 'tooltip', attributes: { role: 'status' } });
  tooltip.hidden = true;
  chart.append(tooltip);

  const caption = html('p', { className: 'chart__caption' });
  const tableHost = html('div', { className: 'table-wrap' });
  tableHost.hidden = true;

  let showTable = false;
  tableToggle.addEventListener('click', () => {
    showTable = !showTable;
    tableToggle.setAttribute('aria-pressed', String(showTable));
    tableToggle.textContent = showTable ? 'Chart' : 'Table';
    chart.hidden = showTable;
    tableHost.hidden = !showTable;
  });

  root.append(head, legend, chart, caption, tableHost);

  let current: Frame | null = null;
  let animation = 0;
  let width = 0;

  const draw = (model: ViewModel, frame: Frame): void => {
    const grid = model.posterior.grid;
    const yMax = Math.max(frame.peak * 1.14, 0.12);
    const plot = new Plot({
      width,
      height: HEIGHT,
      xDomain: THETA_DOMAIN,
      yDomain: [0, yMax],
      hideYAxis: true,
      xTickFormat: (value) => (value === 0 ? '0' : signed(value, 0)),
      xLabel: 'ability θ, logits',
      insets: { bottom: 40 },
    });

    const toX = (theta: number): number => plot.x.to(theta);
    const toY = (value: number): number => plot.y.to(value);
    const baseline = plot.plotBottom;

    // There is no empty state to draw here: with no responses the posterior is
    // the prior, which is a real curve carrying real information — what the
    // engine believes before it has asked anything. An overlay explaining that
    // the plot is empty would be sitting on top of the answer.
    //
    // Nested same-hue fills. The alpha adds where they overlap, which produces
    // the ramp without picking three separate colours for one quantity.
    for (const [bounds, opacity] of [
      [frame.bounds.outer, 'var(--band-outer)'],
      [frame.bounds.middle, 'var(--band-middle)'],
      [frame.bounds.inner, 'var(--band-inner)'],
    ] as const) {
      const d = areaBetween(grid, frame.density, bounds[0], bounds[1], toX, toY, baseline);
      if (d !== '') {
        plot.dataLayer.append(
          svg('path', { d, fill: SERIES, 'fill-opacity': opacity, stroke: 'none' }),
        );
      }
    }

    plot.dataLayer.append(
      svg('path', {
        d: polyline(grid.map((theta, i) => [toX(theta), toY(frame.density[i] as number)])),
        fill: 'none',
        stroke: SERIES,
        'stroke-width': 2,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
      }),
    );

    // The point estimate: a marker with a surface-coloured ring, which
    // separates it from the band without adding a stroke of data-weight ink.
    const markerX = toX(frame.mean);
    if (markerX >= plot.plotLeft && markerX <= plot.plotRight) {
      plot.overlay.append(
        svg('line', {
          x1: markerX,
          x2: markerX,
          y1: baseline,
          y2: toY(Math.max(...frame.density)),
          stroke: SERIES,
          'stroke-width': 1,
          opacity: 0.5,
        }),
        svg('circle', {
          cx: markerX,
          cy: baseline,
          r: 4.5,
          fill: SERIES,
          stroke: 'var(--surface)',
          'stroke-width': 2,
        }),
      );
    }

    plot.referenceX(
      model.configuration.trueTheta,
      `true θ ${signed(model.configuration.trueTheta)}`,
      'measured',
    );

    const existing = chart.querySelector('svg');
    if (existing === null) {
      chart.append(plot.root);
    } else {
      existing.replaceWith(plot.root);
    }

    {
      attachCrosshair(chart, plot, tooltip, grid, (index) => {
        const theta = grid[index] as number;
        return {
          head: `θ = ${signed(theta)}`,
          rows: [
            {
              key: 'density',
              colour: SERIES,
              value: fixed(frame.density[index] as number, 3),
            },
            {
              key: 'inside 95%',
              colour: 'var(--text-muted)',
              value:
                theta >= frame.bounds.outer[0] && theta <= frame.bounds.outer[1] ? 'yes' : 'no',
            },
          ],
        };
      });
    }
  };

  const update = (model: ViewModel): void => {
    const administered = model.snapshot.transcript.length;
    const target = frameOf(model);
    const from = current;
    current = target;

    caption.replaceChildren();
    caption.append(
      html('span', {
        text:
          administered === 0
            ? `No responses yet, so this is the prior: 95% credible interval ${interval(
                model.intervals.outer.lower,
                model.intervals.outer.upper,
              )}, width ${fixed(model.intervals.outer.width)}. Answer an item to see it tighten.`
            : `95% credible interval ${interval(
                model.intervals.outer.lower,
                model.intervals.outer.upper,
              )}, width ${fixed(model.intervals.outer.width)} after ${administered} ${
                administered === 1 ? 'response' : 'responses'
              }. Density at the grid edge is ${percent(model.posterior.edgeRatio, 2)} of the peak.`,
      }),
    );

    tableHost.replaceChildren(quantileTable(model));

    cancelAnimationFrame(animation);
    if (from === null || prefersReducedMotion() || width === 0) {
      draw(model, target);
      return;
    }

    const start = performance.now();
    const step = (now: number): void => {
      const t = Math.min(1, (now - start) / TWEEN_MS);
      draw(model, t >= 1 ? target : blend(from, target, easeOut(t)));
      if (t < 1) {
        animation = requestAnimationFrame(step);
      }
    };
    animation = requestAnimationFrame(step);
  };

  let model: ViewModel | null = null;
  store.subscribe((next) => {
    model = next;
    if (width > 0) {
      update(next);
    }
  });

  // Measure the container and redraw on resize, so the plot is always rendered
  // at its real pixel width rather than stretched by a viewBox.
  const observer = new ResizeObserver((entries) => {
    const measured = Math.floor(entries[0]?.contentRect.width ?? 0);
    if (measured <= 0 || Math.abs(measured - width) < 2) {
      return;
    }
    width = measured;
    if (model !== null) {
      current = null;
      update(model);
    }
  });
  observer.observe(chart);
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

/**
 * The table twin of the density: the quantiles a reader would want to check.
 *
 * A density is hard to read a number off, so the table reports the intervals
 * and the moments rather than 321 rows of density values nobody would read.
 */
function quantileTable(model: ViewModel): HTMLElement {
  const posterior: Posterior = model.posterior;
  const table = html('table');
  const head = html('thead');
  const headRow = html('tr');
  headRow.append(
    html('th', { text: 'quantity' }),
    html('th', { className: 'num', text: 'lower' }),
    html('th', { className: 'num', text: 'upper' }),
    html('th', { className: 'num', text: 'width' }),
  );
  head.append(headRow);

  const body = html('tbody');
  for (const mass of [0.5, 0.8, 0.9, 0.95, 0.99]) {
    const ci = credibleInterval(posterior, mass);
    const row = html('tr');
    row.append(
      html('td', { text: `${percent(mass)} credible` }),
      html('td', { className: 'num', text: fixed(ci.lower, 3) }),
      html('td', { className: 'num', text: fixed(ci.upper, 3) }),
      html('td', { className: 'num', text: fixed(ci.width, 3) }),
    );
    body.append(row);
  }

  for (const [label, value] of [
    ['posterior mean', posterior.mean],
    ['posterior mode', posterior.mode],
    ['posterior SD', posterior.sd],
  ] as const) {
    const row = html('tr');
    row.append(
      html('td', { text: label }),
      html('td', { className: 'num', text: fixed(value, 3) }),
      html('td', { className: 'num', text: '' }),
      html('td', { className: 'num', text: '' }),
    );
    body.append(row);
  }

  table.append(head, body);
  return table;
}
