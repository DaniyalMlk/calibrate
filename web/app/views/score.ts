import {
  linspace,
  maximumTestScore,
  testCharacteristicCurve,
  type AnyItem,
} from '../../../src/index.js';
import { attachCrosshair, Plot } from '../chart/plot.js';
import { bandPath, crisp, html, polyline, svg } from '../chart/svg.js';
import { fixed, percent, signed } from '../format.js';
import { THETA_DOMAIN, type Store, type ViewModel } from '../state.js';

const CURVE = 'var(--series-3)';
const OBSERVED = 'var(--series-6)';

const GRID = linspace(THETA_DOMAIN[0], THETA_DOMAIN[1], 201);

/**
 * The test characteristic curve: expected total score against ability.
 *
 * This is the bridge between the metric the engine works in and the number a
 * candidate is actually told. On a dichotomous form the two are close enough
 * that the curve is a formality — twenty items, twenty points, and a raw score
 * of fourteen means roughly what anyone would guess. On a mixed form it is not:
 * sixteen items can be worth forty-two points, so "21 of 42" is uninterpretable
 * until you know that a candidate at this ability is expected to score 19.7.
 *
 * It is also the function two forms must agree on for their scores to mean the
 * same thing, which is why the linking methods in the library are written
 * against it.
 */
export function mountScore(root: HTMLElement, store: Store): void {
  const heading = html('div');
  heading.append(
    html('h2', { className: 'panel__title', text: 'Expected score' }),
    html('p', {
      className: 'panel__note',
      text:
        'The total score a candidate of each ability is expected to earn on the items ' +
        'administered so far — the bridge between the ability metric and a reported raw ' +
        'score. The observed score is marked against it.',
    }),
  );

  const badge = html('span', { className: 'badge badge--neutral' });
  const head = html('div', { className: 'panel__head' });
  head.append(heading, badge);

  const legend = html('div', { className: 'legend' });
  legend.append(legendItem('Expected total score', CURVE), legendItem('Observed score', OBSERVED));

  const chart = html('div', { className: 'chart' });
  const tip = html('div', { className: 'tooltip', attributes: { role: 'status' } });
  tip.hidden = true;
  chart.append(tip);

  const caption = html('p', { className: 'chart__caption' });
  root.append(head, legend, chart, caption);

  let width = 0;
  let model: ViewModel | null = null;

  const draw = (current: ViewModel): void => {
    const items: readonly AnyItem[] = current.responses.map((response) => response.item);
    const maximum = maximumTestScore(items);
    badge.textContent = `${items.length} ${items.length === 1 ? 'item' : 'items'}`;

    // The axis runs from zero to the form's own maximum, never fitted to the
    // curve. The whole point of the picture is where the observed score sits
    // between nothing and everything, and an axis cropped to the range the
    // curve happens to occupy would move that reference under the reader.
    const ceiling = Math.max(maximum, 1);
    const plot = new Plot({
      width,
      height: 230,
      xDomain: THETA_DOMAIN,
      yDomain: [0, ceiling],
      yTicks: 5,
      xTickFormat: (value) => (value === 0 ? '0' : signed(value, 0)),
      xLabel: 'ability θ, logits',
    });

    if (items.length === 0) {
      plot.dimFrame(0.4);
      replaceSvg(chart, plot.root);
      caption.textContent =
        'No items administered yet. The curve will run from zero to the total the ' +
        'administered items are worth, and will be steepest where they measure best.';
      return;
    }

    const expected = GRID.map((theta) => testCharacteristicCurve(items, theta));

    // The band under the curve, to the observed score's own ability. It reads
    // as "this much of the form, up to here", which is the sentence a raw score
    // is trying to be.
    const estimate = current.snapshot.theta;
    const upTo = GRID.filter((theta) => theta <= estimate);
    if (upTo.length >= 2) {
      const xs = upTo.map((theta) => plot.x.to(theta));
      plot.dataLayer.append(
        svg('path', {
          d: bandPath(
            xs,
            upTo.map((_, i) => plot.y.to(expected[i] as number)),
            xs.map(() => plot.plotBottom),
          ),
          fill: CURVE,
          'fill-opacity': 'var(--band-outer)',
          stroke: 'none',
        }),
      );
    }

    plot.dataLayer.append(
      svg('path', {
        d: polyline(
          GRID.map((theta, i) => [plot.x.to(theta), plot.y.to(expected[i] as number)]),
        ),
        fill: 'none',
        stroke: CURVE,
        'stroke-width': 2,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
      }),
    );

    // The observed score as a horizontal rule and the estimate as a vertical
    // one. Where they meet is on the curve when the model fits the candidate,
    // and off it when it does not — which is the one thing worth seeing here
    // and needs no annotation to be seen.
    const earned = current.score.earned;
    const scoreY = crisp(plot.y.to(earned));
    plot.overlay.append(
      svg('line', {
        x1: plot.plotLeft,
        x2: plot.plotRight,
        y1: scoreY,
        y2: scoreY,
        stroke: OBSERVED,
        'stroke-width': 1.5,
        'stroke-dasharray': '4 3',
        'shape-rendering': 'crispEdges',
      }),
    );
    plot.referenceX(estimate, `estimate θ ${signed(estimate)}`, 'measured');

    const expectedHere = testCharacteristicCurve(items, estimate);
    const x = plot.x.to(estimate);
    plot.overlay.append(
      svg('circle', {
        cx: x,
        cy: plot.y.to(expectedHere),
        r: 4,
        fill: CURVE,
        stroke: 'var(--surface)',
        'stroke-width': 2,
      }),
      svg('circle', {
        cx: x,
        cy: plot.y.to(earned),
        r: 4,
        fill: OBSERVED,
        stroke: 'var(--surface)',
        'stroke-width': 2,
      }),
    );

    replaceSvg(chart, plot.root);
    attachCrosshair(chart, plot, tip, GRID, (index) => ({
      head: `θ = ${signed(GRID[index] as number)}`,
      rows: [
        {
          key: 'expected score',
          colour: CURVE,
          value: `${fixed(expected[index] as number, 2)} of ${maximum}`,
        },
        { key: 'observed', colour: OBSERVED, value: `${earned} of ${maximum}` },
      ],
    }));

    const residual = earned - expectedHere;
    caption.textContent =
      `${items.length} items worth ${maximum} points. At the estimate of ` +
      `${signed(estimate)} the model expects ${fixed(expectedHere, 1)}; the candidate scored ` +
      `${earned}, ${
        Math.abs(residual) < 0.05
          ? 'which is the expectation exactly'
          : `${fixed(Math.abs(residual), 1)} ${residual > 0 ? 'above' : 'below'} it`
      }. Reported as a proportion of the maximum that is ${percent(earned / maximum)}.`;
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
  observer.observe(chart);
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
