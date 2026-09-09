import { crisp, haloText, svg } from './svg.js';
import { linearScale, niceTicks, type Scale } from './scale.js';

export interface Insets {
  readonly left: number;
  readonly right: number;
  readonly top: number;
  readonly bottom: number;
}

export interface PlotOptions {
  readonly width: number;
  readonly height: number;
  readonly xDomain: readonly [number, number];
  readonly yDomain: readonly [number, number];
  readonly insets?: Partial<Insets>;
  /** Target number of x ticks. Reduced automatically on a narrow plot. */
  readonly xTicks?: number;
  /** Target number of y ticks. */
  readonly yTicks?: number;
  readonly xTickFormat?: (value: number) => string;
  readonly yTickFormat?: (value: number) => string;
  readonly xLabel?: string;
  /**
   * Omit the value axis entirely. For a posterior density the y units are
   * density, which means nothing to a reader — the interval labels carry the
   * information instead, so the axis is only ink.
   */
  readonly hideYAxis?: boolean;
}

const DEFAULT_INSETS: Insets = { left: 44, right: 16, top: 12, bottom: 28 };

/**
 * A cartesian plot: insets, gridlines, axes, and layers to draw into.
 *
 * The container is sized to the plot area plus the axis band rather than to a
 * fixed height, so the x-axis labels are never clipped and no chart grows its
 * own scrollbar.
 */
export class Plot {
  readonly root: SVGSVGElement;
  readonly x: Scale;
  readonly y: Scale;
  readonly insets: Insets;
  readonly width: number;
  readonly height: number;

  /** Below the data: gridlines and axis rules. */
  private readonly gridLayer: SVGGElement;
  /** The data itself. */
  readonly dataLayer: SVGGElement;
  /** Above the data: reference lines, direct labels, crosshair. */
  readonly overlay: SVGGElement;

  constructor(options: PlotOptions) {
    const insets = { ...DEFAULT_INSETS, ...options.insets };
    this.insets = options.hideYAxis ? { ...insets, left: 12 } : insets;
    this.width = options.width;
    this.height = options.height;

    this.root = svg('svg', {
      width: options.width,
      height: options.height,
      viewBox: `0 0 ${options.width} ${options.height}`,
      role: 'presentation',
    });

    this.x = linearScale(options.xDomain, [this.insets.left, options.width - this.insets.right]);
    this.y = linearScale(options.yDomain, [
      options.height - this.insets.bottom,
      this.insets.top,
    ]);

    this.gridLayer = svg('g');
    this.dataLayer = svg('g');
    this.overlay = svg('g');
    this.root.append(this.gridLayer, this.dataLayer, this.overlay);

    this.drawFrame(options);
  }

  get plotLeft(): number {
    return this.insets.left;
  }

  get plotRight(): number {
    return this.width - this.insets.right;
  }

  get plotTop(): number {
    return this.insets.top;
  }

  get plotBottom(): number {
    return this.height - this.insets.bottom;
  }

  /**
   * Gridlines, the baseline rule and the tick labels.
   *
   * Gridlines are horizontal only and solid: a dashed grid reads as a
   * projection or a threshold, and the one legitimate dash in the system is
   * reserved for a target reference line. The vertical axis spine is omitted —
   * the horizontal gridlines already establish where the values sit, and a
   * second rule beside their left ends is redundant ink.
   */
  private drawFrame(options: PlotOptions): void {
    const usableWidth = this.plotRight - this.plotLeft;
    const xTickTarget = usableWidth < 360 ? 4 : (options.xTicks ?? 7);
    const yTicks = niceTicks(this.y.domain, options.yTicks ?? 5);
    const xTicks = niceTicks(this.x.domain, xTickTarget);

    const formatY = options.yTickFormat ?? ((value: number) => String(value));
    const formatX = options.xTickFormat ?? ((value: number) => String(value));

    if (!options.hideYAxis) {
      for (const value of yTicks) {
        const y = crisp(this.y.to(value));
        this.gridLayer.append(
          svg('line', {
            x1: this.plotLeft,
            x2: this.plotRight,
            y1: y,
            y2: y,
            stroke: 'var(--grid)',
            'stroke-width': 1,
            'shape-rendering': 'crispEdges',
          }),
        );
        const label = haloText(this.plotLeft - 8, y, formatY(value), {
          'text-anchor': 'end',
          'dominant-baseline': 'middle',
          stroke: 'none',
        });
        label.style.fontVariantNumeric = 'tabular-nums';
        this.gridLayer.append(label);
      }
    }

    const baseline = crisp(this.plotBottom);
    this.gridLayer.append(
      svg('line', {
        x1: this.plotLeft,
        x2: this.plotRight,
        y1: baseline,
        y2: baseline,
        stroke: 'var(--axis)',
        'stroke-width': 1,
        'shape-rendering': 'crispEdges',
      }),
    );

    for (const [index, value] of xTicks.entries()) {
      const x = crisp(this.x.to(value));
      this.gridLayer.append(
        svg('line', {
          x1: x,
          x2: x,
          y1: baseline,
          y2: baseline + 4,
          stroke: 'var(--axis)',
          'stroke-width': 1,
          'shape-rendering': 'crispEdges',
        }),
      );
      // The first and last labels anchor inward so they cannot overhang the plot.
      const anchor = index === 0 ? 'start' : index === xTicks.length - 1 ? 'end' : 'middle';
      const label = haloText(x, baseline + 18, formatX(value), {
        'text-anchor': anchor,
        stroke: 'none',
      });
      label.style.fontVariantNumeric = 'tabular-nums';
      this.gridLayer.append(label);
    }

    if (options.xLabel !== undefined) {
      this.gridLayer.append(
        haloText((this.plotLeft + this.plotRight) / 2, this.height - 2, options.xLabel, {
          'text-anchor': 'middle',
          stroke: 'none',
          fill: 'var(--text-secondary)',
        }),
      );
    }
  }

  /** Dim the whole frame, for the empty state where there is no data yet. */
  dimFrame(opacity: number): void {
    this.gridLayer.setAttribute('opacity', String(opacity));
  }

  /**
   * A horizontal reference line.
   *
   * `measured` is a value the system observed and draws solid; `target` is a
   * goal it has not necessarily reached and draws dashed. Neither wears a
   * series colour, because a coloured reference line reads as one more series.
   * The label carries the value, since "target" alone tells a reader nothing
   * they can check.
   */
  referenceY(value: number, label: string, kind: 'measured' | 'target' = 'target'): void {
    const y = crisp(this.y.to(value));
    if (y < this.plotTop || y > this.plotBottom) {
      return;
    }
    this.overlay.append(
      svg('line', {
        x1: this.plotLeft,
        x2: this.plotRight,
        y1: y,
        y2: y,
        stroke: kind === 'target' ? 'var(--axis)' : 'var(--text-muted)',
        'stroke-width': 1,
        ...(kind === 'target' ? { 'stroke-dasharray': '4 3' } : {}),
        'shape-rendering': 'crispEdges',
      }),
      haloText(this.plotRight - 6, y - 6, label, { 'text-anchor': 'end' }),
    );
  }

  /** A vertical reference line, for a value on the ability axis. */
  referenceX(value: number, label: string, kind: 'measured' | 'target' = 'measured'): void {
    const x = crisp(this.x.to(value));
    if (x < this.plotLeft || x > this.plotRight) {
      return;
    }
    this.overlay.append(
      svg('line', {
        x1: x,
        x2: x,
        y1: this.plotTop,
        y2: this.plotBottom,
        stroke: kind === 'target' ? 'var(--axis)' : 'var(--text-muted)',
        'stroke-width': 1,
        ...(kind === 'target' ? { 'stroke-dasharray': '4 3' } : {}),
        'shape-rendering': 'crispEdges',
      }),
      haloText(x + 6, this.plotTop + 11, label, { 'text-anchor': 'start' }),
    );
  }
}

export interface CrosshairRow {
  readonly key: string;
  readonly colour: string;
  readonly value: string;
}

/**
 * Attach a crosshair that snaps to the nearest sampled x and reports every
 * series at that x.
 *
 * A crosshair rather than per-curve hit testing: the pointer only has to be at
 * the right horizontal position, never on a 2px line. Keyboard users get the
 * same readout through the arrow keys, so the tooltip enhances the chart
 * instead of gating it.
 */
export function attachCrosshair(
  container: HTMLElement,
  plot: Plot,
  tooltip: HTMLElement,
  samples: readonly number[],
  describe: (index: number) => { readonly head: string; readonly rows: readonly CrosshairRow[] },
): void {
  if (samples.length === 0) {
    return;
  }

  const line = svg('line', {
    y1: plot.plotTop,
    y2: plot.plotBottom,
    stroke: 'var(--axis)',
    'stroke-width': 1,
    'shape-rendering': 'crispEdges',
    opacity: 0,
  });
  plot.overlay.append(line);

  let active = -1;

  const show = (index: number): void => {
    active = index;
    const x = crisp(plot.x.to(samples[index] as number));
    line.setAttribute('x1', String(x));
    line.setAttribute('x2', String(x));
    line.setAttribute('opacity', '1');

    const { head, rows } = describe(index);
    tooltip.replaceChildren();
    const heading = document.createElement('div');
    heading.className = 'tooltip__head';
    heading.textContent = head;
    tooltip.append(heading);

    for (const row of rows) {
      const line_ = document.createElement('div');
      line_.className = 'tooltip__row';
      const key = document.createElement('span');
      key.className = 'tooltip__key';
      const swatch = document.createElement('span');
      swatch.className = 'legend__line';
      swatch.style.background = row.colour;
      const name = document.createElement('span');
      name.textContent = row.key;
      key.append(swatch, name);
      const value = document.createElement('span');
      value.className = 'tooltip__value';
      value.textContent = row.value;
      line_.append(key, value);
      tooltip.append(line_);
    }

    tooltip.hidden = false;
    // Flip the tooltip to the other side of the crosshair near the right edge
    // so it is never clipped by the panel.
    const flip = x > (plot.plotLeft + plot.plotRight) / 2;
    tooltip.style.left = `${flip ? x - 12 : x + 12}px`;
    tooltip.style.transform = flip ? 'translateX(-100%)' : 'none';
    tooltip.style.top = `${plot.plotTop}px`;
  };

  const hide = (): void => {
    active = -1;
    line.setAttribute('opacity', '0');
    tooltip.hidden = true;
  };

  const nearest = (clientX: number): number => {
    const bounds = plot.root.getBoundingClientRect();
    const theta = plot.x.from(clientX - bounds.left);
    let best = 0;
    let bestGap = Infinity;
    for (const [i, sample] of samples.entries()) {
      const gap = Math.abs(sample - theta);
      if (gap < bestGap) {
        bestGap = gap;
        best = i;
      }
    }
    return best;
  };

  container.addEventListener('pointermove', (event) => {
    show(nearest(event.clientX));
  });
  container.addEventListener('pointerleave', hide);

  container.tabIndex = 0;
  container.addEventListener('focus', () => {
    show(active < 0 ? Math.floor(samples.length / 2) : active);
  });
  container.addEventListener('blur', hide);
  container.addEventListener('keydown', (event) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') {
      return;
    }
    event.preventDefault();
    const step = event.key === 'ArrowLeft' ? -1 : 1;
    const start = active < 0 ? Math.floor(samples.length / 2) : active;
    show(Math.min(samples.length - 1, Math.max(0, start + step)));
  });
}
