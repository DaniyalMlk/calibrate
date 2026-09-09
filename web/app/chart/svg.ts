const SVG_NS = 'http://www.w3.org/2000/svg';

/** Create an SVG element and set attributes in one call. */
export function svg<K extends keyof SVGElementTagNameMap>(
  name: K,
  attributes: Record<string, string | number> = {},
  children: readonly SVGElement[] = [],
): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  for (const child of children) {
    element.append(child);
  }
  return element;
}

/** Create an HTML element, setting class, text and attributes. */
export function html<K extends keyof HTMLElementTagNameMap>(
  name: K,
  options: {
    readonly className?: string;
    readonly text?: string;
    readonly attributes?: Record<string, string | number>;
    readonly children?: readonly Node[];
  } = {},
): HTMLElementTagNameMap[K] {
  const element = document.createElement(name);
  if (options.className !== undefined) {
    element.className = options.className;
  }
  if (options.text !== undefined) {
    // textContent, never innerHTML: item identifiers and bank labels are data,
    // and a bank loaded from a CSV somebody else produced is untrusted data.
    element.textContent = options.text;
  }
  for (const [key, value] of Object.entries(options.attributes ?? {})) {
    element.setAttribute(key, String(value));
  }
  for (const child of options.children ?? []) {
    element.append(child);
  }
  return element;
}

/**
 * A text label with a halo of the surface colour painted behind the glyphs.
 *
 * `paint-order: stroke fill` draws the stroke first and the fill over it, so a
 * label that crosses a curve or a gridline stays legible without a filled
 * rectangle behind it — which would occlude the very data the label annotates.
 */
export function haloText(
  x: number,
  y: number,
  content: string,
  attributes: Record<string, string | number> = {},
): SVGTextElement {
  const element = svg('text', {
    x,
    y,
    fill: 'var(--text-muted)',
    stroke: 'var(--surface)',
    'stroke-width': 2.5,
    'stroke-linejoin': 'round',
    'paint-order': 'stroke fill',
    'font-size': 11,
    'font-weight': 500,
    ...attributes,
  });
  element.textContent = content;
  return element;
}

/** Build an SVG path `d` from points, skipping any that are not finite. */
export function polyline(points: readonly (readonly [number, number])[]): string {
  const parts: string[] = [];
  let open = false;
  for (const [x, y] of points) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      open = false;
      continue;
    }
    parts.push(`${open ? 'L' : 'M'}${x.toFixed(2)} ${y.toFixed(2)}`);
    open = true;
  }
  return parts.join(' ');
}

/**
 * Build a closed band between an upper and a lower edge sharing an x sequence.
 *
 * Drawn as one path rather than two, so the band has a single fill with no
 * seam down the middle where two half-opacity fills would overlap.
 */
export function bandPath(
  xs: readonly number[],
  upper: readonly number[],
  lower: readonly number[],
): string {
  const forward: (readonly [number, number])[] = [];
  const backward: (readonly [number, number])[] = [];
  for (let i = 0; i < xs.length; i += 1) {
    forward.push([xs[i] as number, upper[i] as number]);
    backward.push([xs[xs.length - 1 - i] as number, lower[xs.length - 1 - i] as number]);
  }
  const top = polyline(forward);
  if (top === '') {
    return '';
  }
  const bottom = polyline(backward).replace(/^M/, 'L');
  return `${top} ${bottom} Z`;
}

/** Snap a coordinate to the half-pixel grid so a 1px line renders as 1px. */
export function crisp(value: number): number {
  return Math.round(value) + 0.5;
}
