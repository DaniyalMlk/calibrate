/**
 * The documentation site, generated from the engine rather than written beside it.
 *
 * Three rules decide what is allowed on the page.
 *
 * **Every figure comes from code that runs.** The evidence table is rendered
 * from `scripts/evidence.ts` — the same gate CI runs and the same rows the
 * README carries — and the counts are read off the command table and the
 * package's own export list. Nothing here is a number somebody typed, because
 * a number typed on a page is a number that will be wrong in a month.
 * `test/docs.test.ts` fails in both directions if that stops being true.
 *
 * **The interface is the interface.** Not a recording and not a mock-up: the
 * adaptive session on the page imports the engine straight from the compiled
 * source, so what it believes about a candidate is what the library believes.
 * It needs no bundler — `tsc` emits ES modules whose import specifiers the
 * browser resolves — so publishing it is a type-check and a copy.
 *
 * **The palette is the interface's.** `web/styles.css` already settled it:
 * a cool neutral ramp, six series hexes chosen to clear 3:1 against both
 * surfaces, and dark by default because every view here is chart-dominant and
 * ink-on-dark puts the brightest pixels on the data. The landing page
 * inherits those tokens exactly, so the page and the thing it is about read as
 * one piece of work.
 *
 * The site is two files plus the built interface, and `site/` is gitignored:
 * the repository holds this generator, not its output.
 */

import { measure, renderTable } from '../scripts/evidence.js';
import type { Check } from '../scripts/evidence.js';
import { COMMAND_NAMES } from '../src/cli/run.js';
import * as calibrate from '../src/index.js';

/** The published name, which is not the repository's name. */
export const PACKAGE = 'calibrate-irt';
export const SOURCE = 'https://github.com/DaniyalMlk/calibrate';

/** Where the interface is published relative to the landing page. */
export const APP = 'session/';

/** Counts the page quotes, each read off the thing it is a count of. */
export interface Counts {
  readonly commands: number;
  readonly exports: number;
}

export function counts(): Counts {
  return {
    commands: COMMAND_NAMES.length,
    exports: Object.keys(calibrate).length,
  };
}

function esc(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** `` `x` `` to `<code>x</code>`, after escaping. */
function code(text: string): string {
  return text.replace(/`([^`]+)`/g, (_, inner: string) => `<code>${inner}</code>`);
}

/**
 * The markdown rows the evidence gate renders, as cells.
 *
 * The gate owns the rows; this only changes the delimiters.
 */
export function evidenceRows(checks: readonly Check[]): readonly (readonly string[])[] {
  return renderTable(checks)
    .split('\n')
    .filter((line) => line.startsWith('| ') && !line.startsWith('| ---'))
    .map((line) =>
      line
        .slice(1, -1)
        .split(' | ')
        .map((cell) => cell.trim()),
    );
}

function evidenceTable(checks: readonly Check[]): string {
  const [header, ...body] = evidenceRows(checks);
  const head = (header ?? []).map((cell) => `<th>${code(esc(cell))}</th>`).join('');
  const lines = body
    .map((row) => `<tr>${row.map((cell) => `<td>${code(esc(cell))}</td>`).join('')}</tr>`)
    .join('\n');
  return `<table class="evidence">\n<thead><tr>${head}</tr></thead>\n<tbody>\n${lines}\n</tbody>\n</table>`;
}

function shell(body: string, description: string): string {
  return `<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>calibrate &middot; an item response theory engine</title>
<meta name="description" content="${esc(description)}">
<meta name="color-scheme" content="dark light">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='4' fill='%231086d3'/%3E%3Cpath d='M3 11.5C5 11.5 5.5 4.5 8 4.5s3 7 5 7' stroke='white' stroke-width='1.6' fill='none' stroke-linecap='round'/%3E%3C/svg%3E">
<link rel="stylesheet" href="style.css">
</head>
<body>
${body}
</body>
</html>
`;
}

/**
 * The landing page.
 *
 * It leads with the one result on the page that is a *statement about the
 * model* rather than about the arithmetic. Under Rasch the total score is
 * sufficient: every response pattern with the same number right gives the same
 * ability, exactly, and the engine reproduces that to the root-finder's
 * tolerance and no further — which is the honest bound, because beyond it the
 * figure is reporting the tolerance rather than the theorem. Under the 2PL the
 * same comparison spans 2.31 logits. One number says the implementation is
 * right and the other says why the choice of model matters, and no amount of
 * prose does either.
 */
function landing(checks: readonly Check[], size: Counts): string {
  const find = (fragment: string): Check => {
    const match = checks.find((check) => check.label.includes(fragment));
    if (match === undefined) throw new Error(`the gate has no row about ${fragment}`);
    return match;
  };
  const peak = find('Peak information');
  const linking = find('linking methods');

  // Pulled out of the gate's own rendered result rather than typed beside it.
  // A headline figure is the one most likely to be written once and left, so
  // it is read back out of the row it summarises; `test/docs.test.ts` asserts
  // both captures found something and that what they found is in the table.
  const sufficiency = find('same total score').result;
  const spread = /spans ([\d.]+) logits/.exec(sufficiency)?.[1];
  const patterns = /over all ([\d,]+) patterns/.exec(sufficiency)?.[1];
  if (spread === undefined || patterns === undefined) {
    throw new Error(`the sufficiency row no longer says what the headline quotes: ${sufficiency}`);
  }

  return shell(
    `<header class="masthead col--wide">
  <a class="mark" href="./"><span class="mark__glyph" aria-hidden="true"></span>calibrate</a>
  <nav class="masthead__nav">
    <a href="#session">The interface</a>
    <a href="#evidence">What is checked</a>
    <a href="#install">Install</a>
    <a href="${esc(SOURCE)}">Source</a>
  </nav>
</header>

<main>

<section class="lede col--wide">
  <h1>Two candidates, the same score, different&nbsp;abilities.</h1>
  <p class="standfirst">An item response theory engine in TypeScript with no
  runtime dependencies. It calibrates an item bank, estimates what a candidate
  can do, chooses what to ask next, puts two calibrations on one scale and says
  which items behave differently for different groups &mdash; and it is checked
  against closed forms and published tables rather than against itself.</p>
  <p class="actions">
    <a class="cta" href="${esc(APP)}">Sit an adaptive test</a>
    <a class="plain" href="${esc(SOURCE)}">Read the source</a>
  </p>
</section>

<section class="claim">
  <div class="claim__inner col--wide">
    <div class="claim__figures">
      <p class="claim__figure"><strong>${esc(spread)}</strong><span>logits
      between two patterns with the <em>same total score</em>, under the
      two-parameter model</span></p>
      <p class="claim__figure"><strong>0</strong><span>difference between those
      same patterns under Rasch, over all ${esc(patterns)} of them</span></p>
      <p class="claim__figure"><strong>${size.exports}</strong><span>names the
      package exports, across ${size.commands} command-line tools</span></p>
    </div>
    <p class="claim__detail">Those first two are one row of the table below,
    and they are the row worth reading twice. Under Rasch the total score is
    <em>sufficient</em>: every pattern with the same number right gives the
    same ability, as a theorem rather than as an approximation, and the engine
    reproduces it to the root-finder&rsquo;s tolerance and no further &mdash;
    which is the honest bound, because past that the figure reports the
    tolerance and not the theorem. Under the two-parameter model the same
    comparison spans ${esc(spread)} logits, because <em>which</em> items a
    candidate got right carries information that the count throws away. One
    figure says the implementation is right; the other says why the choice of
    model is not a detail.</p>
  </div>
</section>

<section id="session" class="session">
  <div class="session__intro col--wide">
    <h2>The engine, running</h2>
    <p>Not a recording and not a mock-up: the session below imports the engine
    from the same compiled source the package ships, so what it believes about
    you is what the library believes. Answer an item and watch the posterior
    move &mdash; the panel beside it reports how much that one answer was
    worth, which is the quantity an adaptive test is actually choosing on.</p>
    <p>It needs no bundler. <code>tsc</code> emits ES modules whose import
    specifiers the browser resolves, so the whole interface is the compiled
    library, a stylesheet and a page, served as files. That is why it can be
    published from a static site at all, and it is the same reason the package
    has no runtime dependencies.</p>
    <p class="session__hint">Try finishing a session, then starting another and
    answering the opposite way. The bank panel shows which items were exposed
    and how concentrated that exposure was &mdash; the thing that decides
    whether a bank survives contact with a real cohort.</p>
  </div>
  <div class="session__frame">
    <iframe src="${esc(APP)}" title="calibrate, running an adaptive session"
            loading="lazy" referrerpolicy="no-referrer"></iframe>
  </div>
  <p class="session__fallback col--wide"><a href="${esc(APP)}">Open the
  interface in its own tab</a> &mdash; it wants a window wider than a
  phone.</p>
</section>

<section id="evidence" class="evidence-section col--wide">
  <h2>What is checked, and against what</h2>
  <p>An ability estimate is a number someone is asked to believe about a
  person, so the question that matters about this engine is how much of it was
  checked and against what. Every figure in this table is produced by
  ${code('`npm run evidence`')}, which runs in CI and exits non-zero if any of
  them moves &mdash; and it is the same gate that renders this page, so the
  page cannot drift from the engine either.</p>
  ${evidenceTable(checks)}
  <p class="aside"><strong>An identity says zero, not
  &ldquo;small&rdquo;.</strong> The peak of a two-parameter item&rsquo;s
  information is ${code('`a&sup2; / 4`')} by construction, and a two-category
  graded item <em>is</em> the two-parameter model &mdash; the polytomous code
  path reduces to the same arithmetic &mdash; so the difference is exactly
  nothing and the row says so. ${esc(peak.result)}. Quoting an identity as a
  bound would be quoting a weaker claim than the one that holds.</p>
  <p class="aside"><strong>The rows that <em>are</em> bounds have a reason to
  be.</strong> A quadrature, a root finder or a sum over a grid leaves
  floating-point residue, and a bound is the true shape of that claim. What
  none of them is, is a digit count chosen to make a number look
  impressive.</p>
  <p class="aside"><strong>Four of these rows check against something that is
  not this package.</strong> A transformation applied on purpose and recovered
  by four independent linking methods (${esc(linking.result)}); a closed-form
  area against numerical quadrature; a chi-square tail against published
  critical values; and a scale inflation against
  ${code('`L / (L - 1)`')}, which is a prediction made before the measurement.
  A calculation checked only against itself is consistent, and consistency is
  not correctness &mdash; an estimator with the wrong convention is perfectly
  self-consistent.</p>
</section>

<section id="install" class="install col--wide">
  <h2>The first minute</h2>
  <pre><code>npm install ${esc(PACKAGE)}</code></pre>
  <pre><code>npx ${esc(PACKAGE)} demo        # one adaptive session, item by item
npx ${esc(PACKAGE)} study       # selection policies over a population
npx ${esc(PACKAGE)} dif         # items behaving differently for two groups</code></pre>
  <p><strong>The package is <code>${esc(PACKAGE)}</code>, not
  <code>calibrate</code>.</strong> <code>calibrate</code> on the registry is an
  unrelated package, so the distribution name differs; the command is available
  under both names, so <code>npx ${esc(PACKAGE)}</code> resolves. Node 20.19 or
  newer. The package ships compiled JavaScript and type declarations, exports
  ${size.exports} names and has no runtime dependencies.</p>
  <pre><code>import { gradedResponse, estimateEap, nextItem } from "${esc(PACKAGE)}";</code></pre>
  <p class="aside">Until the trusted publisher is registered on the registry,
  the install line is the one thing on this page that does not work yet, and it
  installs from source in the meantime:
  ${code('`git clone ' + SOURCE + '.git && npm ci`')}.</p>
</section>

<section class="covers col--wide">
  <h2>What is in it</h2>
  <p>The one-, two- and three-parameter logistic models, the graded response
  model and the partial credit model, with the polytomous path checked against
  the dichotomous one where the two must agree. Ability by maximum likelihood,
  by expected a posteriori and by maximum a posteriori, each with its standard
  error, and the whole posterior rather than only its mean. Item and test
  information, the test characteristic curve, and the summed-score distribution
  by recursion. Item selection by information, by a posterior-weighted
  criterion and with exposure control, compared over a simulated population
  rather than argued about. Calibration both jointly and marginally, with the
  scale inflation the joint estimator is known to carry measured against its
  closed form. Four linking methods, differential item functioning by matched
  groups and by model, person fit with the correction for an ability estimated
  from the same responses, and bank health as coverage across the ability
  range.</p>
  <p>The reasoning is in the source rather than in a wiki. Every module says
  which convention it implements and why that one rather than the other, which
  is the part of a measurement library that is actually hard &mdash; and
  ${code('`ROADMAP.md`')} says what is not modelled, which is the other
  part.</p>
</section>

</main>

<footer class="foot col--wide">
  <p>calibrate, published as ${code('`' + PACKAGE + '`')}. Built from the
  engine&rsquo;s own evidence gate, so no figure on this page was typed by
  hand. <a href="${esc(SOURCE)}">Source</a>.</p>
</footer>
`,
    'calibrate is an item response theory engine in TypeScript: item models, ' +
      'ability estimation, adaptive item selection, calibration, scale ' +
      'linking and differential item functioning, checked against closed ' +
      'forms and published tables.',
  );
}

/**
 * The stylesheet, inheriting the interface's tokens so the two read as one thing.
 *
 * Dark by default and stamped on the document, exactly as `web/index.html` is,
 * because the thing this page is about is chart-dominant and the interface
 * sitting in the middle of it already made that choice. The light override is
 * the interface's too, guarded the same way so an explicit choice still wins.
 */
const STYLE = `/*
 * Generated by docs/site.ts. The tokens are web/styles.css's.
 */

:root {
  --n00: #090a0d;
  --n05: #0f1014;
  --n10: #16171b;
  --n15: #1f2025;
  --n20: #282a2f;
  --n30: #3b3d43;
  --n40: #56585f;
  --n60: #909299;
  --n70: #aeafb5;
  --n80: #cbccd0;
  --n90: #e5e6e9;
  --n95: #f2f2f4;
  --n99: #fbfbfc;

  --series-1: #1086d3;
  --series-2: #cc7200;

  --font: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
    "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;

  --measure: 38rem;
  --wide: 64rem;
  --gutter: 1.25rem;
  --rail: max(var(--gutter), 50% - var(--wide) / 2);

  /* Dark is the default here, as it is on the interface. */
  --canvas: var(--n05);
  --surface: var(--n10);
  --sunken: var(--n00);
  --rule: var(--n20);
  --rule-strong: var(--n30);
  --ink: var(--n95);
  --ink-soft: var(--n70);
  --ink-mute: var(--n50, #74757d);
  --accent: var(--series-1);
}

:root[data-theme="light"],
:root:where(:not([data-theme="dark"])) {
  color-scheme: light dark;
}

@media (prefers-color-scheme: light) {
  :root:where(:not([data-theme="dark"])) {
    --canvas: var(--n99);
    --surface: #ffffff;
    --sunken: var(--n95);
    --rule: var(--n90);
    --rule-strong: var(--n80);
    --ink: var(--n05);
    --ink-soft: var(--n40);
    --ink-mute: var(--n50);
  }
}

.col--wide { padding-inline: var(--rail); }

/*
 * Prose is capped at the measure by a child selector, never on the element
 * carrying the rail: with border-box sizing the padding would eat the
 * max-width and leave the text a few words across.
 */
.lede > *, .session__intro > *, .install > *, .covers > *, .foot > p,
.evidence-section > p { max-width: var(--measure); }
.lede h1 { max-width: 20ch; }

*, *::before, *::after { box-sizing: border-box; }

html { -webkit-text-size-adjust: 100%; }

body {
  margin: 0;
  background: var(--canvas);
  color: var(--ink);
  font-family: var(--font);
  font-size: 1.0625rem;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}

a { color: var(--accent); text-underline-offset: 2px; }
a:hover { color: var(--ink); }

code, pre, .claim__figure strong, .evidence td, .evidence th {
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
}

code { font-size: 0.9em; }

pre {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 7px;
  padding: 0.9rem 1.05rem;
  overflow-x: auto;
  font-size: 0.8125rem;
  line-height: 1.6;
  margin: 1.05rem 0;
  max-width: var(--measure);
}

pre code { font-size: inherit; }

h1, h2 { letter-spacing: -0.022em; font-weight: 640; text-wrap: balance; }

/* -- masthead ------------------------------------------------------------ */

.masthead {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1.5rem;
  align-items: baseline;
  justify-content: space-between;
  padding-block: 1.35rem;
  border-bottom: 1px solid var(--rule);
}

.mark {
  font-weight: 650;
  font-size: 1.0625rem;
  color: var(--ink);
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
}

.mark__glyph {
  width: 0.9rem;
  height: 0.9rem;
  border-radius: 3px;
  background: var(--series-1);
  flex: none;
}

.masthead__nav { display: flex; flex-wrap: wrap; gap: 0.35rem 1.2rem; font-size: 0.9375rem; }
.masthead__nav a { color: var(--ink-soft); text-decoration: none; }
.masthead__nav a:hover { color: var(--accent); }

/* -- lede ---------------------------------------------------------------- */

.lede { padding-block: 4rem 2.4rem; }

h1 { margin: 0 0 1.05rem; font-size: clamp(2rem, 5.6vw, 2.95rem); line-height: 1.08; }

.standfirst { margin: 0; font-size: 1.125rem; color: var(--ink-soft); }

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem 1.2rem;
  align-items: center;
  margin: 1.8rem 0 0;
}

.cta {
  background: var(--series-1);
  color: #ffffff;
  padding: 0.58rem 1.1rem;
  border-radius: 6px;
  font-size: 0.9375rem;
  font-weight: 550;
  text-decoration: none;
}

.cta:hover { filter: brightness(1.12); color: #ffffff; }

.plain { font-size: 0.9375rem; color: var(--ink-soft); }

/* -- the claim ----------------------------------------------------------- */

.claim { background: var(--sunken); border-block: 1px solid var(--rule); padding-block: 2.5rem; }

.claim__figures {
  display: grid;
  gap: 1.5rem 2.4rem;
  grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
  margin-bottom: 1.7rem;
}

.claim__figure { margin: 0; display: flex; flex-direction: column; gap: 0.3rem; }

.claim__figure strong {
  font-size: clamp(1.7rem, 4.2vw, 2.3rem);
  line-height: 1;
  font-weight: 600;
  letter-spacing: -0.03em;
  color: var(--series-1);
}

.claim__figure span { font-size: 0.875rem; color: var(--ink-soft); line-height: 1.45; }
.claim__detail { margin: 0; max-width: 44rem; font-size: 0.9375rem; color: var(--ink-soft); }

/* -- the interface ------------------------------------------------------- */

.session { padding-block: 3.2rem 1rem; }

h2 { margin: 0 0 1rem; font-size: 1.45rem; }
.session p, .evidence-section p, .install p, .covers p { margin: 0 0 1rem; }

.session__hint {
  font-size: 0.9375rem;
  color: var(--ink-soft);
  border-left: 2px solid var(--series-2);
  padding-left: 0.9rem;
}

.session__frame { margin: 2rem 0 0; padding-inline: var(--rail); }

.session__frame iframe {
  display: block;
  width: 100%;
  height: min(80vh, 960px);
  border: 1px solid var(--rule-strong);
  border-radius: 10px;
  background: var(--surface);
}

.session__fallback { margin: 0.9rem 0 0; font-size: 0.875rem; color: var(--ink-mute); }

/* -- evidence ------------------------------------------------------------ */

.evidence-section { padding-block: 3.2rem; }
.install { padding-block: 1rem 2rem; }
.covers { padding-block: 1rem 3.2rem; }

.evidence {
  width: 100%;
  border-collapse: collapse;
  margin: 1.5rem 0;
  font-size: 0.8125rem;
  background: var(--surface);
  border: 1px solid var(--rule);
}

.evidence th, .evidence td {
  text-align: left;
  vertical-align: top;
  padding: 0.55rem 0.8rem;
  border-bottom: 1px solid var(--rule);
  line-height: 1.5;
}

.evidence thead th {
  background: var(--sunken);
  font-weight: 600;
  color: var(--ink-soft);
  white-space: nowrap;
  font-size: 0.75rem;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

.evidence tbody tr:last-child td { border-bottom: 0; }
.evidence td:first-child { color: var(--ink-soft); width: 40%; min-width: 15rem; }
.evidence code { font-size: 1em; }

.aside {
  font-size: 0.9375rem;
  color: var(--ink-soft);
  border-left: 2px solid var(--rule-strong);
  padding-left: 0.9rem;
}

.aside strong { color: var(--ink); font-weight: 600; }

/* -- footer -------------------------------------------------------------- */

.foot {
  border-top: 1px solid var(--rule);
  padding-block: 1.8rem 2.5rem;
  font-size: 0.875rem;
  color: var(--ink-mute);
}

.foot p { margin: 0; }

@media (max-width: 46rem) {
  .evidence { display: block; overflow-x: auto; }
  .session__frame iframe { height: 72vh; }
}
`;

/**
 * Every file the site is made of, keyed by the name it is published under.
 *
 * Pure: it reads the engine and returns strings, so `test/docs.test.ts` can
 * call it without a filesystem and the entry point beside it is the only thing
 * that writes anything.
 */
export function buildSite(checks: readonly Check[] = measure()): Map<string, string> {
  // A row the gate is unhappy with has moved past what it claims, and the
  // rendered cell still reads like a result. Publishing it would put a figure
  // the gate has already rejected on a public page under the heading "what is
  // checked", which is worse than publishing nothing: the page would be
  // claiming the opposite of what the gate just said.
  const moved = checks.filter((check) => !check.ok);
  if (moved.length > 0) {
    const complaints = moved
      .map((check) => `${check.label}: ${check.complaint ?? check.result}`)
      .join('; ');
    throw new Error(`the gate is unhappy with ${moved.length} row(s): ${complaints}`);
  }
  return new Map([
    ['index.html', landing(checks, counts())],
    ['style.css', STYLE],
  ]);
}
