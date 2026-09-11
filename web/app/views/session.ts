import {
  describeModel,
  describePolytomousModel,
  isPolytomous,
  maximumScoreOf,
  type AnyItem,
} from '../../../src/index.js';
import { describeItemParameters, fixed, interval, signed } from '../format.js';
import { html } from '../chart/svg.js';
import type { Store, ViewModel } from '../state.js';

/**
 * The session panel: the item awaiting a response, the running estimate, and
 * the transcript.
 *
 * The estimate is the one hero figure on the page. Everything else here is a
 * supporting readout, and the transcript is a table rather than a chart because
 * what a reader wants from it is to check a specific row.
 */
export function mountSession(root: HTMLElement, store: Store): void {
  const heroValue = html('div', { className: 'hero__value', text: '0.00' });
  const heroUnit = html('span', { className: 'hero__unit', text: 'ability, logits' });
  const status = html('span', { className: 'badge badge--neutral' });
  const statusDot = html('span', { className: 'badge__dot' });
  const statusText = html('span', { text: 'Ready' });
  status.append(statusDot, statusText);

  const readout = html('div', { className: 'readout' });

  const itemHost = html('div');
  const answerRow = html('div', { className: 'answer-row' });

  const simulateOne = html('button', {
    className: 'button',
    text: 'Simulate one',
    attributes: { type: 'button' },
  });
  const simulateRest = html('button', {
    className: 'button',
    text: 'Simulate to stop',
    attributes: { type: 'button' },
  });
  const restart = html('button', {
    className: 'button button--ghost',
    text: 'New session',
    attributes: { type: 'button' },
  });
  simulateOne.addEventListener('click', () => store.simulateOne());
  simulateRest.addEventListener('click', () => store.simulateRest());
  restart.addEventListener('click', () => store.restart());

  const simulateRow = html('div', { className: 'answer-row' });
  simulateRow.append(simulateOne, simulateRest);

  const head = html('div', { className: 'panel__head' });
  const heading = html('div');
  heading.append(html('h2', { className: 'panel__title', text: 'Session' }));
  head.append(heading, status);

  const hero = html('div', { className: 'hero' });
  hero.append(heroValue, heroUnit);

  root.append(
    head,
    hero,
    readout,
    html('div', {
      className: 'panel__body',
      children: [itemHost, answerRow, simulateRow, restart],
    }),
  );

  // The pending item decides what a valid answer is, so the keyboard map has to
  // be read at the moment a key is pressed rather than bound once. Pressing `2`
  // while a two-category item is pending is not a score of 2 to be clamped — it
  // is a key with no meaning here, and doing nothing is the honest response.
  let pending: AnyItem | null = null;

  document.addEventListener('keydown', (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || pending === null) {
      return;
    }
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement) {
      return;
    }
    if (event.key === 's' || event.key === 'S') {
      store.simulateOne();
      return;
    }

    const top = maximumScoreOf(pending);
    // Digits score directly, on either format: `0` and `1` are the two
    // categories of a dichotomous item as much as they are the bottom two of a
    // rubric. F and J stay bound to wrong and right, which is the muscle memory
    // of anyone who has driven this panel, but only where those words mean
    // something.
    if (/^[0-9]$/.test(event.key)) {
      const category = Number(event.key);
      if (category <= top) {
        store.answer(category);
      }
      return;
    }
    if (top === 1 && (event.key === 'j' || event.key === 'J')) {
      store.answer(1);
    } else if (top === 1 && (event.key === 'f' || event.key === 'F')) {
      store.answer(0);
    }
  });

  store.subscribe((model) => {
    pending = model.current;
    answerRow.replaceChildren(...scoreButtons(model.current, store));
    render(model, {
      heroValue,
      status,
      statusText,
      readout,
      itemHost,
      simulateOne,
      simulateRest,
    });
  });
}

/**
 * The score entry for the pending item: one button per category it can award.
 *
 * A dichotomous item keeps the two labelled buttons it has always had, because
 * "Correct" and "Incorrect" are what those two categories mean and numbering
 * them 0 and 1 would make the panel harder to read in exchange for uniformity
 * nobody asked for. A rubric item gets a button per level, labelled by the
 * points it awards — the number that goes on the transcript and into the raw
 * score, not an ordinal position.
 */
function scoreButtons(item: AnyItem | null, store: Store): HTMLElement[] {
  if (item === null) {
    return [];
  }

  const top = maximumScoreOf(item);
  if (top === 1) {
    const wrong = html('button', { className: 'button', attributes: { type: 'button' } });
    wrong.append(html('span', { text: 'Incorrect' }), html('span', { className: 'kbd', text: 'F' }));
    wrong.addEventListener('click', () => store.answer(0));

    const right = html('button', { className: 'button', attributes: { type: 'button' } });
    right.append(html('span', { text: 'Correct' }), html('span', { className: 'kbd', text: 'J' }));
    right.addEventListener('click', () => store.answer(1));
    return [wrong, right];
  }

  const buttons: HTMLElement[] = [];
  for (let category = 0; category <= top; category += 1) {
    const button = html('button', {
      className: 'button button--score',
      attributes: {
        type: 'button',
        'aria-label': `Award ${category} of ${top} points`,
      },
    });
    button.append(
      html('span', { className: 'button__score', text: String(category) }),
      html('span', { className: 'kbd', text: String(category) }),
    );
    button.addEventListener('click', () => store.answer(category));
    buttons.push(button);
  }
  return buttons;
}

/**
 * The transcript, as its own panel.
 *
 * It lives in the wide column rather than beside the item card: six numeric
 * columns in a 360px aside would need a horizontal scrollbar inside a card,
 * and a reader who has to scroll a table sideways to compare two rows cannot
 * compare them at all.
 */
export function mountTranscript(root: HTMLElement, store: Store): void {
  const heading = html('div');
  heading.append(
    html('h2', { className: 'panel__title', text: 'Transcript' }),
    html('p', {
      className: 'panel__note',
      text:
        'Every administered item, the response, and the estimate it produced. Seeded and ' +
        'replayable: the same configuration and the same answers give the same rows.',
    }),
  );

  const count = html('span', { className: 'badge badge--neutral' });
  const head = html('div', { className: 'panel__head' });
  head.append(heading, count);

  const host = html('div', { className: 'table-wrap' });
  root.append(head, host);

  store.subscribe((model) => {
    const administered = model.snapshot.transcript.length;
    count.textContent = `${administered} ${administered === 1 ? 'item' : 'items'}`;
    host.replaceChildren(transcriptTable(model));
  });
}

interface Parts {
  readonly heroValue: HTMLElement;
  readonly status: HTMLElement;
  readonly statusText: HTMLElement;
  readonly readout: HTMLElement;
  readonly itemHost: HTMLElement;
  readonly simulateOne: HTMLButtonElement;
  readonly simulateRest: HTMLButtonElement;
}

function render(model: ViewModel, parts: Parts): void {
  const { snapshot, posterior, intervals, configuration } = model;
  const administered = snapshot.transcript.length;

  // The headline is the estimate the engine reports — the same number the last
  // transcript row shows. The posterior mean sits beside it as its own labelled
  // readout rather than in the hero: the two estimators disagree slightly by
  // construction, and a headline that quietly showed one while the transcript
  // showed the other would read as an arithmetic error.
  parts.heroValue.textContent = fixed(snapshot.theta, 2);

  const meetsTarget = administered > 0 && snapshot.standardError <= configuration.target;
  parts.status.className = `badge badge--${
    model.finished ? (meetsTarget ? 'good' : 'warn') : 'neutral'
  }`;
  parts.statusText.textContent = model.finished
    ? meetsTarget
      ? 'Target met'
      : 'Stopped short'
    : `Item ${administered + 1}`;

  parts.readout.replaceChildren(
    cell('Standard error', fixed(snapshot.standardError, 3)),
    cell('Items', String(administered)),
    // Points rather than items, because on a mixed form those are two different
    // numbers and the raw score a candidate is reported against is this one.
    cell('Raw score', `${model.score.earned} / ${model.score.available}`),
    cell('Estimator', administered === 0 ? 'prior' : snapshot.method),
    cell('Posterior mean', fixed(posterior.mean, 3)),
    cell('95% credible', interval(intervals.outer.lower, intervals.outer.upper)),
  );

  // The item card, or the closing summary once the session has stopped.
  if (model.current !== null) {
    const item = model.current;
    const rubric = isPolytomous(item);
    const card = html('div', { className: 'item-card' });
    const identity = html('div', { className: 'item-card__head' });
    identity.append(html('span', { className: 'item-card__id', text: item.id }));
    if (rubric) {
      identity.append(
        html('span', {
          className: 'tag tag--rubric',
          text: `rubric, 0–${maximumScoreOf(item)}`,
        }),
      );
    }
    card.append(
      identity,
      html('div', {
        className: 'item-card__params',
        children: [
          html('span', {
            text: rubric
              ? describePolytomousModel(item.parameters)
              : describeModel(item.parameters),
          }),
          html('span', { text: describeItemParameters(item.parameters) }),
        ],
      }),
    );
    if (item.domain !== undefined) {
      card.append(html('div', { className: 'item-card__params', text: item.domain }));
    }
    parts.itemHost.replaceChildren(card);
  } else {
    const card = html('div', { className: 'item-card' });
    const error = snapshot.theta - configuration.trueTheta;
    card.append(
      html('div', {
        className: 'item-card__id',
        text: administered === 0 ? 'No items administered' : `Session complete`,
      }),
      html('div', {
        className: 'item-card__params',
        text:
          administered === 0
            ? 'The bank has no eligible items for this configuration.'
            : (snapshot.stopReason?.detail ?? 'The bank ran out of eligible items.'),
      }),
    );
    if (administered > 0) {
      card.append(
        html('div', {
          className: 'item-card__params',
          text: `true θ ${signed(configuration.trueTheta)}   error ${signed(error)}   ${fixed(
            Math.abs(error) / (snapshot.standardError || Infinity),
            2,
          )} SE`,
        }),
      );
    }
    parts.itemHost.replaceChildren(card);
  }

  const idle = model.current === null;
  parts.simulateOne.disabled = idle;
  parts.simulateRest.disabled = idle;
}

function cell(label: string, value: string): HTMLElement {
  const wrap = html('div');
  wrap.append(
    html('div', { className: 'readout__label', text: label }),
    html('div', { className: 'readout__value', text: value }),
  );
  return wrap;
}

function transcriptTable(model: ViewModel): HTMLElement {
  const table = html('table');
  const head = html('thead');
  const headRow = html('tr');
  const columns = [
    ['#', false],
    ['item', false],
    ['domain', false],
    ['format', false],
    ['score', false],
    ['θ', true],
    ['SE', true],
    ['estimator', false],
  ] as const;
  for (const [label, numeric] of columns) {
    headRow.append(html('th', { text: label, ...(numeric ? { className: 'num' } : {}) }));
  }
  head.append(headRow);

  const body = html('tbody');
  if (model.snapshot.transcript.length === 0) {
    const row = html('tr');
    row.append(
      html('td', {
        text: 'No responses yet — answer the first item, or simulate one.',
        attributes: { colspan: columns.length },
      }),
    );
    body.append(row);
  }

  const domains = new Map(model.bank.map((item) => [item.id, item.domain ?? '']));
  for (const entry of model.snapshot.transcript) {
    const rubric = entry.maximumScore > 1;
    const row = html('tr');
    row.append(
      html('td', { className: 'num', text: String(entry.position) }),
      html('td', { className: 'mono', text: entry.itemId }),
      html('td', { text: domains.get(entry.itemId) ?? '' }),
      html('td', { text: rubric ? `rubric 0–${entry.maximumScore}` : 'dichotomous' }),
      // A rubric level is not a right/wrong verdict and must not be rendered as
      // one: "2 / 4" is the fact, where "wrong" — which is what a dichotomous
      // reading of a non-1 category produces — is false.
      html('td', {
        className: rubric ? 'num' : '',
        text: rubric
          ? `${entry.response} / ${entry.maximumScore}`
          : entry.response === 1
            ? 'right'
            : 'wrong',
      }),
      html('td', { className: 'num', text: fixed(entry.thetaAfter, 3) }),
      html('td', { className: 'num', text: fixed(entry.standardErrorAfter, 3) }),
      html('td', { text: entry.method }),
    );
    body.append(row);
  }

  table.append(head, body);
  return table;
}
