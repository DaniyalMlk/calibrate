import { html } from './chart/svg.js';
import { DEFAULT_CONFIGURATION, Store, type PolicyName } from './state.js';
import { mountBank } from './views/bank.js';
import { mountInformation } from './views/information.js';
import { mountPosterior } from './views/posterior.js';
import { mountSession, mountTranscript } from './views/session.js';

const POLICIES: readonly { readonly value: PolicyName; readonly label: string }[] = [
  { value: 'max-information', label: 'Maximum information' },
  { value: 'kullback-leibler', label: 'Kullback–Leibler' },
  { value: 'randomesque-5', label: 'Randomesque, top 5' },
  { value: 'balanced', label: 'Content balanced' },
];

function field(label: string, control: HTMLElement): HTMLElement {
  const wrap = html('div', { className: 'field' });
  const id = `field-${label.toLowerCase().replace(/[^a-z]+/g, '-')}`;
  control.id = id;
  const labelEl = html('label', { className: 'field__label', text: label });
  labelEl.setAttribute('for', id);
  wrap.append(labelEl, control);
  return wrap;
}

/**
 * One control row above everything it scopes.
 *
 * Every panel below re-renders from the same configuration, so the numbers in
 * one panel always describe the same session as the numbers in the next. A
 * per-panel control would let two panels disagree about what they are showing,
 * which is the failure a reader cannot detect by looking.
 */
function mountControls(root: HTMLElement, store: Store): void {
  const policy = html('select', { className: 'field__control' });
  for (const option of POLICIES) {
    const element = html('option', { text: option.label, attributes: { value: option.value } });
    policy.append(element);
  }
  policy.value = DEFAULT_CONFIGURATION.policy;
  policy.addEventListener('change', () => {
    store.reconfigure({ policy: policy.value as PolicyName });
  });

  const target = numberInput(DEFAULT_CONFIGURATION.target, 0.05, 0.1, 1, (value) =>
    store.reconfigure({ target: value }),
  );
  const maximum = numberInput(DEFAULT_CONFIGURATION.maximum, 1, 5, 80, (value) =>
    store.reconfigure({ maximum: Math.round(value) }),
  );
  const trueTheta = numberInput(DEFAULT_CONFIGURATION.trueTheta, 0.1, -3, 3, (value) =>
    store.reconfigure({ trueTheta: value }),
  );
  const rubricShare = numberInput(DEFAULT_CONFIGURATION.polytomousFraction, 0.05, 0, 1, (value) =>
    store.reconfigure({ polytomousFraction: value }),
  );

  root.append(
    field('Selection policy', policy),
    field('Target SE', target),
    field('Max items', maximum),
    field('Simulated ability', trueTheta),
    field('Rubric share', rubricShare),
  );
}

function numberInput(
  initial: number,
  step: number,
  min: number,
  max: number,
  onChange: (value: number) => void,
): HTMLInputElement {
  const input = html('input', {
    className: 'field__control',
    attributes: { type: 'number', step, min, max, value: String(initial) },
  });
  input.addEventListener('change', () => {
    const value = Number(input.value);
    if (!Number.isFinite(value) || value < min || value > max) {
      input.value = String(initial);
      return;
    }
    onChange(value);
  });
  return input;
}

function mountThemeToggle(root: HTMLElement): void {
  const button = html('button', {
    className: 'button button--ghost',
    attributes: { type: 'button' },
  });

  const apply = (theme: 'dark' | 'light'): void => {
    document.documentElement.dataset.theme = theme;
    button.textContent = theme === 'dark' ? 'Light theme' : 'Dark theme';
  };

  apply('dark');
  button.addEventListener('click', () => {
    apply(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  });
  root.append(button);
}

function main(): void {
  const store = new Store();

  const controls = document.querySelector('#controls');
  const sessionPanel = document.querySelector('#session');
  const posteriorPanel = document.querySelector('#posterior');
  const transcriptPanel = document.querySelector('#transcript');
  const informationPanel = document.querySelector('#information');
  const bankPanel = document.querySelector('#bank');
  const themeSlot = document.querySelector('#theme');

  if (
    !(controls instanceof HTMLElement) ||
    !(sessionPanel instanceof HTMLElement) ||
    !(posteriorPanel instanceof HTMLElement) ||
    !(transcriptPanel instanceof HTMLElement) ||
    !(informationPanel instanceof HTMLElement) ||
    !(bankPanel instanceof HTMLElement) ||
    !(themeSlot instanceof HTMLElement)
  ) {
    throw new Error('main: the document is missing one of its mount points');
  }

  mountThemeToggle(themeSlot);
  mountControls(controls, store);
  mountSession(sessionPanel, store);
  mountPosterior(posteriorPanel, store);
  mountInformation(informationPanel, store);
  mountBank(bankPanel, store);
  mountTranscript(transcriptPanel, store);
}

main();
