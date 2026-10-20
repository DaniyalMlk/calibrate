import { describe, expect, it } from 'vitest';
import { linspace } from '../src/core/numeric.js';
import { createRng } from '../src/core/random.js';
import { makeItem, twoPL, type Item } from '../src/models/item.js';
import { probabilityCorrect } from '../src/models/response.js';
import { estimateMle } from '../src/estimation/mle.js';
import { scanPersons } from '../src/estimation/person.js';
import {
  detectionRates,
  simulateCandidate,
  simulateSitting,
  type Aberrance,
} from '../src/simulation/aberrance.js';

const form: Item[] = linspace(-2, 2, 40).map((b, index) =>
  makeItem(`i${index}`, twoPL(0.7 + ((index * 7) % 40) / 40, b)),
);

/** Run a sitting through the scan and score the flags against the key. */
function run(kind: Aberrance, extent: number, share: number, seed = 77) {
  const sitting = simulateSitting({
    form,
    candidates: 1200,
    aberrance: { kind, extent },
    share,
    seed,
  });
  const scored = sitting.candidates.map((candidate) => ({
    id: candidate.id,
    responses: candidate.responses,
    theta: estimateMle(candidate.responses).theta,
  }));
  const flagged = scanPersons(scored);
  return detectionRates(
    sitting,
    flagged.map((row) => row.id),
  );
}

describe('simulateSitting', () => {
  it('affects about the share asked for, and records which', () => {
    const sitting = simulateSitting({
      form,
      candidates: 2000,
      aberrance: { kind: 'careless', extent: 0.2 },
      share: 0.25,
      seed: 5,
    });
    expect(sitting.candidates.length).toBe(2000);
    expect(sitting.affected.size / 2000).toBeGreaterThan(0.22);
    expect(sitting.affected.size / 2000).toBeLessThan(0.28);
    for (const candidate of sitting.candidates) {
      expect(candidate.kind === 'none').toBe(!sitting.affected.has(candidate.id));
    }
  });

  it('is deterministic for a seed, and different for another', () => {
    const options = {
      form,
      candidates: 50,
      aberrance: { kind: 'guessing' as const, extent: 0.25 },
      share: 0.3,
    };
    const first = simulateSitting({ ...options, seed: 11 });
    const again = simulateSitting({ ...options, seed: 11 });
    const other = simulateSitting({ ...options, seed: 12 });
    const pattern = (s: ReturnType<typeof simulateSitting>): string =>
      s.candidates.map((c) => c.responses.map((r) => r.response).join('')).join('|');
    expect(pattern(first)).toBe(pattern(again));
    expect(pattern(first)).not.toBe(pattern(other));
  });

  it('refuses an empty form, a bad share and a bad extent', () => {
    const base = { candidates: 10, aberrance: { kind: 'careless' as const, extent: 0.2 }, share: 0.5 };
    expect(() => simulateSitting({ ...base, form: [] })).toThrow(/form is empty/);
    expect(() => simulateSitting({ ...base, form, share: 1.4 })).toThrow(/share must be/);
    expect(() => simulateSitting({ ...base, form, candidates: 0 })).toThrow(/candidates must be/);
    expect(() =>
      simulateSitting({ ...base, form, aberrance: { kind: 'careless', extent: 0 } }),
    ).toThrow(/extent must be/);
  });
});

describe('simulateCandidate: where each departure lands', () => {
  const rng = () => createRng(2026);

  it('puts preknowledge on the items the candidate was least likely to have', () => {
    const theta = -0.5;
    const responses = simulateCandidate(form, theta, { kind: 'preknowledge', extent: 0.2 }, rng());
    const correct = responses.filter((r) => r.response === 1).map((r) => r.item);
    const hardest = [...form]
      .sort(
        (left, right) =>
          probabilityCorrect(left.parameters, theta) - probabilityCorrect(right.parameters, theta),
      )
      .slice(0, 8);
    // Every planted item is answered correctly, and they are the hard ones.
    for (const item of hardest) {
      expect(correct.some((candidate) => candidate.id === item.id)).toBe(true);
    }
  });

  it('puts carelessness on the items the candidate would almost certainly have had', () => {
    const theta = 1.2;
    const responses = simulateCandidate(form, theta, { kind: 'careless', extent: 0.2 }, rng());
    const easiest = [...form]
      .sort(
        (left, right) =>
          probabilityCorrect(right.parameters, theta) - probabilityCorrect(left.parameters, theta),
      )
      .slice(0, 8);
    for (const item of easiest) {
      const answered = responses.find((response) => response.item.id === item.id);
      expect(answered?.response).toBe(0);
    }
  });

  it('leaves an unaffected candidate exactly as the model generated them', () => {
    const plain = simulateCandidate(form, 0.3, { kind: 'none', extent: 1 }, createRng(8));
    const again = simulateCandidate(form, 0.3, { kind: 'none', extent: 1 }, createRng(8));
    expect(plain.map((r) => r.response)).toEqual(again.map((r) => r.response));
  });

  it('guesses at roughly the reciprocal of the options given', () => {
    // Over a long tail of guessed items the success rate is 1/options,
    // whatever the candidate's ability.
    const long = linspace(-2, 2, 400).map((b, i) => makeItem(`g${i}`, twoPL(1, b)));
    const responses = simulateCandidate(
      long,
      -3,
      { kind: 'guessing', extent: 1, options: 4 },
      createRng(17),
    );
    const rate = responses.filter((r) => r.response === 1).length / responses.length;
    expect(rate).toBeGreaterThan(0.2);
    expect(rate).toBeLessThan(0.3);
  });
});

describe('detectionRates', () => {
  it('reports both rates against the key', () => {
    const sitting = simulateSitting({
      form,
      candidates: 100,
      aberrance: { kind: 'careless', extent: 0.2 },
      share: 0.5,
      seed: 3,
    });
    const everyone = sitting.candidates.map((candidate) => candidate.id);
    const all = detectionRates(sitting, everyone);
    // Flagging everybody detects everybody, and that is the point of carrying
    // the second number.
    expect(all.sensitivity).toBe(1);
    expect(all.falsePositiveRate).toBe(1);
    const none = detectionRates(sitting, []);
    expect(none.sensitivity).toBe(0);
    expect(none.falsePositiveRate).toBe(0);
  });
});

describe('the scan against a planted answer key', () => {
  it('holds its false positive rate when nothing was planted', () => {
    const rates = run('none', 1, 0);
    expect(rates.affected).toBe(0);
    expect(rates.falsePositiveRate).toBeGreaterThan(0.02);
    expect(rates.falsePositiveRate).toBeLessThan(0.09);
  });

  it('catches most preknowledge on a fifth of the form', () => {
    const rates = run('preknowledge', 0.2, 0.2);
    expect(rates.sensitivity).toBeGreaterThan(0.85);
    expect(rates.falsePositiveRate).toBeLessThan(0.09);
  });

  it('catches most carelessness on a fifth of the form', () => {
    const rates = run('careless', 0.2, 0.2);
    expect(rates.sensitivity).toBeGreaterThan(0.8);
  });

  it('catches far less guessing, which is the honest result', () => {
    // Guessing runs to the end of a form ordered by difficulty, so it lands on
    // the items the candidate was least likely to get right anyway. One in
    // four on the hardest items is close to what the model already expects of
    // a weak candidate there, so there is little for a residual to find. The
    // bound is upper as well as lower: a change that pushed this up to the
    // preknowledge rate would mean the generator had stopped putting the
    // guessing where it says it does.
    const rates = run('guessing', 0.2, 0.2);
    expect(rates.sensitivity).toBeGreaterThan(0.15);
    expect(rates.sensitivity).toBeLessThan(0.45);
  });

  it('finds more as the departure covers more of the form', () => {
    const small = run('careless', 0.1, 0.2);
    const large = run('careless', 0.3, 0.2);
    expect(large.sensitivity).toBeGreaterThan(small.sensitivity);
  });
});
