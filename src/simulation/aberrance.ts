/**
 * Candidates who did not answer the way the model says they would.
 *
 * A person-fit statistic can only be shown to work on data where the answer is
 * known, so this generates sittings in which some candidates are deliberately
 * made to misfit and the rest are not — and records which were which.
 *
 * The three behaviours below are not three settings of one knob. They misfit
 * in different places on the form, which is exactly why a statistic can be good
 * at one and poor at another, and why reporting a single detection rate over a
 * mixture of them would say less than it appears to.
 */

import { createRng, type Rng } from '../core/random.js';
import type { Item } from '../models/item.js';
import { probabilityCorrect } from '../models/response.js';
import type { ScoredResponse } from '../models/response.js';

/** How a candidate departs from the model. */
export type Aberrance =
  /** Answers as the model says, at their own ability. */
  | 'none'
  /**
   * Knows the answers to a run of items in advance.
   *
   * The classic *spuriously high* pattern: hard items answered correctly that
   * the candidate's ability does not support, leaving the rest of the form
   * looking too difficult for the score.
   */
  | 'preknowledge'
  /**
   * Answers at random from a fixed number of options.
   *
   * A candidate who has stopped engaging, or who ran out of time and filled in
   * the rest. Shows up at the easy end, where the model expects near-certain
   * success and gets a coin flip.
   */
  | 'guessing'
  /**
   * Slips on items well within their reach.
   *
   * Carelessness, or a disturbance partway through. *Spuriously low*: the
   * misfit is concentrated where the model is most confident, which is where a
   * residual costs the most.
   */
  | 'careless';

export interface AberranceSpec {
  /** Which departure to apply. */
  readonly kind: Aberrance;
  /** Share of the form affected, between 0 and 1. */
  readonly extent: number;
  /** Options to guess from, for `guessing`. */
  readonly options?: number;
}

export interface SimulatedCandidate {
  readonly id: string;
  /** The ability the responses were generated at. */
  readonly theta: number;
  readonly responses: ScoredResponse[];
  /** What was done to this candidate — `none` for the honest majority. */
  readonly kind: Aberrance;
}

export interface SittingOptions {
  readonly form: readonly Item[];
  /** How many candidates to generate. */
  readonly candidates: number;
  /** What to do to the affected ones. */
  readonly aberrance: AberranceSpec;
  /** Share of candidates affected. */
  readonly share: number;
  readonly seed?: number;
  /** Mean and standard deviation of the ability distribution. */
  readonly population?: { readonly mean: number; readonly sd: number };
}

export interface Sitting {
  readonly candidates: readonly SimulatedCandidate[];
  /** Ids of the candidates something was done to, which is the answer key. */
  readonly affected: ReadonlySet<string>;
}

/** The indices a departure touches, drawn without replacement. */
function affectedItems(
  kind: Aberrance,
  form: readonly Item[],
  theta: number,
  extent: number,
): readonly number[] {
  const count = Math.max(1, Math.round(form.length * extent));
  const indices = form.map((_, index) => index);

  if (kind === 'preknowledge') {
    // The items the candidate is least likely to get right unaided: knowing
    // the answer to something they would have answered anyway leaves no trace.
    return indices
      .sort(
        (left, right) =>
          probabilityCorrect((form[left] as Item).parameters, theta) -
          probabilityCorrect((form[right] as Item).parameters, theta),
      )
      .slice(0, count);
  }
  if (kind === 'careless') {
    // The mirror image: slipping on what they would almost certainly have had.
    return indices
      .sort(
        (left, right) =>
          probabilityCorrect((form[right] as Item).parameters, theta) -
          probabilityCorrect((form[left] as Item).parameters, theta),
      )
      .slice(0, count);
  }
  // Guessing runs to the end of the form, as running out of time does.
  return indices.slice(form.length - count);
}

/**
 * One candidate's responses, with a departure applied.
 *
 * The honest responses are generated first and then overwritten, so that the
 * departure is visibly a change to a pattern the model did produce rather than
 * a different generating process wearing the same name.
 */
export function simulateCandidate(
  form: readonly Item[],
  theta: number,
  spec: AberranceSpec,
  rng: Rng,
): ScoredResponse[] {
  const responses: ScoredResponse[] = form.map((item) => ({
    item,
    response: rng.next() < probabilityCorrect(item.parameters, theta) ? 1 : 0,
  }));
  if (spec.kind === 'none') return responses;

  const options = spec.options ?? 4;
  for (const index of affectedItems(spec.kind, form, theta, spec.extent)) {
    const item = form[index] as Item;
    const response: 0 | 1 =
      spec.kind === 'preknowledge'
        ? 1
        : spec.kind === 'careless'
          ? 0
          : rng.next() < 1 / options
            ? 1
            : 0;
    responses[index] = { item, response };
  }
  return responses;
}

/** A sitting of candidates, a known share of whom did not answer as the model says. */
export function simulateSitting(options: SittingOptions): Sitting {
  const { form, candidates, aberrance, share } = options;
  if (form.length === 0) throw new RangeError('simulateSitting: the form is empty');
  if (!(candidates > 0)) {
    throw new RangeError(`simulateSitting: candidates must be positive, received ${candidates}`);
  }
  if (!(share >= 0 && share <= 1)) {
    throw new RangeError(`simulateSitting: share must be in [0, 1], received ${share}`);
  }
  if (!(aberrance.extent > 0 && aberrance.extent <= 1)) {
    throw new RangeError(
      `simulateSitting: extent must be in (0, 1], received ${aberrance.extent}`,
    );
  }

  const rng = createRng(options.seed ?? 20260101);
  const centre = options.population?.mean ?? 0;
  const spread = options.population?.sd ?? 1;
  const affected = new Set<string>();
  const people: SimulatedCandidate[] = [];

  for (let index = 0; index < candidates; index += 1) {
    const id = `p-${index}`;
    const theta = centre + spread * rng.nextNormal();
    // Drawn per candidate rather than by taking the first `share * n`, so the
    // affected ones are not correlated with anything else about the order.
    const isAffected = rng.next() < share;
    const spec: AberranceSpec = isAffected ? aberrance : { kind: 'none', extent: 1 };
    if (isAffected) affected.add(id);
    people.push({
      id,
      theta,
      responses: simulateCandidate(form, theta, spec, rng),
      kind: spec.kind,
    });
  }

  return { candidates: people, affected };
}

export interface DetectionRates {
  /** Affected candidates the scan flagged, as a share of the affected. */
  readonly sensitivity: number;
  /** Unaffected candidates the scan flagged, as a share of the unaffected. */
  readonly falsePositiveRate: number;
  readonly flagged: number;
  readonly affected: number;
  readonly unaffected: number;
}

/**
 * Score a scan's flags against the key.
 *
 * Both rates, always. A detection rate on its own is not a measurement: a scan
 * that flags everybody detects everybody, and the false positive rate is what
 * says whether the first number cost anything.
 */
export function detectionRates(
  sitting: Sitting,
  flaggedIds: Iterable<string>,
): DetectionRates {
  const flagged = new Set(flaggedIds);
  let truePositives = 0;
  let falsePositives = 0;
  for (const candidate of sitting.candidates) {
    if (!flagged.has(candidate.id)) continue;
    if (sitting.affected.has(candidate.id)) truePositives += 1;
    else falsePositives += 1;
  }
  const affected = sitting.affected.size;
  const unaffected = sitting.candidates.length - affected;
  return {
    sensitivity: affected > 0 ? truePositives / affected : Number.NaN,
    falsePositiveRate: unaffected > 0 ? falsePositives / unaffected : Number.NaN,
    flagged: flagged.size,
    affected,
    unaffected,
  };
}
