import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';
import { maximiseItem } from '../src/calibration/maximise.js';
import { normalGridRule } from '../src/core/quadrature.js';
import { scoreFunction } from '../src/estimation/likelihood.js';
import { estimateMle } from '../src/estimation/mle.js';
import { estimateWle } from '../src/estimation/wle.js';
import { normalPrior } from '../src/estimation/prior.js';
import { threePL, makeItem, twoPL } from '../src/models/item.js';
import { isPolytomous } from '../src/models/mixed.js';
import { hybridEstimator } from '../src/session/estimator.js';
import { syntheticMixedBank } from '../src/simulation/bank.js';
import type { ScoredResponse } from '../src/models/response.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

/**
 * Every documented option, and whether anything ever sets it.
 *
 * An optional property with a default is a claim: that the caller may choose,
 * and that choosing works. A property no call site anywhere ever supplies has
 * never had that claim tested — the code that reads it runs, so it is covered,
 * but only ever with the one value the author happened to pick. Two of the
 * defects found in this family of repositories were exactly that shape: a
 * documented parameter that only ever got its default, and was wrong for every
 * other value.
 *
 * So the gate is mechanical rather than a habit. It walks the source, collects
 * the optional properties of every exported interface, and requires that each
 * one be assigned somewhere — in the library, a test, the evidence gate, the
 * docs generator or the web interface. A new option arrives undocumented by
 * use, and this says so.
 *
 * Exported interfaces only. An optional field on a type a module keeps to
 * itself is a shape, not an offer, and the row table inside the misfit command
 * is the clearest example: it sets `ordered` on one of its own five rows, and
 * nobody outside that file has any business doing so.
 *
 * There are no exemptions, and the gate has no way to grant one. Adding that
 * machinery is the first thing that will be wanted the next time this fails,
 * and it is the wrong thing: an option nobody can be bothered to choose once
 * in a test is an option to delete.
 */

const SEARCH = ['src', 'test', 'scripts', 'docs', 'web'];

function sources(): string[] {
  const found: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir)) {
      if (entry === 'node_modules' || entry === 'dist' || entry === 'build') continue;
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) walk(path);
      else if (extname(path) === '.ts') found.push(path);
    }
  };
  for (const directory of SEARCH) walk(join(ROOT, directory));
  return found;
}

interface Declared {
  readonly type: string;
  readonly property: string;
  readonly file: string;
}

function survey(): { declared: Declared[]; assigned: Map<string, Set<string>> } {
  const declared: Declared[] = [];
  const assigned = new Map<string, Set<string>>();

  for (const file of sources()) {
    const tree = ts.createSourceFile(
      file,
      readFileSync(file, 'utf8'),
      ts.ScriptTarget.ES2022,
      true,
    );
    const visit = (node: ts.Node): void => {
      if (
        ts.isInterfaceDeclaration(node) &&
        node.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword) === true
      ) {
        for (const member of node.members) {
          if (
            ts.isPropertySignature(member) &&
            member.questionToken !== undefined &&
            member.name !== undefined &&
            ts.isIdentifier(member.name)
          ) {
            declared.push({ type: node.name.text, property: member.name.text, file });
          }
        }
      }
      if (ts.isPropertyAssignment(node) || ts.isShorthandPropertyAssignment(node)) {
        if (ts.isIdentifier(node.name)) {
          const files = assigned.get(node.name.text) ?? new Set<string>();
          files.add(file);
          assigned.set(node.name.text, files);
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(tree);
  }
  return { declared, assigned };
}

const { declared, assigned } = survey();

/** Sign changes of a pattern's score function across the default window. */
function signChanges(responses: readonly ScoredResponse[]): number {
  let previous = scoreFunction(responses, -5);
  let changes = 0;
  for (let step = 1; step <= 2000; step += 1) {
    const current = scoreFunction(responses, -5 + (10 * step) / 2000);
    if (previous * current < 0) changes += 1;
    previous = current;
  }
  return changes;
}

describe('every documented option is exercised somewhere', () => {
  it('surveyed the source it claims to have surveyed', () => {
    // A gate of the form "nothing is unexercised" passes trivially over an
    // empty survey, so the survey is asserted first: enough files, enough
    // optional properties, and three that are certainly there.
    expect(sources().length).toBeGreaterThan(120);
    expect(declared.length).toBeGreaterThan(80);
    const names = new Set(declared.map((entry) => `${entry.type}.${entry.property}`));
    for (const certain of ['MleOptions.scanSteps', 'MaximiseOptions.maxStep', 'BayesOptions.prior'])
      expect(names).toContain(certain);
  });

  it('leaves no optional property only ever at its default', () => {
    const unexercised = declared
      .filter((entry) => {
        const files = assigned.get(entry.property);
        if (files === undefined) return true;
        // An assignment inside the file that declares the property is usually
        // the default being written down rather than a caller choosing, so it
        // does not count on its own.
        return [...files].every((file) => file === entry.file);
      })
      .map((entry) => `${entry.type}.${entry.property}`);
    expect([...new Set(unexercised)].sort()).toEqual([]);
  });

});

describe('the options that had never been chosen', () => {
  it('scans the likelihood on the grid it is told to', () => {
    // A 3PL pattern whose score function changes sign twice inside the window.
    // Three hard items answered correctly and one easy one wrong: the
    // likelihood has an interior maximum near zero, and the score has the same
    // sign at both ends of the window, so a scan that looks only at the ends
    // finds no bracket at all and the estimate is reported at a boundary.
    // That is what the option is for — the 3PL likelihood is not guaranteed
    // unimodal, and how finely it is searched decides whether the maximum is
    // found or missed.
    const responses: ScoredResponse[] = [
      { item: makeItem('g1', threePL(1.5, 1.2, 0.2)), response: 1 },
      { item: makeItem('g2', threePL(1.5, 1.3, 0.2)), response: 1 },
      { item: makeItem('g3', threePL(1.5, 1.4, 0.2)), response: 1 },
      { item: makeItem('g4', threePL(1.5, -2.5, 0.2)), response: 0 },
    ];
    expect(signChanges(responses)).toBe(2);
    const fine = estimateMle(responses, { scanSteps: 512 });
    const coarse = estimateMle(responses, { scanSteps: 1 });
    expect(fine.boundary).toBe('none');
    expect(fine.theta).toBeCloseTo(0.1971, 3);
    expect(coarse.boundary).toBe('lower');
    expect(coarse.theta).toBe(-5);
    // The default of 64 is fine enough for this one, which is why nothing
    // noticed: the option only shows itself when it is made coarse.
    expect(estimateMle(responses).theta).toBeCloseTo(fine.theta, 6);
    expect(() => estimateMle(responses, { scanSteps: 0 })).toThrow(/positive integer/);
    expect(() => estimateMle(responses, { scanSteps: 2.5 })).toThrow(/positive integer/);
  });

  it('scans a weighted likelihood on the grid it is told to', () => {
    const responses: ScoredResponse[] = [
      { item: makeItem('w1', twoPL(1.2, -0.4)), response: 1 },
      { item: makeItem('w2', twoPL(0.9, 0.3)), response: 0 },
      { item: makeItem('w3', twoPL(1.5, 1.1)), response: 1 },
    ];
    const fine = estimateWle(responses, { scanSteps: 192 });
    const coarse = estimateWle(responses, { scanSteps: 2 });
    // Both find the same root here — the weighted likelihood of a mixed
    // pattern is well behaved — so what the option has to do is be accepted
    // and validated rather than change the answer.
    expect(coarse.theta).toBeCloseTo(fine.theta, 8);
    expect(() => estimateWle(responses, { scanSteps: -4 })).toThrow(/positive integer/);
  });

  it('damps a Newton step to the length it is given', () => {
    // One item's expected counts, started a long way from the optimum. A step
    // cap of a hundredth of a logit cannot cross that distance in few
    // iterations, so it takes more of them and arrives at the same place.
    const rule = normalGridRule(0, 1, 41);
    const answered = new Float64Array(rule.nodes.length).fill(50);
    const correct = Float64Array.from(rule.nodes, (theta) =>
      Math.round(50 / (1 + Math.exp(-1.4 * (theta - 0.3)))),
    );
    const options = {
      model: '2pl' as const,
      discriminationBounds: [0.1, 4] as const,
      difficultyBounds: [-4, 4] as const,
    };
    const generous = maximiseItem(
      { discrimination: 3.5, difficulty: -3 },
      rule,
      answered,
      correct,
      { ...options, maxIterations: 400 },
    );
    const damped = maximiseItem(
      { discrimination: 3.5, difficulty: -3 },
      rule,
      answered,
      correct,
      { ...options, maxStep: 0.01, maxIterations: 400 },
    );
    expect(damped.iterations).toBeGreaterThan(generous.iterations);
    expect(damped.stationary).toBe(true);
    expect(damped.item.discrimination).toBeCloseTo(generous.item.discrimination, 4);
    expect(damped.item.difficulty).toBeCloseTo(generous.item.difficulty, 4);
  });

  it('carries the prior the Bayesian phase was given', () => {
    // While the pattern is all-correct there is no finite likelihood estimate
    // and the prior is the whole of the answer, so a prior centred elsewhere
    // has to show up in it.
    const allCorrect: ScoredResponse[] = [
      { item: makeItem('h1', twoPL(1.1, 0)), response: 1 },
      { item: makeItem('h2', twoPL(1.3, 0.4)), response: 1 },
    ];
    const standard = hybridEstimator()(allCorrect);
    const shifted = hybridEstimator({ bayes: { prior: normalPrior(1.5, 0.5) } })(allCorrect);
    expect(standard.method).toBe('eap');
    expect(shifted.method).toBe('eap');
    expect(shifted.theta).toBeGreaterThan(standard.theta + 0.5);
  });

  it('carries the window the likelihood phase was given', () => {
    // Once the pattern is mixed the estimator switches to the weighted
    // likelihood, and a window narrower than the estimate pins it to the edge
    // and says so — which is the only way to see from outside that the
    // likelihood options reached it.
    const mixed: ScoredResponse[] = [
      { item: makeItem('k1', twoPL(1.4, 1.2)), response: 1 },
      { item: makeItem('k2', twoPL(1.4, 1.4)), response: 1 },
      { item: makeItem('k3', twoPL(1.4, -2.0)), response: 0 },
    ];
    const open = hybridEstimator()(mixed);
    const pinned = hybridEstimator({ likelihood: { min: -0.2, max: 0.2 } })(mixed);
    expect(open.method).toBe('wle');
    expect(open.boundary).toBe('none');
    expect(pinned.boundary).not.toBe('none');
    expect(Math.abs(pinned.theta)).toBeCloseTo(0.2, 12);
  });

  it('draws rubric items with the category counts it is given', () => {
    const narrow = syntheticMixedBank({
      size: 60,
      polytomousFraction: 0.5,
      seed: 11,
      categories: { min: 4, max: 4 },
    });
    const rubrics = narrow.filter(isPolytomous);
    expect(rubrics.length).toBeGreaterThan(20);
    for (const item of rubrics) expect(item.parameters.thresholds).toHaveLength(3);

    const wide = syntheticMixedBank({
      size: 60,
      polytomousFraction: 0.5,
      seed: 11,
      categories: { min: 2, max: 7 },
    });
    const spread = new Set(
      wide.filter(isPolytomous).map((item) => item.parameters.thresholds.length),
    );
    expect(spread.size).toBeGreaterThan(1);
    expect(Math.max(...spread)).toBeLessThanOrEqual(6);
    expect(Math.min(...spread)).toBeGreaterThanOrEqual(1);

    expect(() => syntheticMixedBank({ categories: { min: 1 } })).toThrow(/at least 2/);
    expect(() => syntheticMixedBank({ categories: { min: 5, max: 4 } })).toThrow(/at least/);
  });
});
