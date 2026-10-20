import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { measure, renderTable, type Check } from '../scripts/evidence.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const README = readFileSync(join(ROOT, 'README.md'), 'utf8');

/**
 * The checks are measured once for the whole file.
 *
 * Several of them calibrate 2,000-respondent matrices, so recomputing them per
 * assertion would turn a twenty-five second gate into a several-minute one.
 */
const checks: readonly Check[] = measure();

/** The rows of the README's evidence table, exactly as written. */
function readmeRows(): readonly string[] {
  const section = /## What is checked, and against what\n([\s\S]*?)\n## /.exec(README);
  if (section === null) throw new Error('the README has no evidence section');
  return (section[1] as string)
    .split('\n')
    .filter((line) => line.startsWith('|'))
    .map((line) => line.trim());
}

describe('the evidence gate', () => {
  it('measures something, and every check passes', () => {
    expect(checks.length).toBeGreaterThan(5);
    for (const check of checks) {
      expect(check.ok, `${check.label}: ${check.complaint ?? ''}`).toBe(true);
    }
  });

  it('is run by CI, or the table is pinned to nothing', () => {
    // A gate nobody runs is a comment. The workflow has to invoke it, and the
    // manifest has to define the script the workflow invokes.
    const workflow = readFileSync(join(ROOT, '.github/workflows/ci.yml'), 'utf8');
    expect(workflow).toContain('npm run evidence');
    const manifest = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')) as {
      scripts: Record<string, string>;
    };
    expect(manifest.scripts['evidence']).toBeDefined();
  });
});

describe('the README table against the gate', () => {
  const rows = readmeRows();

  it('has the header and one row per check', () => {
    expect(rows[0]).toBe('| Check | Result |');
    expect(rows[1]).toBe('| --- | --- |');
    expect(rows.length - 2).toBe(checks.length);
  });

  it('prints every check as the gate renders it, row for row', () => {
    // The whole rendered row, in order. Checking the figures alone would pass
    // a table whose numbers had been shuffled between rows — each one would
    // still appear somewhere on the page — and a figure under the wrong claim
    // is worse than a figure that is simply out of date.
    expect(rows.join('\n')).toBe(renderTable(checks));
  });

  it('carries the label of every check', () => {
    for (const check of checks) {
      expect(README).toContain(check.label);
    }
  });

  it('quotes no figure the gate does not produce', () => {
    // The forward direction: every number written into a row has to come from
    // the gate's own output for that row, so a hand-edited figure fails even
    // if it looks plausible.
    for (const [index, check] of checks.entries()) {
      const row = rows[index + 2] as string;
      for (const figure of row.matchAll(/\d+(?:[.,]\d+)*(?:e-?\d+)?/g)) {
        expect(`${check.label} ${check.result}`).toContain(figure[0]);
      }
    }
  });
});

describe('the prose around the table', () => {
  it('does not still claim fourteen digits for a reduction that is exact', () => {
    // The two-category reduction was described as agreeing "to fourteen
    // digits" long after it was exact. The claim is quoted once now, in the
    // past tense, as the example of why an identity should not be published
    // as a tolerance.
    expect(README).toContain('it agrees to all of them');
    const loose = [...README.matchAll(/agreeing to fourteen digits/g)];
    expect(loose.length).toBe(0);
  });

  it('quotes the sufficiency contrast the gate measured', () => {
    const sufficiency = checks.find((check) => check.label.includes('same total score'));
    expect(sufficiency).toBeDefined();
    const logits = /spans (\d+\.\d+) logits/.exec((sufficiency as Check).result);
    expect(logits).not.toBeNull();
    expect(README).toContain(`${(logits as RegExpExecArray)[1]} logits`);
  });

  it('quotes the joint-bias figures the gate measured', () => {
    const tenItem = checks.find((check) => check.label.includes('10-item test'));
    expect(tenItem).toBeDefined();
    const numbers = [...(tenItem as Check).result.matchAll(/\d\.\d{4}/g)].map((m) => m[0]);
    expect(numbers.length).toBe(3);
    for (const number of numbers) {
      expect(README).toContain(number);
    }
  });
});
