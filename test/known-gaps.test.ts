import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

/**
 * The README's known gaps, and whether each one is still a gap.
 *
 * What an engine does not do is harder to find out than what it does, and
 * reading the tests will not tell you: a suite says nothing about the cases
 * nobody wrote. So the README carries the list, and this file pairs every line
 * of it with a predicate over the package's own surface.
 *
 * The failure mode worth guarding against is not an incomplete list but a stale
 * one. A line saying something is missing after it has been built is worse than
 * no line at all, because it is believed. So the gate fails five ways: a line
 * with no predicate, a predicate with no line, the list shrinking, a bullet
 * going thin, and a predicate that no longer holds because the thing it says is
 * missing has been built. Closing a gap therefore breaks this file, which is the
 * point -- the line has to come out deliberately rather than be left behind.
 *
 * Most of the predicates are deliberately crude. They ask whether anything in
 * the surface is *named* for the missing concept, which is a weaker question
 * than whether the concept is modelled, and weaker in the safe direction: a
 * name can appear without the behaviour, and then the line comes out and is
 * replaced by something sharper. What cannot happen is the behaviour appearing
 * while the line stays. Where a crude question would be the wrong one, the
 * predicate reads the declaration instead -- the members of one interface, or
 * the arms of one union -- and those are the sharper ones.
 *
 * There is no exemption mechanism, deliberately.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const README = join(ROOT, 'README.md');

function sources(): string[] {
  const found: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir)) {
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) walk(path);
      else if (extname(path) === '.ts') found.push(path);
    }
  };
  walk(join(ROOT, 'src'));
  return found;
}

interface Surface {
  /** Every name the source declares, at any nesting depth. */
  readonly names: ReadonlySet<string>;
  /** The property and method names of each interface, by interface name. */
  readonly members: ReadonlyMap<string, ReadonlySet<string>>;
  /** The string-literal arms of each union type alias, by alias name. */
  readonly unions: ReadonlyMap<string, ReadonlySet<string>>;
  /** The written-out right-hand side of each type alias, by alias name. */
  readonly aliases: ReadonlyMap<string, string>;
}

/**
 * The inventory every predicate below reads, built by parsing rather than
 * grepping.
 *
 * Parsed, because a grep over the source finds the concept in the prose that
 * explains why it is absent: `testlet` and `multidimensional` both appear in
 * this repository's comments, and a grep-backed predicate would read those as
 * the feature arriving. Only declarations count here, so a paragraph about what
 * is missing cannot make the gate think it is present.
 *
 * Deliberately inclusive: declarations at any depth, exported or not, including
 * local variables and parameters. An over-wide inventory errs towards finding a
 * name and so towards failing -- which forces the README line out and a human
 * to look -- while a narrow one would quietly keep a line that had stopped
 * being true.
 */
function surface(): Surface {
  const names = new Set<string>();
  const members = new Map<string, Set<string>>();
  const unions = new Map<string, Set<string>>();
  const aliases = new Map<string, string>();

  const memberNames = (nodes: ts.NodeArray<ts.TypeElement>): Set<string> => {
    const found = new Set<string>();
    for (const member of nodes) {
      if (
        (ts.isPropertySignature(member) || ts.isMethodSignature(member)) &&
        member.name !== undefined &&
        ts.isIdentifier(member.name)
      ) {
        found.add(member.name.text);
      }
    }
    return found;
  };

  for (const file of sources()) {
    const tree = ts.createSourceFile(
      file,
      readFileSync(file, 'utf8'),
      ts.ScriptTarget.ES2022,
      true,
    );
    const visit = (node: ts.Node): void => {
      if (ts.isIdentifier(node) && node.parent !== undefined) {
        const parent = node.parent;
        const declares =
          (ts.isInterfaceDeclaration(parent) ||
            ts.isTypeAliasDeclaration(parent) ||
            ts.isClassDeclaration(parent) ||
            ts.isEnumDeclaration(parent) ||
            ts.isEnumMember(parent) ||
            ts.isFunctionDeclaration(parent) ||
            ts.isVariableDeclaration(parent) ||
            ts.isParameter(parent) ||
            ts.isPropertySignature(parent) ||
            ts.isMethodSignature(parent) ||
            ts.isPropertyDeclaration(parent) ||
            ts.isMethodDeclaration(parent) ||
            ts.isPropertyAssignment(parent) ||
            ts.isShorthandPropertyAssignment(parent)) &&
          parent.name === node;
        if (declares) names.add(node.text);
      }
      if (ts.isInterfaceDeclaration(node)) {
        const found = members.get(node.name.text) ?? new Set<string>();
        for (const name of memberNames(node.members)) found.add(name);
        members.set(node.name.text, found);
      }
      if (ts.isTypeAliasDeclaration(node)) {
        aliases.set(node.name.text, node.type.getText());
        if (ts.isTypeLiteralNode(node.type)) {
          const found = members.get(node.name.text) ?? new Set<string>();
          for (const name of memberNames(node.type.members)) found.add(name);
          members.set(node.name.text, found);
        }
        if (ts.isUnionTypeNode(node.type)) {
          const arms = new Set<string>();
          for (const arm of node.type.types) {
            if (ts.isLiteralTypeNode(arm) && ts.isStringLiteral(arm.literal)) {
              arms.add(arm.literal.text);
            }
          }
          if (arms.size > 0) unions.set(node.name.text, arms);
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(tree);
  }
  return { names, members, unions, aliases };
}

const SURFACE = surface();

/** True while nothing the source declares is named for `pattern`. */
function nothingNamed(pattern: RegExp): boolean {
  for (const name of SURFACE.names) if (pattern.test(name)) return false;
  return true;
}

/** The members of one interface, which must exist. */
function membersOf(type: string): ReadonlySet<string> {
  const found = SURFACE.members.get(type);
  if (found === undefined) throw new Error(`known-gaps: no interface named ${type}`);
  return found;
}

/** True while no member of `type` is named for `pattern`. */
function noMemberNamed(type: string, pattern: RegExp): boolean {
  for (const name of membersOf(type)) if (pattern.test(name)) return false;
  return true;
}

interface Gap {
  /** The bolded lead of the README bullet, exactly as written. */
  readonly heading: string;
  /** True while the gap is still a gap. */
  readonly stillOpen: () => boolean;
}

const GAPS: readonly Gap[] = [
  {
    heading: 'Calibration fits the Rasch model and the 2PL, and nothing else.',
    // The scoring side carries the 3PL and the 4PL, so a guessing floor can be
    // written down, scored, linked, and checked for fit. What cannot happen is
    // producing one from response data: both estimators take their model from
    // this union, and the union has two arms.
    stillOpen: () => {
      const arms = SURFACE.unions.get('CalibrationModel');
      return arms !== undefined && arms.size === 2 && arms.has('rasch') && arms.has('2pl');
    },
  },
  {
    heading: 'Only dichotomous responses can be calibrated.',
    // Two independent halves of the same gap: what the response matrix accepts,
    // and what a calibrated item comes back as. A rubric scored 0-3 cannot be
    // entered, and no estimator returns a threshold.
    stillOpen: () =>
      SURFACE.aliases.get('Cell') === 'Response | null' &&
      noMemberNamed('CalibratedItem', /threshold|categor|step/i),
  },
  {
    heading: 'No prior on an item parameter.',
    // Scoped to the calibration options on purpose: `BayesOptions.prior` is a
    // prior on *ability*, and that one exists. What does not is a prior on a
    // difficulty or a discrimination, which is why the bounds are hard clamps.
    stillOpen: () =>
      noMemberNamed('CalibrationOptions', /prior|penalt|ridge/i) &&
      noMemberNamed('MarginalOptions', /prior|penalt|ridge/i) &&
      noMemberNamed('MaximiseOptions', /prior|penalt|ridge/i),
  },
  {
    heading: 'One latent dimension.',
    stillOpen: () => nothingNamed(/multidimension|bifactor|loading|latentDimension/i),
  },
  {
    heading: 'Responses are conditionally independent.',
    stillOpen: () => nothingNamed(/testlet|localDepend|itemBundle|residualCorrelation/i),
  },
  {
    heading: 'A missing response is one kind of missing.',
    stillOpen: () => nothingNamed(/notReached|omitted|nonresponse|abandon/i),
  },
  {
    heading: 'No sampling error on a linking coefficient.',
    stillOpen: () =>
      noMemberNamed('LinkingResult', /error|variance|interval/i) &&
      nothingNamed(/bootstrap|jackknife/i),
  },
  {
    heading: 'No equating between forms.',
    // `conversionTable` maps a raw score to an ability on one form, which is a
    // different thing and deliberately not matched here.
    stillOpen: () =>
      nothingNamed(/equipercentile|equating|^equate|equivalentScore|concordance/i),
  },
  {
    heading: 'Content balance runs on one classification at a time.',
    stillOpen: () =>
      SURFACE.aliases.get('Blueprint') === 'ReadonlyMap<string, number>' &&
      noMemberNamed('Item', /cognitive|classification|constraint/i) &&
      nothingNamed(/enemy|shadowTest|constraintSet/i),
  },
  {
    heading: 'The test is adaptive item by item; there is no multistage design.',
    stillOpen: () => nothingNamed(/multistage|routing|^panels?$|^modules?$/i),
  },
  {
    heading: 'Nothing is timed.',
    stillOpen: () => nothingNamed(/responseTime|^durations?$|elapsed|timeLimit|speeded/i),
  },
];

/** The bolded lead and the body of every bullet in the README's gaps section. */
function bullets(): { heading: string; body: string }[] {
  const text = readFileSync(README, 'utf8');
  const section = /\n## Known gaps\n([\s\S]*?)(?=\n## )/.exec(text);
  expect(section, 'the README has no known-gaps section').not.toBeNull();
  const found: { heading: string; body: string }[] = [];
  for (const chunk of (section?.[1] ?? '').split(/\n(?=- \*\*)/)) {
    const bullet = /^- \*\*(.+?)\*\*([\s\S]*)$/.exec(chunk.trim());
    if (bullet === null) continue;
    found.push({ heading: bullet[1] ?? '', body: (bullet[2] ?? '').trim() });
  }
  return found;
}

describe('the known-gaps list', () => {
  it('surveyed the source it claims to have surveyed', () => {
    // Every predicate of the form "nothing is named X" passes trivially over an
    // empty inventory, so the inventory is asserted before any of them runs:
    // enough files, enough names, and a handful that are certainly there.
    expect(sources().length).toBeGreaterThan(60);
    expect(SURFACE.names.size).toBeGreaterThan(900);
    for (const certain of ['metricScale', 'slopeCoefficients', 'AdaptiveSession', 'thresholds'])
      expect(SURFACE.names).toContain(certain);
    expect(nothingNamed(/metricScale/)).toBe(false);
    expect(SURFACE.unions.get('CalibrationModel')).toBeDefined();
    expect(membersOf('LinkingResult').size).toBeGreaterThan(3);
    expect(SURFACE.aliases.get('Cell')).toBe('Response | null');
  });

  it('is in the README, in this order', () => {
    expect(bullets().map((entry) => entry.heading)).toEqual(GAPS.map((gap) => gap.heading));
  });

  it('has not quietly shrunk', () => {
    expect(GAPS).toHaveLength(11);
  });

  it('says what each gap costs, rather than only naming it', () => {
    // A bullet reduced to its lead is a list entry, not a disclosure: the lead
    // says what is absent and the body has to say what that means for someone
    // who assumed otherwise.
    for (const entry of bullets()) {
      expect(entry.body.length, `the bullet "${entry.heading}" has gone thin`).toBeGreaterThan(160);
    }
  });

  it('is still true of every line', () => {
    const closed = GAPS.filter((gap) => !gap.stillOpen()).map((gap) => gap.heading);
    expect(
      closed,
      'these gaps look closed; take the lines out of the README and the entries out of this table',
    ).toEqual([]);
  });
});
