/**
 * A small declarative option parser for the command line.
 *
 * Written rather than taken from a dependency because the whole engine has none,
 * and because what the commands need is narrow: long flags, one value each, all
 * of them optional. The declarative form matters more than the parsing — a
 * command declares its options once and gets both the parsed values and the
 * help text from the same object, so the two cannot drift apart.
 */

export type OptionKind = 'number' | 'integer' | 'string';

export interface OptionSpec {
  readonly kind: OptionKind;
  /** Value used when the flag is absent. */
  readonly fallback: number | string;
  /** One-line description, rendered in the usage text. */
  readonly describe: string;
  /** Permitted values, for string options that name a choice. */
  readonly choices?: readonly string[];
}

export type OptionSpecs = Readonly<Record<string, OptionSpec>>;

export type ParsedOptions<S extends OptionSpecs> = {
  readonly [K in keyof S]: S[K]['kind'] extends 'string' ? string : number;
};

/**
 * Parse `--flag value` pairs against a specification.
 *
 * Unknown flags, missing values and values of the wrong shape all throw. An
 * unrecognised flag is much more likely to be a typo in a flag that matters than
 * something the caller is happy to have ignored — silently dropping
 * `--replications 500` and running 200 would produce a report that looks right
 * and is not.
 */
export function parseOptions<S extends OptionSpecs>(
  argv: readonly string[],
  specs: S,
): ParsedOptions<S> {
  const values: Record<string, number | string> = {};
  for (const [name, spec] of Object.entries(specs)) values[name] = spec.fallback;

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i] as string;
    if (!flag.startsWith('--')) throw new Error(`unexpected argument "${flag}"`);
    const key = flag.slice(2);
    const spec = specs[key];
    if (spec === undefined) throw new Error(`unknown option "${flag}"`);

    const raw = argv[i + 1];
    if (raw === undefined) throw new Error(`option "${flag}" needs a value`);
    i += 1;

    if (spec.kind === 'string') {
      if (spec.choices !== undefined && !spec.choices.includes(raw)) {
        throw new Error(`option "${flag}" must be one of: ${spec.choices.join(', ')}`);
      }
      values[key] = raw;
      continue;
    }

    const value = Number(raw);
    if (!Number.isFinite(value)) throw new Error(`option "${flag}" needs a number, got "${raw}"`);
    if (spec.kind === 'integer' && !Number.isInteger(value)) {
      throw new Error(`option "${flag}" needs an integer, got "${raw}"`);
    }
    values[key] = value;
  }

  return values as ParsedOptions<S>;
}

/** Render the option list of a command as aligned help text. */
export function describeOptions(specs: OptionSpecs): string {
  const entries = Object.entries(specs);
  if (entries.length === 0) return '';
  const width = Math.max(...entries.map(([name, spec]) => `--${name} <${spec.kind}>`.length)) + 2;
  return entries
    .map(([name, spec]) => {
      const flag = `--${name} <${spec.kind}>`;
      const choices = spec.choices === undefined ? '' : ` (${spec.choices.join('|')})`;
      return `  ${flag.padEnd(width)}${spec.describe}${choices} [${spec.fallback}]`;
    })
    .join('\n');
}
