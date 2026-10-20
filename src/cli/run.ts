import { DEMO_OPTIONS, runDemo } from './demo.js';
import { DIF_OPTIONS, runDif } from './dif.js';
import { FIT_OPTIONS, runFit } from './fit.js';
import { LINK_OPTIONS, runLink } from './link.js';
import { MARGINAL_OPTIONS, runMarginal } from './marginal.js';
import { MISFIT_OPTIONS, runMisfit } from './misfit.js';
import { MIXED_OPTIONS, runMixed } from './mixed.js';
import { describeOptions, describeWidth, type OptionSpecs } from './options.js';
import { RECOVER_OPTIONS, runRecover } from './recover.js';
import { STUDY_OPTIONS, runStudyCommand } from './study.js';

interface Command {
  readonly summary: string;
  readonly options: OptionSpecs;
  run(argv: readonly string[]): void;
}

const COMMANDS: Readonly<Record<string, Command>> = {
  demo: {
    summary: 'Run one adaptive session against a synthetic bank and print the transcript',
    options: DEMO_OPTIONS,
    run: runDemo,
  },
  mixed: {
    summary: 'Run an adaptive session over a bank mixing binary and rubric-scored items',
    options: MIXED_OPTIONS,
    run: runMixed,
  },
  study: {
    summary: 'Compare selection policies over a simulated population',
    options: STUDY_OPTIONS,
    run: runStudyCommand,
  },
  recover: {
    summary: 'Measure how well an estimator recovers ability across the range',
    options: RECOVER_OPTIONS,
    run: runRecover,
  },
  link: {
    summary: 'Calibrate one anchor on two samples and put the two scales on one metric',
    options: LINK_OPTIONS,
    run: runLink,
  },
  fit: {
    summary: 'Calibrate item parameters from responses, then report fit and bank health',
    options: FIT_OPTIONS,
    run: runFit,
  },
  mml: {
    summary:
      'Calibrate items by marginal maximum likelihood, and compare against joint estimation',
    options: MARGINAL_OPTIONS,
    run: runMarginal,
  },
  dif: {
    summary: 'Scan a two-group administration for items that behave differently at equal ability',
    options: DIF_OPTIONS,
    run: runDif,
  },
  misfit: {
    summary: 'Scan a simulated sitting for candidates whose pattern does not fit the model',
    options: MISFIT_OPTIONS,
    run: runMisfit,
  },
};

/** The names of every command, in the order the usage text lists them. */
export const COMMAND_NAMES: readonly string[] = Object.keys(COMMANDS);

/**
 * The option specification of each command, keyed by command name.
 *
 * The help text is rendered from this, so anything checking the help can be
 * written against the same source rather than against a list of commands and
 * flags kept by hand somewhere else.
 */
export const COMMAND_OPTIONS: Readonly<Record<string, OptionSpecs>> = Object.fromEntries(
  Object.entries(COMMANDS).map(([name, command]) => [name, command.options]),
);

/** The flags that ask for help rather than supplying a value. */
const HELP_FLAGS: readonly string[] = ['-h', '--help', 'help'];

function usage(): string {
  let out = 'calibrate — adaptive testing engine\n\nUsage:\n';
  for (const name of Object.keys(COMMANDS)) out += `  calibrate ${name} [options]\n`;
  out += '\nCommands:\n';
  for (const [name, command] of Object.entries(COMMANDS)) {
    out += `\n  ${name} — ${command.summary}\n${describeOptions(command.options)}\n`;
  }
  out += '\n  -h, --help          Show this message\n';
  out += '\nRun `calibrate <command> --help` for one command on its own.\n';
  return out;
}

/**
 * The help text for a single command.
 *
 * The whole-program usage lists eight commands and every option each of them
 * takes, which is far more than someone who has already chosen a command wants
 * to read. This is the same information narrowed to the one command asked about.
 */
export function commandUsage(name: string): string {
  const command = COMMANDS[name];
  if (command === undefined) throw new Error(`unknown command "${name}"`);
  return (
    `calibrate ${name} — ${command.summary}\n\n` +
    `Usage:\n  calibrate ${name} [options]\n\nOptions:\n` +
    `${describeOptions(command.options)}\n` +
    `  ${'-h, --help'.padEnd(describeWidth(command.options))}Show this message\n`
  );
}

export function main(argv: readonly string[]): number {
  const [name, ...rest] = argv;
  if (name === undefined || HELP_FLAGS.includes(name)) {
    process.stdout.write(usage());
    return name === undefined ? 1 : 0;
  }

  const command = COMMANDS[name];
  if (command === undefined) {
    process.stderr.write(`unknown command "${name}"\n\n${usage()}`);
    return 1;
  }

  // Asking a command for help is a request, not a parse error. It has to be
  // answered before the options are parsed at all, because the parser rejects
  // any flag it does not recognise — which is the right behaviour for a typo
  // in `--replications` and the wrong one for `--help`.
  if (rest.some((argument) => HELP_FLAGS.includes(argument))) {
    process.stdout.write(commandUsage(name));
    return 0;
  }

  try {
    command.run(rest);
    return 0;
  } catch (error) {
    process.stderr.write(
      `${error instanceof Error ? error.message : String(error)}\n\n${commandUsage(name)}`,
    );
    return 1;
  }
}
