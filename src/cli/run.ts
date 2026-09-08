import { DEMO_OPTIONS, runDemo } from './demo.js';
import { FIT_OPTIONS, runFit } from './fit.js';
import { describeOptions, type OptionSpecs } from './options.js';
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
  fit: {
    summary: 'Calibrate item parameters from responses, then report fit and bank health',
    options: FIT_OPTIONS,
    run: runFit,
  },
};

function usage(): string {
  let out = 'calibrate — adaptive testing engine\n\nUsage:\n';
  for (const name of Object.keys(COMMANDS)) out += `  calibrate ${name} [options]\n`;
  out += '\nCommands:\n';
  for (const [name, command] of Object.entries(COMMANDS)) {
    out += `\n  ${name} — ${command.summary}\n${describeOptions(command.options)}\n`;
  }
  out += '\n  -h, --help          Show this message\n';
  return out;
}

export function main(argv: readonly string[]): number {
  const [name, ...rest] = argv;
  if (name === undefined || name === '-h' || name === '--help' || name === 'help') {
    process.stdout.write(usage());
    return name === undefined ? 1 : 0;
  }

  const command = COMMANDS[name];
  if (command === undefined) {
    process.stderr.write(`unknown command "${name}"\n\n${usage()}`);
    return 1;
  }

  try {
    command.run(rest);
    return 0;
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n\n${usage()}`);
    return 1;
  }
}
