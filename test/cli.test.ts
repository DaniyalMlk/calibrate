import { afterEach, describe, expect, it, vi } from 'vitest';
import { main } from '../src/cli/run.js';

function capture(argv: readonly string[]): { code: number; out: string; err: string } {
  let out = '';
  let err = '';
  const outSpy = vi.spyOn(process.stdout, 'write').mockImplementation((chunk) => {
    out += String(chunk);
    return true;
  });
  const errSpy = vi.spyOn(process.stderr, 'write').mockImplementation((chunk) => {
    err += String(chunk);
    return true;
  });
  try {
    const code = main(argv);
    return { code, out, err };
  } finally {
    outSpy.mockRestore();
    errSpy.mockRestore();
  }
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('cli', () => {
  it('prints usage and succeeds for --help', () => {
    const { code, out } = capture(['--help']);
    expect(code).toBe(0);
    expect(out).toContain('calibrate demo');
  });

  it('prints usage and fails when given no command', () => {
    const { code, out } = capture([]);
    expect(code).toBe(1);
    expect(out).toContain('Usage:');
  });

  it('rejects an unknown command', () => {
    const { code, err } = capture(['analyse']);
    expect(code).toBe(1);
    expect(err).toContain('unknown command "analyse"');
  });

  it('rejects an unknown option', () => {
    const { code, err } = capture(['demo', '--nope', '1']);
    expect(code).toBe(1);
    expect(err).toContain('unknown option "--nope"');
  });

  it('rejects a missing or non-numeric option value', () => {
    expect(capture(['demo', '--theta']).err).toContain('needs a value');
    expect(capture(['demo', '--theta', 'high']).err).toContain('needs a number');
  });

  it('runs a demo session and reports a transcript', () => {
    const { code, out } = capture(['demo', '--theta', '1.0', '--max', '12', '--bank', '120']);
    expect(code).toBe(0);
    expect(out).toContain('Bank: 120 items across 3 domains');
    expect(out).toContain('true ability 1.00');
    expect(out).toContain('Finished after');
    // 12 transcript rows, numbered.
    expect(out.split('\n').filter((line) => /^\d+\s+\w/.test(line))).toHaveLength(12);
  });

  it('is deterministic for a given seed', () => {
    const first = capture(['demo', '--seed', '99', '--max', '10']);
    const second = capture(['demo', '--seed', '99', '--max', '10']);
    expect(first.out).toBe(second.out);
  });

  it('stops early when the precision target is loose', () => {
    const { out } = capture(['demo', '--target', '5', '--min', '4', '--max', '30']);
    expect(out).toContain('Finished after 4 items');
    expect(out).toContain('standard error');
  });

  it('propagates a bad configuration as a failure rather than a crash', () => {
    const { code, err } = capture(['demo', '--min', '20', '--max', '10']);
    expect(code).toBe(1);
    expect(err).toContain('must not exceed');
  });
});

describe('cli option parsing', () => {
  it('rejects a non-integer where an integer is required', () => {
    expect(capture(['demo', '--max', '12.5']).err).toContain('needs an integer');
  });

  it('rejects a value outside the choices a string option allows', () => {
    const { code, err } = capture(['study', '--population', 'lognormal']);
    expect(code).toBe(1);
    expect(err).toContain('must be one of: normal, uniform, grid');
  });

  it('rejects a bare argument that is not a flag', () => {
    expect(capture(['demo', '12']).err).toContain('unexpected argument "12"');
  });

  it('lists every command and its options in the usage text', () => {
    const { out } = capture(['--help']);
    expect(out).toContain('calibrate demo');
    expect(out).toContain('calibrate study');
    expect(out).toContain('calibrate recover');
    expect(out).toContain('--replications <integer>');
  });
});

describe('study command', () => {
  it('compares every policy over a simulated population', () => {
    const { code, out } = capture([
      'study',
      '--examinees',
      '12',
      '--bank',
      '60',
      '--max',
      '10',
      '--bins',
      '4',
    ]);
    expect(code).toBe(0);
    expect(out).toContain('Population: normal(0, 1), 12 examinees per policy');
    expect(out).toContain('max-information');
    expect(out).toContain('kullback-leibler');
    expect(out).toContain('balanced+randomesque');
    expect(out).toContain('fixed-10');
    expect(out).toContain('conditional on true ability');
    expect(out).toContain('items used');
  });

  it('accepts each population it offers', () => {
    for (const population of ['normal', 'uniform', 'grid']) {
      const { code, out } = capture([
        'study',
        '--population',
        population,
        '--examinees',
        '6',
        '--bank',
        '40',
        '--max',
        '6',
      ]);
      expect(code).toBe(0);
      expect(out).toContain(`Population: ${population}`);
    }
  });

  it('is deterministic for a given seed', () => {
    const argv = ['study', '--examinees', '8', '--bank', '40', '--max', '8', '--seed', '5'];
    expect(capture(argv).out).toBe(capture(argv).out);
  });
});

describe('recover command', () => {
  it('reports every estimator by default', () => {
    const { code, out } = capture([
      'recover',
      '--length',
      '8',
      '--points',
      '3',
      '--replications',
      '20',
    ]);
    expect(code).toBe(0);
    expect(out).toContain('Form: 8 items, 20 replications');
    for (const name of ['mle', 'eap', 'map', 'wle', 'hybrid']) {
      expect(out).toContain(`\n${name} — mean absolute bias`);
    }
  });

  it('reports a single estimator when asked for one', () => {
    const { code, out } = capture([
      'recover',
      '--estimator',
      'wle',
      '--length',
      '8',
      '--points',
      '3',
      '--replications',
      '20',
    ]);
    expect(code).toBe(0);
    expect(out).toContain('wle — mean absolute bias');
    expect(out).not.toContain('eap — mean absolute bias');
  });

  it('rejects a non-positive ability range', () => {
    const { code, err } = capture(['recover', '--range', '0']);
    expect(code).toBe(1);
    expect(err).toContain('must be positive');
  });
});
