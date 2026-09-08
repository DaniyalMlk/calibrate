import { rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
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

describe('fit command', () => {
  it('calibrates a simulated bank and reports fit, recovery and health', () => {
    const { code, out } = capture([
      'fit',
      '--bank',
      '30',
      '--respondents',
      '250',
      '--target',
      '0.6',
    ]);
    expect(code).toBe(0);
    expect(out).toContain('simulated: 250 respondents, 30 rasch items');
    expect(out).toContain('converged in');
    expect(out).toContain('items outside [0.7, 1.3]');
    expect(out).toContain('Recovery against the generating difficulties');
    expect(out).toContain('Bank health at a target standard error of 0.6');
    // One table row per calibrated item, under a header and a rule.
    expect(
      out.split('\n').filter((line) => /^(arrays|graphs|dynamic-programming)-\d/.test(line)).length,
    ).toBe(30);
  });

  it('recovers the generating difficulties closely enough to say so', () => {
    const { out } = capture(['fit', '--bank', '40', '--respondents', '600']);
    const match = /correlation (\d+\.\d+), mean absolute error (\d+\.\d+)/.exec(out);
    expect(match).not.toBeNull();
    expect(Number(match?.[1])).toBeGreaterThan(0.95);
    expect(Number(match?.[2])).toBeLessThan(0.2);
  });

  it('fits a 2PL when asked, and reports discriminations that vary', () => {
    const { code, out } = capture([
      'fit',
      '--model',
      '2pl',
      '--bank',
      '25',
      '--respondents',
      '400',
      '--target',
      '0.6',
    ]);
    expect(code).toBe(0);
    expect(out).toContain('25 2pl items');
    const discriminations = new Set(
      out
        .split('\n')
        .filter((line) => /^(arrays|graphs|dynamic)-/.test(line))
        .map((line) => line.slice(50, 57).trim()),
    );
    expect(discriminations.size).toBeGreaterThan(1);
  });

  it('reads a response matrix from a CSV file', () => {
    const path = join(tmpdir(), `calibrate-cli-${process.pid}.csv`);
    writeFileSync(path, 'id,q1,q2,q3,q4\na,1,0,1,0\nb,0,1,0,1\nc,1,1,0,0\nd,0,0,1,1\ne,1,0,0,1\n');
    try {
      const { code, out } = capture(['fit', '--file', path, '--index', '1', '--target', '1.5']);
      expect(code).toBe(0);
      expect(out).toContain('5 respondents, 4 items');
      expect(out).toContain('q1');
      // Real data has no generating parameters to recover.
      expect(out).not.toContain('Recovery against');
    } finally {
      rmSync(path, { force: true });
    }
  });

  it('reports a bad CSV as a failure rather than a crash', () => {
    const path = join(tmpdir(), `calibrate-cli-bad-${process.pid}.csv`);
    writeFileSync(path, 'q1,q2\n1,YES\n');
    try {
      const { code, err } = capture(['fit', '--file', path]);
      expect(code).toBe(1);
      expect(err).toContain('received "YES"');
    } finally {
      rmSync(path, { force: true });
    }
  });

  it('is deterministic for a given seed', () => {
    const argv = ['fit', '--bank', '20', '--respondents', '200', '--seed', '7', '--target', '0.8'];
    expect(capture(argv).out).toBe(capture(argv).out);
  });
});
