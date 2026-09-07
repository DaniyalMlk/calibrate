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
