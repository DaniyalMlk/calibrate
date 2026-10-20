import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

interface Manifest {
  readonly name: string;
  readonly version: string;
  readonly type: string;
  readonly license: string;
  readonly engines: Record<string, string>;
  readonly files: readonly string[];
  readonly bin: Record<string, string>;
  readonly main: string;
  readonly types: string;
  readonly exports: Record<string, Record<string, string>>;
  readonly scripts: Record<string, string>;
  readonly dependencies?: Record<string, string>;
  readonly devDependencies: Record<string, string>;
  readonly publishConfig: Record<string, string>;
  readonly repository: { readonly type: string; readonly url: string };
}

const manifest = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')) as Manifest;

/**
 * What `npm pack` would actually put in the tarball.
 *
 * The question this file exists to answer is what a consumer receives, which
 * is not the same question as what `files` lists: `files` is a set of patterns,
 * npm applies its own always-included and never-included rules on top, and the
 * build decides what exists under `dist` in the first place. So the list is
 * taken from npm rather than reasoned about.
 */
function packedFiles(): readonly string[] {
  const out = execFileSync('npm', ['pack', '--dry-run', '--json'], {
    cwd: ROOT,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
    maxBuffer: 32 * 1024 * 1024,
  });
  const packed = JSON.parse(out) as [{ files: { path: string }[]; name: string }];
  return packed[0].files.map((file) => file.path);
}

describe('the published manifest', () => {
  it('is published under a name the registry has free for it', () => {
    // `calibrate` is an unrelated package at 4.2.0. Publishing under it is not
    // a thing that fails gracefully — it fails against a project nobody here
    // owns, with an error that reads like a permissions problem.
    expect(manifest.name).toBe('calibrate-irt');
  });

  it('points at the repository this is, which is what trusted publishing matches', () => {
    // The registry matches this against the repository the publishing workflow
    // runs in. That is the GitHub repository name, `calibrate`, and not the
    // distribution name — they differ here, which is exactly the case where
    // writing the distribution name in by reflex would fail the publish.
    expect(manifest.repository.url).toBe('git+https://github.com/DaniyalMlk/calibrate.git');
  });

  it('declares a floor that resolves rather than a bare major', () => {
    expect(manifest.engines['node']).toBe('>=20.19.0');
  });

  it('publishes publicly and builds on pack', () => {
    expect(manifest.publishConfig['access']).toBe('public');
    // prepack, not prepublishOnly: `npm pack` has to rebuild too, or a stale
    // dist from someone's working copy is what ships.
    expect(manifest.scripts['prepack']).toBe('npm run build');
  });

  it('has no runtime dependencies', () => {
    expect(manifest.dependencies ?? {}).toEqual({});
  });

  it('installs the command line under the distribution name as well', () => {
    // `npx calibrate-irt` resolves a bin of that name, so a package whose
    // distribution name differs from its command needs both or the obvious
    // invocation fails.
    expect(Object.keys(manifest.bin).sort()).toEqual(['calibrate', 'calibrate-irt']);
    for (const target of Object.values(manifest.bin)) {
      expect(target).toBe('./dist/cli/main.js');
    }
  });
});

describe('the tarball', () => {
  const files = packedFiles();

  it('contains the entry point, its declarations and the bin target', () => {
    expect(files).toContain('dist/index.js');
    expect(files).toContain('dist/index.d.ts');
    expect(files).toContain('dist/cli/main.js');
  });

  it('ships the licence and the README', () => {
    expect(files).toContain('LICENSE');
    expect(files).toContain('README.md');
  });

  it('every path the manifest advertises is in it', () => {
    for (const advertised of [manifest.main, manifest.types, ...Object.values(manifest.bin)]) {
      expect(files).toContain(advertised.replace(/^\.\//, ''));
    }
    for (const entry of Object.values(manifest.exports)) {
      for (const target of Object.values(entry)) {
        expect(files).toContain(target.replace(/^\.\//, ''));
      }
    }
  });

  it('carries no source maps, which would point at sources it does not ship', () => {
    expect(files.filter((file) => file.endsWith('.map'))).toEqual([]);
  });

  it('carries declarations but no TypeScript sources', () => {
    // .d.ts ends in .ts and belongs here, so the check is for sources rather
    // than for the extension.
    const declarations = files.filter((file) => file.endsWith('.d.ts'));
    expect(declarations.length).toBeGreaterThan(20);
    const sources = files.filter((file) => file.endsWith('.ts') && !file.endsWith('.d.ts'));
    expect(sources).toEqual([]);
  });

  it('ships no tests, no web interface and no build configuration', () => {
    // `package.json` is in the tarball whatever `files` says — npm always
    // includes it, along with the licence and the README. Everything else
    // here has to have been put there deliberately.
    const always = new Set(['LICENSE', 'README.md', 'package.json']);
    for (const file of files) {
      expect(file.startsWith('dist/') || always.has(file)).toBe(true);
      expect(file).not.toMatch(/(^|\/)(test|web)\//);
    }
    expect(files.some((file) => file.startsWith('dist/tests'))).toBe(false);
  });

  it('is emitted flat from src, not nested under it', () => {
    // A tsconfig that includes `tests` alongside `src` emits `dist/src/...`
    // and `dist/tests/...`, which silently moves every advertised path.
    expect(files.some((file) => file.startsWith('dist/src/'))).toBe(false);
  });
});

describe('the command line as published', () => {
  it('has a shebang on every bin target, or the shim runs it as a shell script', () => {
    for (const target of new Set(Object.values(manifest.bin))) {
      const source = readFileSync(join(ROOT, target), 'utf8');
      expect(source.startsWith('#!/usr/bin/env node')).toBe(true);
    }
  });
});

describe('the README', () => {
  const readme = readFileSync(join(ROOT, 'README.md'), 'utf8');

  it('tells the reader to install the name that is actually published', () => {
    expect(readme).toContain(`npm install ${manifest.name}`);
  });

  it('imports the published name in every example, not the repository name', () => {
    // Every example used to import 'calibrate', which is a different package
    // owned by somebody else: the snippets would have run, against the wrong
    // code, for anyone who copied them.
    const imports = [...readme.matchAll(/from '([^']+)'/g)].map((match) => match[1]);
    expect(imports.length).toBeGreaterThan(10);
    for (const imported of imports) {
      expect(imported).toBe(manifest.name);
    }
  });
});
