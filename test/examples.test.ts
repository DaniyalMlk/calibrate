/**
 * What a reader copies off the README or the front page, checked against the
 * package it is meant to run against.
 *
 * Two of the three names the front page's import line used, `gradedResponse`
 * and `nextItem`, were never exports. An ES module that imports a name its
 * target does not export fails at link time, so that line could not run at all,
 * and nothing read it: the README's examples are checked by hand, and the
 * page's by nobody.
 *
 * The install line had the same shape. It said `npm install calibrate-irt`
 * while the package was not on the registry, and the fallback a reader would
 * reach for, installing from the repository, produced a package with no `dist`
 * in it (see the packaging test). Whether the package is published is recorded
 * in one place, the roadmap's release checkbox, so the install lines are read
 * against that.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildSite, INSTALL_FROM_SOURCE, PACKAGE } from "../docs/site.js";
import { COMMAND_NAMES } from "../src/cli/run.js";
import * as calibrate from "../src/index.js";

const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8");
const roadmap = readFileSync(new URL("../ROADMAP.md", import.meta.url), "utf8");
const manifest = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
  repository: { url: string };
};
const page = buildSite().get("index.html") ?? "";

/** Every name an `import { ... } from "<pkg>"` in `text` asks for. */
export function importedNames(text: string, pkg: string): string[] {
  const quoted = pkg.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`import\\s+(?:type\\s+)?\\{([^}]*)\\}\\s*from\\s*['"]${quoted}['"]`, "g");
  const names: string[] = [];
  for (const match of text.matchAll(pattern)) {
    for (const raw of (match[1] ?? "").split(",")) {
      const name = raw.trim().replace(/^type\s+/, "").split(/\s+as\s+/)[0] ?? "";
      if (name !== "") names.push(name);
    }
  }
  return names;
}

/** The text of every fenced or `<pre>` code block in `text`. */
export function codeBlocks(text: string): string[] {
  const fenced = [...text.matchAll(/```[a-z]*\n([\s\S]*?)```/g)].map((m) => m[1] ?? "");
  const pre = [...text.matchAll(/<pre><code>([\s\S]*?)<\/code><\/pre>/g)].map((m) => m[1] ?? "");
  return [...fenced, ...pre];
}

const NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];

describe("the example import parser", () => {
  it("reads names across lines, past `type` and `as`, from one package only", () => {
    const text = [
      "import {\n  alpha,\n  type Beta,\n  gamma as g,\n} from 'calibrate-irt';",
      'import type { Delta } from "calibrate-irt";',
      "import { epsilon } from 'calibrate-irt-other';",
      "import { zeta } from './local.js';",
    ].join("\n");
    expect(importedNames(text, PACKAGE)).toEqual(["alpha", "Beta", "gamma", "Delta"]);
  });

  it("finds code in both fenced and HTML blocks, and nothing outside them", () => {
    expect(codeBlocks("a `npm x` b\n```bash\nnpm y\n```\n<pre><code>npm z</code></pre>")).toEqual([
      "npm y\n",
      "npm z",
    ]);
  });
});

describe("every name an example imports is one the package exports", () => {
  it.each([
    ["the README", readme, 50],
    ["the front page", page, 3],
  ])("in %s", (_label, text, atLeast) => {
    const names = importedNames(text, PACKAGE);
    // A parser that matched nothing would find nothing missing.
    expect(names.length).toBeGreaterThanOrEqual(atLeast);
    expect(names.filter((name) => !(name in calibrate))).toEqual([]);
  });
});

describe("the install lines say what is true today", () => {
  const releaseLine = /^- \[( |x)\] A published release on the registry$/m.exec(roadmap);
  const published = releaseLine?.[1] === "x";
  const registryInstall = `npm install ${PACKAGE}`;
  const sourceInstall = `npm install ${INSTALL_FROM_SOURCE}`;

  it("reads the release state off the roadmap", () => {
    expect(releaseLine).not.toBeNull();
  });

  it("installs from the repository it describes", () => {
    const [owner, repo] = INSTALL_FROM_SOURCE.replace(/^github:/, "").split("/");
    expect(manifest.repository.url).toBe(`git+https://github.com/${owner}/${repo}.git`);
  });

  it.each([
    ["the README", readme],
    ["the front page", page],
  ])("gives %s a command that works", (_label, text) => {
    const blocks = codeBlocks(text).join("\n");
    if (published) {
      expect(blocks).toContain(registryInstall);
    } else {
      // Only a command is checked: the prose may say what the line will be.
      expect(blocks).not.toContain(registryInstall);
      expect(blocks).toContain(sourceInstall);
    }
  });
});

describe("the README's command count", () => {
  it("is the number of commands the CLI has", () => {
    const stated = /not all (\w+)\)?$/m.exec(readme)?.[1];
    expect(stated).toBeDefined();
    expect(NUMBER_WORDS.indexOf(stated ?? "")).toBe(COMMAND_NAMES.length);
  });
});
