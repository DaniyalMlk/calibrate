/**
 * The site against the engine it describes.
 *
 * The page is generated, so the usual failure -- a figure edited in the code
 * and not on the page -- cannot happen. The failures that *can* happen are the
 * two this file is about: the page quoting something no gate produced, and the
 * page quietly dropping something the gate does produce.
 *
 * So the assertions run in both directions. Every row the gate renders appears
 * whole -- label and result in one string, because asserting them separately
 * passes on a generator that paired the wrong result with a label -- and the
 * rendered rows are counted against the gate's, because a forward-only check
 * passes on an empty page.
 *
 * Figures quoted *outside* the table are matched inside the element they are
 * rendered into rather than anywhere on the page. Containment against a whole
 * page is satisfied by the evidence table sitting further down it, which is a
 * hole a break sweep found in a sibling repository's gate: a wrong headline
 * passed because the right figure was two screens below it.
 */

import { readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { APP, buildSite, counts, evidenceRows, PACKAGE, SOURCE } from "../docs/site.js";
import { measure } from "../scripts/evidence.js";
import type { Check } from "../scripts/evidence.js";
import { COMMAND_NAMES } from "../src/cli/run.js";
import * as calibrate from "../src/index.js";

const files = buildSite();
const page = files.get("index.html") ?? "";
const style = files.get("style.css") ?? "";
const checks = measure();
const rows = evidenceRows(checks);
const bodyRows = rows.slice(1);

describe("what the site ships", () => {
  it("is a page and a stylesheet, and nothing else", () => {
    expect([...files.keys()].sort()).toEqual(["index.html", "style.css"]);
    expect(style.length).toBeGreaterThan(3000);
  });

  it("links the stylesheet it ships and fetches nothing else", () => {
    expect(page).toContain('<link rel="stylesheet" href="style.css">');
    // The interface has no bundler and no runtime dependencies; a landing page
    // around it that pulled in a font or a CDN script would make that look
    // like an accident rather than a choice.
    expect(page).not.toMatch(/https?:\/\/[^"' ]*\.(?:css|js)/);
    expect(page.toLowerCase()).not.toContain("<script");
    expect(page).not.toContain("@import");
  });

  it("points at the interface it publishes", () => {
    expect(page).toContain(`href="${APP}"`);
    expect(page).toContain(`src="${APP}"`);
    expect(APP.endsWith("/")).toBe(true);
  });

  it("names the distribution, which is not the repository's name", () => {
    expect(PACKAGE).toBe("calibrate-irt");
    expect(page).toContain(`npm install ${PACKAGE}`);
    expect(page).toContain(SOURCE);
  });

  it("is stamped dark, as the interface it frames is", () => {
    // The interface ships `data-theme="dark"`. A light page wrapped around a
    // dark application reads as two pieces of work.
    expect(page).toContain('<html lang="en" data-theme="dark">');
  });
});

describe("every row the gate produces reaches the page", () => {
  it("renders each row whole, label and result together", () => {
    const escape = (text: string): string =>
      text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/`([^`]+)`/g, (_, inner: string) => `<code>${inner}</code>`);

    expect(bodyRows.length).toBeGreaterThanOrEqual(11);
    for (const row of bodyRows) {
      expect(row).toHaveLength(2);
      expect(page).toContain(
        `<tr><td>${escape(row[0] ?? "")}</td><td>${escape(row[1] ?? "")}</td></tr>`,
      );
    }
  });

  it("renders no rows the gate did not produce", () => {
    const body = page.split('<table class="evidence">')[1]?.split("</table>")[0] ?? "";
    expect((body.match(/<tr>/g) ?? []).length).toBe(rows.length);
    expect((body.match(/<td>/g) ?? []).length).toBe(bodyRows.length * 2);
  });

  it("is built from a gate that is currently happy", () => {
    for (const check of checks) {
      expect(check.ok, check.complaint ?? check.label).toBe(true);
    }
  });

  it("refuses to build at all when a row has moved", () => {
    // A row the gate is unhappy with still renders as a plausible result, and
    // publishing it would put a rejected figure on a public page under the
    // heading "what is checked". A break sweep caught this: the assertion
    // above is about the *gate*, and left the generator free to publish a
    // failing one, so it could not fail for the right reason.
    const first = checks[0];
    expect(first).toBeDefined();
    const doctored = [
      { ...(first as Check), ok: false, complaint: "moved to 1e-2" },
      ...checks.slice(1),
    ];
    expect(() => buildSite(doctored)).toThrow(/unhappy with 1 row/);
    expect(() => buildSite(doctored)).toThrow(/moved to 1e-2/);
    // And it still builds from a set it is happy with, so the guard is not
    // simply refusing everything.
    expect(buildSite(checks).size).toBe(2);
  });
});

describe("the headline figures are read out of the row they summarise", () => {
  const sufficiency = checks.find((check) => check.label.includes("same total score"));

  it("found the row at all", () => {
    expect(sufficiency).toBeDefined();
  });

  it("quotes the spread the gate measured, not a number typed beside it", () => {
    const spread = /spans ([\d.]+) logits/.exec(sufficiency?.result ?? "")?.[1];
    expect(spread).toBeDefined();
    expect(page).toContain(`<strong>${spread}</strong>`);
    // And it is a spread worth leading with: the whole point of the row is
    // that the two models disagree by something a reader would mind.
    expect(Number(spread)).toBeGreaterThan(1);
  });

  it("quotes the pattern count the gate measured", () => {
    const patterns = /over all ([\d,]+) patterns/.exec(sufficiency?.result ?? "")?.[1];
    expect(patterns).toBeDefined();
    expect(page).toContain(`over all ${patterns} of them`);
  });

  it("says zero for the Rasch side, which is the claim that row makes", () => {
    expect(page).toContain("<strong>0</strong>");
    expect(sufficiency?.result).toContain("the same ability");
  });
});

describe("every count is read off the thing it counts", () => {
  it("agrees with the command table", () => {
    const size = counts();
    expect(size.commands).toBe(COMMAND_NAMES.length);
    expect(size.commands).toBeGreaterThan(5);
    expect(page).toContain(`across ${size.commands} command-line tools`);
  });

  it("agrees with what the package exports", () => {
    const size = counts();
    expect(size.exports).toBe(Object.keys(calibrate).length);
    expect(size.exports).toBeGreaterThan(100);
    expect(page).toContain(`<strong>${size.exports}</strong>`);
    expect(page).toContain(`exports\n  ${size.exports} names`);
  });
});

describe("the suite size the roadmap states", () => {
  // The same stale-prose check a sibling repository added after finding its
  // own count was two hundred tests out of date. A number in a document with
  // nothing recomputing it stops being true quietly.
  const roadmap = readFileSync(new URL("../ROADMAP.md", import.meta.url), "utf8");
  const stated = /(\d[\d,]*) tests across (\d+) files/.exec(roadmap);
  const testFiles = readdirSync(new URL("../test/", import.meta.url)).filter((name) =>
    name.endsWith(".test.ts"),
  );

  it("names a size at all", () => {
    expect(stated).not.toBeNull();
  });

  it("counts the files exactly, because that is countable", () => {
    expect(Number(stated?.[2])).toBe(testFiles.length);
  });

  it("counts the tests to within a quarter", () => {
    // A suite cannot count itself from inside, so this parses the call sites
    // instead. That undercounts -- a table-driven block is one call site and
    // many tests -- which is why the bound is a bound.
    let callSites = 0;
    for (const name of testFiles) {
      const source = readFileSync(new URL(`../test/${name}`, import.meta.url), "utf8");
      callSites += (source.match(/^\s*(?:it|test)(?:\.each\([^)]*\))?\s*\(/gm) ?? []).length;
    }
    const claimed = Number((stated?.[1] ?? "0").replace(/,/g, ""));
    expect(callSites).toBeGreaterThan(100);
    expect(claimed).toBeGreaterThan(callSites * 0.75);
    expect(claimed).toBeLessThan(callSites * 2.5);
  });
});
