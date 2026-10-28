/**
 * Write the site out, and put the built interface next to it.
 *
 *     npm run docs
 *
 * Kept apart from `docs/site.ts` so that the generator itself touches no
 * filesystem and the tests can call it directly.
 *
 * The interface is expected to be compiled already — `npm run web:build` —
 * because emitting it is `tsc`'s job and this script should not have an
 * opinion about how that was done. What it does have an opinion about is the
 * *shape* of the copy: `web/index.html` loads `./build/web/app/main.js`, so
 * the published directory has to keep that relative arrangement intact rather
 * than flatten it, and the page and the stylesheet sit beside it.
 */

import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { APP, buildSite } from "./site.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "site");
const WEB = join(ROOT, "web");
const BUILD = join(WEB, "build");

// Cleared rather than written over: a page the generator no longer produces
// should not survive in the published site because it was there last time.
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const files = buildSite();
for (const [name, content] of files) {
  writeFileSync(join(OUT, name), content, "utf8");
}

if (!existsSync(BUILD)) {
  process.stderr.write(
    `the interface has not been compiled: ${BUILD} is missing — run \`npm run web:build\` first\n`,
  );
  process.exit(1);
}

const app = join(OUT, APP);
mkdirSync(app, { recursive: true });
cpSync(BUILD, join(app, "build"), { recursive: true });
for (const name of ["index.html", "styles.css"]) {
  cpSync(join(WEB, name), join(app, name));
}

process.stdout.write(
  `site: ${[...files.keys()].join(", ")} and the interface under ${APP}\n`,
);
