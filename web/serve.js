// A static file server for the web interface, in the standard library.
//
// The interface has no bundler and no runtime dependencies: `tsc` emits ES
// modules whose import specifiers are already browser-resolvable, and the
// browser loads them directly. All that is missing is something to answer HTTP
// with the right content types, which is this.

import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';

const ROOT = resolve(import.meta.dirname, '..');
const PORT = Number(process.env.PORT ?? 5173);

const CONTENT_TYPES = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.map', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.ico', 'image/x-icon'],
  ['.woff2', 'font/woff2'],
]);

/**
 * Resolve a request path to a file inside ROOT, or null if it escapes.
 *
 * A static server that joins user input onto a directory and opens the result
 * will serve anything on the disk to anyone who can write `../`. Normalising
 * first and then checking that the result is still under ROOT is the whole fix.
 */
function resolveWithinRoot(urlPath) {
  const decoded = decodeURIComponent(urlPath.split('?')[0] ?? '/');
  const relative = normalize(decoded).replace(/^(\.\.[/\\])+/, '');
  const candidate = resolve(join(ROOT, relative));
  return candidate === ROOT || candidate.startsWith(ROOT + sep) ? candidate : null;
}

const server = createServer(async (request, response) => {
  const target = resolveWithinRoot(request.url ?? '/');
  if (target === null) {
    response.writeHead(403, { 'content-type': 'text/plain; charset=utf-8' });
    response.end('Outside the served directory\n');
    return;
  }

  let file = target;
  try {
    const info = await stat(file);
    if (info.isDirectory()) {
      file = join(file, 'index.html');
      await stat(file);
    }
  } catch {
    response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    response.end(`Not found: ${request.url}\n`);
    return;
  }

  response.writeHead(200, {
    'content-type': CONTENT_TYPES.get(extname(file)) ?? 'application/octet-stream',
    'cache-control': 'no-store',
  });
  createReadStream(file).pipe(response);
});

server.listen(PORT, () => {
  process.stdout.write(`calibrate web interface on http://localhost:${PORT}/web/\n`);
});
