import type { Response } from '../models/response.js';

/**
 * A missing response.
 *
 * Distinct from an incorrect one, and the distinction is not pedantic: an item
 * a candidate never reached carries no information about their ability, while an
 * item they reached and failed carries a great deal. Scoring the first as the
 * second is the single most common way to make a calibration wrong, and it
 * biases hardest against the candidates who ran out of time.
 */
export const MISSING = null;
export type Cell = Response | null;

export interface ResponseMatrixInit {
  /** One row per person, one column per item. */
  readonly rows: readonly (readonly Cell[])[];
  /** Person identifiers. Defaults to `p-0`, `p-1`, ... */
  readonly personIds?: readonly string[];
  /** Item identifiers. Defaults to `i-0`, `i-1`, ... */
  readonly itemIds?: readonly string[];
}

/**
 * A rectangular matrix of dichotomous responses.
 *
 * Immutable, and validated at construction: a ragged matrix, a value that is
 * neither 0, 1 nor missing, or a duplicated identifier all throw here rather
 * than surfacing as a strange parameter estimate several thousand iterations
 * later.
 */
export class ResponseMatrix {
  readonly #rows: readonly (readonly Cell[])[];
  readonly personIds: readonly string[];
  readonly itemIds: readonly string[];

  constructor(init: ResponseMatrixInit) {
    const { rows } = init;
    if (rows.length === 0) throw new RangeError('ResponseMatrix: at least one person is required');
    const width = (rows[0] as readonly Cell[]).length;
    if (width === 0) throw new RangeError('ResponseMatrix: at least one item is required');

    for (const [index, row] of rows.entries()) {
      if (row.length !== width) {
        throw new RangeError(
          `ResponseMatrix: row ${index} has ${row.length} responses, expected ${width}`,
        );
      }
      for (const [column, cell] of row.entries()) {
        if (cell !== 0 && cell !== 1 && cell !== MISSING) {
          throw new RangeError(
            `ResponseMatrix: cell (${index}, ${column}) must be 0, 1 or null, received ${String(cell)}`,
          );
        }
      }
    }

    this.personIds = Object.freeze(
      init.personIds === undefined
        ? rows.map((_, index) => `p-${index}`)
        : checkIds(init.personIds, rows.length, 'person'),
    );
    this.itemIds = Object.freeze(
      init.itemIds === undefined
        ? Array.from({ length: width }, (_, index) => `i-${index}`)
        : checkIds(init.itemIds, width, 'item'),
    );
    this.#rows = Object.freeze(rows.map((row) => Object.freeze([...row])));
  }

  get personCount(): number {
    return this.#rows.length;
  }

  get itemCount(): number {
    return this.itemIds.length;
  }

  /** The response of person `person` to item `item`, or `null` if missing. */
  at(person: number, item: number): Cell {
    const row = this.#rows[person];
    if (row === undefined) throw new RangeError(`ResponseMatrix: no person at index ${person}`);
    const cell = row[item];
    if (cell === undefined) throw new RangeError(`ResponseMatrix: no item at index ${item}`);
    return cell;
  }

  row(person: number): readonly Cell[] {
    const row = this.#rows[person];
    if (row === undefined) throw new RangeError(`ResponseMatrix: no person at index ${person}`);
    return row;
  }

  column(item: number): Cell[] {
    if (item < 0 || item >= this.itemCount) {
      throw new RangeError(`ResponseMatrix: no item at index ${item}`);
    }
    return this.#rows.map((row) => row[item] as Cell);
  }

  /** Number correct and number answered, for one person. */
  personScore(person: number): { correct: number; answered: number } {
    return tally(this.row(person));
  }

  /** Number correct and number answered, for one item. */
  itemScore(item: number): { correct: number; answered: number } {
    return tally(this.column(item));
  }

  /** Every row, as a plain array of arrays. */
  toArray(): Cell[][] {
    return this.#rows.map((row) => [...row]);
  }
}

function tally(cells: readonly Cell[]): { correct: number; answered: number } {
  let correct = 0;
  let answered = 0;
  for (const cell of cells) {
    if (cell === MISSING) continue;
    answered += 1;
    if (cell === 1) correct += 1;
  }
  return { correct, answered };
}

function checkIds(ids: readonly string[], expected: number, label: string): string[] {
  if (ids.length !== expected) {
    throw new RangeError(
      `ResponseMatrix: ${ids.length} ${label} ids for ${expected} ${label}s`,
    );
  }
  const seen = new Set<string>();
  for (const id of ids) {
    if (id.length === 0) throw new RangeError(`ResponseMatrix: empty ${label} id`);
    if (seen.has(id)) throw new RangeError(`ResponseMatrix: duplicate ${label} id "${id}"`);
    seen.add(id);
  }
  return [...ids];
}

export interface ParseCsvOptions {
  /** Treat the first row as item ids. Default true. */
  readonly header?: boolean;
  /** Treat the first column as person ids. Default false. */
  readonly index?: boolean;
  /** Field separator. Default a comma. */
  readonly separator?: string;
  /** Values read as a missing response, in addition to an empty field. */
  readonly missing?: readonly string[];
}

/**
 * Parse a response matrix from CSV text.
 *
 * Deliberately strict about what a cell may contain. A response file is nearly
 * always exported from somewhere else, and the interesting failure is not a
 * malformed file — that is obvious — but a file where one column came through as
 * `TRUE`/`FALSE` or `Y`/`N` and would be silently coerced. Anything that is not
 * a recognised score or an explicit missing marker throws, with its position.
 */
export function parseResponseCsv(text: string, options: ParseCsvOptions = {}): ResponseMatrix {
  const header = options.header ?? true;
  const index = options.index ?? false;
  const separator = options.separator ?? ',';
  const missing = new Set(['', 'NA', 'na', '.', ...(options.missing ?? [])]);

  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  if (lines.length === 0) throw new RangeError('parseResponseCsv: no rows found');

  let itemIds: string[] | undefined;
  let start = 0;
  if (header) {
    const cells = (lines[0] as string).split(separator).map((cell) => cell.trim());
    itemIds = index ? cells.slice(1) : cells;
    start = 1;
  }

  const personIds: string[] = [];
  const rows: Cell[][] = [];
  for (let line = start; line < lines.length; line += 1) {
    const cells = (lines[line] as string).split(separator).map((cell) => cell.trim());
    if (index) {
      const id = cells.shift();
      if (id === undefined || id.length === 0) {
        throw new RangeError(`parseResponseCsv: row ${line + 1} has no person id`);
      }
      personIds.push(id);
    }
    rows.push(
      cells.map((cell, column) => {
        if (missing.has(cell)) return MISSING;
        if (cell === '0') return 0;
        if (cell === '1') return 1;
        throw new RangeError(
          `parseResponseCsv: row ${line + 1}, column ${column + 1}: ` +
            `expected 0, 1 or a missing marker, received "${cell}"`,
        );
      }),
    );
  }

  return new ResponseMatrix({
    rows,
    ...(itemIds === undefined ? {} : { itemIds }),
    ...(index ? { personIds } : {}),
  });
}
