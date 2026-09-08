import { describe, expect, it } from 'vitest';
import { MISSING, parseResponseCsv, ResponseMatrix, type Cell } from '../src/calibration/matrix.js';
import { screenExtremes } from '../src/calibration/screen.js';

const rows: Cell[][] = [
  [1, 1, 0],
  [0, 1, 0],
  [1, MISSING, 1],
];

describe('ResponseMatrix', () => {
  const matrix = new ResponseMatrix({ rows, itemIds: ['a', 'b', 'c'] });

  it('reports its shape and identifiers', () => {
    expect(matrix.personCount).toBe(3);
    expect(matrix.itemCount).toBe(3);
    expect(matrix.itemIds).toEqual(['a', 'b', 'c']);
    expect(matrix.personIds).toEqual(['p-0', 'p-1', 'p-2']);
  });

  it('reads cells, rows and columns', () => {
    expect(matrix.at(0, 0)).toBe(1);
    expect(matrix.at(2, 1)).toBe(MISSING);
    expect(matrix.row(1)).toEqual([0, 1, 0]);
    expect(matrix.column(2)).toEqual([0, 0, 1]);
  });

  it('excludes missing responses from both scores', () => {
    // Person 2 answered two items and got both right.
    expect(matrix.personScore(2)).toEqual({ correct: 2, answered: 2 });
    // Item b was answered by two people, both correctly.
    expect(matrix.itemScore(1)).toEqual({ correct: 2, answered: 2 });
  });

  it('copies its rows rather than aliasing the caller’s', () => {
    const mutable: Cell[][] = [[1, 0]];
    const built = new ResponseMatrix({ rows: mutable });
    mutable[0] = [0, 0];
    expect(built.row(0)).toEqual([1, 0]);
  });

  it('rejects a ragged matrix', () => {
    expect(() => new ResponseMatrix({ rows: [[1, 0], [1]] })).toThrow(/row 1 has 1 responses/);
  });

  it('rejects a cell that is not a score or a missing marker', () => {
    expect(() => new ResponseMatrix({ rows: [[1, 2 as unknown as Cell]] })).toThrow(/must be 0, 1 or null/);
  });

  it('rejects an empty matrix in either direction', () => {
    expect(() => new ResponseMatrix({ rows: [] })).toThrow(RangeError);
    expect(() => new ResponseMatrix({ rows: [[]] })).toThrow(RangeError);
  });

  it('rejects mismatched, duplicated or empty identifiers', () => {
    expect(() => new ResponseMatrix({ rows, itemIds: ['a', 'b'] })).toThrow(/2 item ids for 3 items/);
    expect(() => new ResponseMatrix({ rows, itemIds: ['a', 'a', 'c'] })).toThrow(/duplicate item id/);
    expect(() => new ResponseMatrix({ rows, personIds: ['x', '', 'z'] })).toThrow(/empty person id/);
  });

  it('rejects out-of-range indices', () => {
    expect(() => matrix.at(9, 0)).toThrow(RangeError);
    expect(() => matrix.at(0, 9)).toThrow(RangeError);
    expect(() => matrix.column(9)).toThrow(RangeError);
  });
});

describe('parseResponseCsv', () => {
  it('reads a header of item ids by default', () => {
    const matrix = parseResponseCsv('q1,q2,q3\n1,0,1\n0,0,1\n');
    expect(matrix.itemIds).toEqual(['q1', 'q2', 'q3']);
    expect(matrix.personCount).toBe(2);
    expect(matrix.row(0)).toEqual([1, 0, 1]);
  });

  it('reads person ids from the first column when asked', () => {
    const matrix = parseResponseCsv('id,q1,q2\nalice,1,0\nbob,0,1\n', { index: true });
    expect(matrix.personIds).toEqual(['alice', 'bob']);
    expect(matrix.itemIds).toEqual(['q1', 'q2']);
  });

  it('treats blanks and the usual markers as missing', () => {
    const matrix = parseResponseCsv('q1,q2,q3,q4\n1,,NA,.\n', { missing: ['-'] });
    expect(matrix.row(0)).toEqual([1, MISSING, MISSING, MISSING]);
    expect(parseResponseCsv('q1\n-\n', { missing: ['-'] }).at(0, 0)).toBe(MISSING);
  });

  it('ignores blank lines and surrounding whitespace', () => {
    const matrix = parseResponseCsv('\n q1 , q2 \n 1 , 0 \n\n 0 , 1 \n\n');
    expect(matrix.itemIds).toEqual(['q1', 'q2']);
    expect(matrix.personCount).toBe(2);
  });

  it('accepts an alternative separator and a headerless file', () => {
    const matrix = parseResponseCsv('1\t0\n0\t1', { header: false, separator: '\t' });
    expect(matrix.itemIds).toEqual(['i-0', 'i-1']);
    expect(matrix.row(1)).toEqual([0, 1]);
  });

  it('refuses to guess at a value it does not recognise', () => {
    // The failure that matters: a column exported as TRUE/FALSE would be
    // silently coerced by anything more forgiving.
    expect(() => parseResponseCsv('q1,q2\n1,TRUE\n')).toThrow(
      /row 2, column 2: expected 0, 1 or a missing marker, received "TRUE"/,
    );
  });

  it('rejects an empty file and a row with no person id', () => {
    expect(() => parseResponseCsv('   \n\n')).toThrow(RangeError);
    expect(() => parseResponseCsv('id,q1\n,1\n', { index: true })).toThrow(/no person id/);
  });
});

describe('screenExtremes', () => {
  it('keeps a matrix with nothing extreme untouched, in one pass', () => {
    const matrix = new ResponseMatrix({
      rows: [
        [1, 0],
        [0, 1],
      ],
    });
    const result = screenExtremes(matrix);
    expect(result.matrix.personCount).toBe(2);
    expect(result.matrix.itemCount).toBe(2);
    expect(result.excludedPersons).toEqual([]);
    expect(result.passes).toBe(1);
  });

  it('removes perfect and zero scoring persons', () => {
    const matrix = new ResponseMatrix({
      rows: [
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
      ],
      personIds: ['perfect', 'mixed-a', 'zero', 'mixed-b'],
    });
    const result = screenExtremes(matrix);
    expect(result.matrix.personIds).toEqual(['mixed-a', 'mixed-b']);
    expect(result.excludedPersons.map((exclusion) => [exclusion.id, exclusion.reason])).toEqual([
      ['perfect', 'perfect-score'],
      ['zero', 'zero-score'],
    ]);
  });

  it('removes items nobody or everybody answered correctly', () => {
    // Six people whose answers to q1, q2 and q3 are mixed in every direction,
    // so nothing cascades once the two extreme items are gone.
    const matrix = new ResponseMatrix({
      rows: [
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
      ],
      itemIds: ['all-right', 'q1', 'q2', 'q3', 'all-wrong'],
    });
    const result = screenExtremes(matrix);
    expect(result.matrix.itemIds).toEqual(['q1', 'q2', 'q3']);
    expect(result.matrix.personCount).toBe(6);
    expect(result.excludedItems.map((exclusion) => [exclusion.id, exclusion.reason])).toEqual([
      ['all-right', 'answered-by-all'],
      ['all-wrong', 'answered-by-none'],
    ]);
  });

  it('iterates, because removing one extreme can create another', () => {
    // `perfect` answers everything, and is the only person to get q4 right.
    // Dropping them leaves q4 answered by nobody, and dropping q4 in turn
    // leaves `nearly` — whose only wrong answer was q4 — with a perfect score.
    // That last removal cannot happen until a second pass, which is the whole
    // reason screening is a loop rather than a single sweep.
    const matrix = new ResponseMatrix({
      rows: [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
      ],
      personIds: ['a', 'b', 'c', 'd', 'e', 'f', 'nearly', 'perfect'],
      itemIds: ['q1', 'q2', 'q3', 'q4'],
    });
    const result = screenExtremes(matrix);

    expect(result.passes).toBe(3);
    expect(result.matrix.personIds).toEqual(['a', 'b', 'c', 'd', 'e', 'f']);
    expect(result.matrix.itemIds).toEqual(['q1', 'q2', 'q3']);
    expect(result.excludedPersons).toEqual([
      { id: 'perfect', reason: 'perfect-score', pass: 1 },
      { id: 'nearly', reason: 'perfect-score', pass: 2 },
    ]);
    expect(result.excludedItems).toEqual([
      { id: 'q4', reason: 'answered-by-none', pass: 1 },
    ]);
  });

  it('treats an all-missing row as uninformative rather than as a zero score', () => {
    const matrix = new ResponseMatrix({
      rows: [
        [1, 0],
        [0, 1],
        [MISSING, MISSING],
      ],
      personIds: ['a', 'b', 'absent'],
    });
    const result = screenExtremes(matrix);
    expect(result.excludedPersons).toEqual([{ id: 'absent', reason: 'no-responses', pass: 1 }]);
  });

  it('refuses data that screening empties completely', () => {
    expect(() =>
      screenExtremes(
        new ResponseMatrix({
          rows: [
            [1, 1],
            [0, 0],
          ],
        }),
      ),
    ).toThrow(/carry no information/);
  });
});
