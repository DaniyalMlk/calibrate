import { MISSING, ResponseMatrix, type Cell } from './matrix.js';

/** Why a person or item was removed before calibration. */
export type ExclusionReason =
  | 'perfect-score'
  | 'zero-score'
  | 'answered-by-all'
  | 'answered-by-none'
  | 'no-responses';

export interface Exclusion {
  readonly id: string;
  readonly reason: ExclusionReason;
  /** The screening pass that removed it, counting from 1. */
  readonly pass: number;
}

export interface ScreenResult {
  /** The matrix with extreme rows and columns removed. */
  readonly matrix: ResponseMatrix;
  readonly excludedPersons: readonly Exclusion[];
  readonly excludedItems: readonly Exclusion[];
  /** How many passes were needed before nothing more was removed. */
  readonly passes: number;
}

/**
 * Remove the rows and columns that carry no information about the parameters.
 *
 * A person who answered everything correctly has a likelihood that is monotone
 * in ability — there is no finite estimate of how able they are, only a lower
 * bound. An item everybody answered correctly has the same problem in the other
 * direction. Joint estimation with either present does not fail loudly; it walks
 * off towards infinity, dragging the scale and every other parameter with it.
 *
 * Screening has to iterate, because the two interact. Dropping the strongest
 * candidates can leave an item that nobody remaining answered correctly, which
 * then has to go too, which can in turn make another person extreme. The loop
 * runs until a pass removes nothing, and the passes are recorded so a bank owner
 * can see whether the screen took one clean pass or unravelled the data.
 */
export function screenExtremes(matrix: ResponseMatrix): ScreenResult {
  let rows: Cell[][] = matrix.toArray();
  let personIds: string[] = [...matrix.personIds];
  let itemIds: string[] = [...matrix.itemIds];
  const excludedPersons: Exclusion[] = [];
  const excludedItems: Exclusion[] = [];

  for (let pass = 1; ; pass += 1) {
    const keptPersons: number[] = [];
    for (const [index, row] of rows.entries()) {
      const { correct, answered } = tally(row);
      const reason = extremeReason(correct, answered, 'person');
      if (reason === null) keptPersons.push(index);
      else excludedPersons.push({ id: personIds[index] as string, reason, pass });
    }

    const keptItems: number[] = [];
    for (let column = 0; column < itemIds.length; column += 1) {
      // Item extremity is judged on the persons who survive this pass, not on
      // the original matrix: an item is only uninformative relative to the
      // people who are actually being calibrated against it.
      const { correct, answered } = tally(keptPersons.map((row) => (rows[row] as Cell[])[column] as Cell));
      const reason = extremeReason(correct, answered, 'item');
      if (reason === null) keptItems.push(column);
      else excludedItems.push({ id: itemIds[column] as string, reason, pass });
    }

    if (keptPersons.length === rows.length && keptItems.length === itemIds.length) {
      return {
        matrix: new ResponseMatrix({ rows, personIds, itemIds }),
        excludedPersons,
        excludedItems,
        passes: pass,
      };
    }

    if (keptPersons.length === 0 || keptItems.length === 0) {
      throw new RangeError(
        'screenExtremes: screening removed every person or every item; ' +
          'the data carry no information about any parameter',
      );
    }

    rows = keptPersons.map((row) => keptItems.map((column) => (rows[row] as Cell[])[column] as Cell));
    personIds = keptPersons.map((row) => personIds[row] as string);
    itemIds = keptItems.map((column) => itemIds[column] as string);
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

function extremeReason(
  correct: number,
  answered: number,
  kind: 'person' | 'item',
): ExclusionReason | null {
  if (answered === 0) return 'no-responses';
  if (correct === answered) return kind === 'person' ? 'perfect-score' : 'answered-by-all';
  if (correct === 0) return kind === 'person' ? 'zero-score' : 'answered-by-none';
  return null;
}
