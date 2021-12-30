#!/usr/bin/env python

import numpy as np
import random

DEBUG = False


class Status:
    UNKNOWN = 0
    SOLVED = 1
    UNSOLVED = 2
    UNSOLVABLE = 3


# https://www.sudokuoftheday.com/about/difficulty/
# https://www.sudokuoftheday.com/techniques/
class StepType:
    SingleCandidate = 'scd'
    SinglePosition = 'spo'
    CandidateLines = 'cdl'
    DoublePairs = 'dbp'
    MultipleLines = 'mtl'
    NakedPair = 'nkp'
    HiddenPair = 'hdp'
    NakedTriple = 'nkt'
    HiddenTriple = 'hdt'
    XWing = 'xwg'
    ForcingChains = 'fcc'
    NakedQuad = 'nkq'
    HiddenQuad = 'hdq'
    Swordfish = 'swf'
    Guessing = 'gsg'
    CleanCandidates = 'cct'
    Unknown = 'unk'
    ScanCandidates = 'scan'
    InitState = 'init'


StepsCosts = {
    StepType.SingleCandidate: {'first': 100, 'seconds': 100},
    StepType.SinglePosition: {'first': 100, 'seconds': 100},
    StepType.CandidateLines: {'first': 350, 'seconds': 200},
    StepType.DoublePairs: {'first': 500, 'seconds': 250},
    StepType.MultipleLines: {'first': 700, 'seconds': 400},
    StepType.NakedPair: {'first': 750, 'seconds': 500},
    StepType.HiddenPair: {'first': 1500, 'seconds': 1200},
    StepType.NakedTriple: {'first': 2000, 'seconds': 1400},
    StepType.HiddenTriple: {'first': 2400, 'seconds': 1600},
    StepType.XWing: {'first': 2800, 'seconds': 1600},
    StepType.ForcingChains: {'first': 4200, 'seconds': 2100},
    StepType.NakedQuad: {'first': 5000, 'seconds': 4000},
    StepType.HiddenQuad: {'first': 7000, 'seconds': 5000},
    StepType.Swordfish: {'first': 8000, 'seconds': 6000},
    StepType.Guessing: {'first': 9000, 'seconds': 10000},
    StepType.CleanCandidates: {'first': 0, 'seconds': 0},
    StepType.Unknown: {'first': 0, 'seconds': 0},
    StepType.ScanCandidates: {'first': 0, 'seconds': 0},
    StepType.InitState: {'first': 0, 'seconds': 0},
}


class Matrix:
    def __init__(self, v):
        self.v = v

    def get_column(self, index):
        return self.v[:, index]

    def get_row(self, index):
        return self.v[index, :]

    def __str__(self):
        tmp = ''
        for i, row in enumerate(self.v):
            if i in [3, 6]:
                tmp += '------+------+------\n'
            for j, el in enumerate(row):
                if j in [3, 6]:
                    tmp += '|'
                tmp += f"{self.v[i, j]} "
            tmp += '\n'
        return tmp

    def as_str(self):
        return ''.join(self.v.flatten())


class SudokuMatrix(Matrix):
    def __init__(self, str_values: str = None, list_values: list = None, dtype='<U1') -> None:
        if str_values:
            super().__init__(np.array(list(str_values), dtype=dtype).reshape((9, 9)))
        elif list_values:
            super().__init__(np.array(list_values, dtype=dtype).reshape((9, 9)))

    def get_block(self, i, j):
        return Matrix(self.v[i * 3:i * 3 + 3, j * 3:j * 3 + 3])


class SudokuMatrixMultiValue(SudokuMatrix):
    @classmethod
    def from_str(cls, string_repr):
        values, symb = string_repr.split('|')
        return cls(symbols=sorted(list(set(symb))),
                   list_values=values.split(';'))

    def __init__(self, symbols: list, list_values: list = None) -> None:
        if list_values:
            super().__init__(list_values=list_values, dtype='<U10')
        else:
            super().__init__(list_values=['' for _ in range(81)], dtype='<U10')
        self.symbols = symbols

    def as_str(self):
        return ';'.join(self.v.flatten()) + '|' + ''.join(self.symbols)

    def __str__(self):
        tmp = ''
        for i, row in enumerate(self.v):
            if i in [3, 6]:
                tmp += '---------------+---------------+-------------\n'
            for ix in range(3):
                for j in range(9):
                    if j in [3, 6]:
                        tmp += '|'
                    for jx in range(1, 4):
                        index = ix * 3 + jx
                        if str(index) in self.v[i, j]:
                            tmp += f"{index}"
                        else:
                            tmp += '·'
                    tmp += '  '
                tmp += '\n'
            if i not in [2, 5]:
                tmp += '\n'
            #     tmp += "- - - - - - -|- - - - - - -|- - - - - - -\n"
        return tmp


class CandidatesMatrix(SudokuMatrixMultiValue):

    def found_value_clean(self, row, col, value):
        """Clean possible candidates of values when a cell value has been fixed"""
        modified = False
        changes = []
        self.v[row, col] = ''
        for j, el in enumerate(self.get_row(row)):
            if value in el:
                self.v[row, j] = el.replace(value, '')
                changes.append((row, j))
                modified = True
        for i, el in enumerate(self.get_column(col)):
            if value in el:
                self.v[i, col] = el.replace(value, '')
                changes.append((i, col))
                modified = True
        bi = row // 3
        bj = col // 3
        for i in range(3 * bi, 3 * bi + 3):
            for j in range(3 * bj, 3 * bj + 3):
                if value in self.v[i, j]:
                    self.v[i, j] = self.v[i, j].replace(value, '')
                    changes.append((i, j))
                    modified = True
        return modified

    def is_broken(self):
        """
        Check if several cells in a row, column or block have the same unique value as candidate
        """
        errors = []
        for bi in range(3):
            for bj in range(3):
                b = self.get_block(bi, bj)
                flat = b.v.flatten()
                for s in self.symbols:
                    arr = [s in x and len(x) == 1 for x in flat]
                    if np.count_nonzero(np.array(arr)) > 1:
                        errors.append(f"Found several cells on block {bi * 3 + bj} with unique candidate '{s}'")
        for ri in range(len(self.symbols)):
            b = self.get_row(ri)
            flat = b.flatten()
            for s in self.symbols:
                arr = [s in x and len(x) == 1 for x in flat]
                if np.count_nonzero(np.array(arr)) > 1:
                    errors.append(f"Found several cells on row {ri} with unique candidate '{s}'")
        for ci in range(len(self.symbols)):
            b = self.get_column(ci)
            flat = b.flatten()
            for s in self.symbols:
                arr = [s in x and len(x) == 1 for x in flat]
                if np.count_nonzero(np.array(arr)) > 1:
                    errors.append(f"Found several cells on column {ri} with unique candidate '{s}'")
        return errors


class Resolutions:
    solved = 0
    unsolved = 1
    unsolvable = -1

    @classmethod
    def as_str(cls, value):
        if value == cls.solved:
            return "solved"
        if value == cls.unsolved:
            return "unsolved"
        if value == cls.unsolvable:
            return "unsolvable"
        return "unknown"


class Step:
    def __init__(self, solutions_str, candidates_str, step_type: str, msg='', prev_step=None,
                 index=0):
        self.solutions = SudokuMatrix(solutions_str)
        self.candidates = CandidatesMatrix.from_str(candidates_str)
        self.step_type = step_type
        self.msg = msg
        self.prev = prev_step
        self.acc_types = [self.step_type]
        if prev_step:
            prev_step.next = self
            self.acc_types.extend(prev_step.acc_types)
        self.next = None
        self.index = index
        self.symbols = [a for a in list(set(self.solutions.v.flatten())) if a != '-']
        self.fill_symbols()

    @property
    def cost(self):
        _repeat = 'first'
        if self.acc_types.count(self.step_type) > 1:
            _repeat = 'seconds'
        return StepsCosts[self.step_type][_repeat]

    def fill_symbols(self):
        if len(self.symbols) < 9:
            for i in range(9):
                if str(i) not in self.symbols:
                    self.symbols.append(str(i))
                if len(self.symbols) == 9:
                    break

    def is_broken(self):
        for i, row in enumerate(self.solutions.v):
            for j, cell in enumerate(row):
                if cell == '-':
                    if len(self.candidates.v[i, j]) == 0:
                        return i, j
        ans = self.candidates.is_broken()
        if ans:
            return ans
        return False

    def solved(self) -> bool:
        if '-' in self.solutions.v.flatten():
            return False
        s = set(self.symbols)
        for r in range(9):
            if set(self.solutions.get_row(r)) != s:
                return False
        for c in range(9):
            if set(self.solutions.get_column(c)) != s:
                return False
        return True


class StepResolution:
    all = 0
    mid = 5
    none = 100


class Sudoku:

    def __init__(self, init_values: str, candidates_str: str = None, step_resolution: int = StepResolution.all,
                 init_step: Step = None) -> None:
        if len(init_values) != 81:
            raise ValueError("Invalid Sudoku input")
        self.sr = step_resolution
        self.init_str = init_values
        self._clues = SudokuMatrix(init_values)
        self.solve_tree = []

        self._symbols = None
        self.fill_symbols()

        if candidates_str is None:
            self._candidates = CandidatesMatrix(symbols=self._symbols, list_values=None)
            self._scanned_candidates = False
        else:
            self._candidates = CandidatesMatrix.from_str(string_repr=candidates_str)
            self._scanned_candidates = True

        self.iterations = 0
        self.costs = {'iterations': [],
                      'guesses': 0}
        if init_step is None:
            self.steps = [Step(self._clues.as_str(), self._candidates.as_str(), step_type=StepType.InitState,
                          msg='Initial State', prev_step=None)]
        else:
            self.steps = [init_step]

        self.valid_solutions = []

    @property
    def clues(self):
        return SudokuMatrix(self._clues.as_str())

    @property
    def candidates(self):
        return CandidatesMatrix.from_str(self._candidates.as_str())

    @property
    def missing_cells(self) -> zip:
        return zip(*np.where(self._clues.v == '-'))

    @property
    def num_of_missing_cells(self):
        return self.as_str()[0].count('-')

    def fill_symbols(self):
        self._symbols = sorted(list(set(self.init_str)))
        if '-' in self._symbols:
            self._symbols.pop(self._symbols.index('-'))
        if len(self._symbols) < 9:
            # Symbols could be any ascii character, but we fill missing ones with numbers
            for i in range(1, 10):
                if str(i) not in self._symbols:
                    self._symbols.append(str(i))
                if len(self._symbols) == 9:
                    break
        elif len(self._symbols) > 9:
            raise ValueError("Too many symbols")

    def as_str(self):
        return self._clues.as_str(), self._candidates.as_str()

    @property
    def cost(self):
        cost = 0
        for ste in self.steps:
            cost += ste.cost
        return cost

    def add_step(self, step_type: str, message, resolution: int = 100):
        if resolution > self.sr:
            self.steps.append(Step(self._clues.as_str(), self._candidates.as_str(), step_type=step_type, msg=message,
                                   prev_step=self.steps[-1], index=len(self.steps)))

    def clean_candidates(self):
        """Clean candidates based on solution values already found"""
        modified = False
        for r in range(9):
            for c in range(9):
                if self._clues.v[r, c] != '-':
                    modified |= self._candidates.found_value_clean(r, c, self._clues.v[r, c])
        if modified:
            self.add_step(step_type=StepType.CleanCandidates, message='Cleaned candidates', resolution=5)
            return True
        return False

    def clean_candidates_in_single_row_of_block(self, iterations=0, skip=[]) -> int:
        """
        Check if block contains candidate values in a single row of the block. If this is the case, that value
        can be removed from other blocks on that row.
        skip field is used to know which value, row and column have already been processed avoiding an infinite loop
        """
        changes = False
        for bi in range(3):
            for bj in range(3):
                bopt = self._candidates.get_block(bi, bj)
                for s in self._symbols:
                    if (bi, bj, s) in skip:
                        continue
                    bin = []
                    for brow in bopt.v:
                        bin.append([s in el for el in brow])
                    bin = np.array(bin)
                    byrows = np.count_nonzero(bin, axis=1)
                    if np.count_nonzero(byrows) == 1:
                        brix = np.where(byrows > 0)[0][0]
                        for i in range(9):
                            if i < bj * 3 or i > bj * 3 + 2:
                                if s in self._candidates.v[bi * 3 + brix, i]:
                                    changes = True
                                    self._candidates.v[bi * 3 + brix, i] = self._candidates.v[bi * 3 + brix, i].replace(
                                        s, '')
                        skip.append((bi, bj, s))
                        if changes:
                            # new_it = self.clean_candidates_in_single_row_of_block(iterations + 1, skip)
                            if iterations == 0:
                                # print("rows", skip)
                                self.add_step(step_type=StepType.CandidateLines,
                                              message=f"Found candidate value {s} only in a row of block {bi * 3 + bj} "
                                                      f"and cleaned outside it.\n")
                            return 1
        return iterations

    def clean_candidates_in_single_col_of_block(self, iterations=0, skip=[]) -> int:
        """
        Check if block contains a candidate value only in a column on the block. If this is the case, that value
        can be removed from other blocks on that column.
        skip field is used to know which values, row and column have already been processed
        """
        changes = False
        for bi in range(3):
            for bj in range(3):
                bopt = self._candidates.get_block(bi, bj)
                for s in self._symbols:
                    if (bi, bj, s) in skip:
                        continue
                    bin = []
                    for row in bopt.v:
                        bin.append([s in el for el in row])
                    bin = np.array(bin)

                    bycols = np.count_nonzero(bin, axis=0)
                    if np.count_nonzero(bycols) == 1:
                        bcix = np.where(bycols > 0)[0][0]

                        for i in range(9):
                            if i < bi * 3 or i > bi * 3 + 2:
                                if s in self._candidates.v[i, bj * 3 + bcix]:
                                    changes = True
                                    # print(bi, bj, s, bopt, bin, bycols, bcix)
                                    self._candidates.v[i, bj * 3 + bcix] = self._candidates.v[i, bj * 3 + bcix].replace(
                                        s, '')
                        skip.append((bi, bj, s))
                        if changes:
                            # new_it = self.clean_candidates_in_single_col_of_block(iterations + 1, skip)
                            if iterations == 0:
                                # print(skip)
                                self.add_step(
                                    step_type=StepType.CandidateLines,
                                    message=f"Found candidate value {s} only in a column of block {bi * 3 + bj} and "
                                            f"cleaned outside it.\n")
                            return 1
        return iterations

    def _clean_couples(self, bi, bj, ij_0, ij_1, values):
        values_in_row_or_col = False
        if ij_0[0] == ij_1[0]:
            values_in_row_or_col = True
            row = bi * 3 + ij_0[0]
            cols = (bj * 3 + ij_0[1], bj * 3 + ij_1[1])
            for j in range(9):
                if j not in cols:
                    for v in values:
                        self._candidates.v[row, j] = self._candidates.v[row, j].replace(v, '')

        elif ij_0[1] == ij_1[1]:
            values_in_row_or_col = True
            rows = (bi * 3 + ij_0[0], bi * 3 + ij_1[0])
            col = bj * 3 + ij_0[1]
            for i in range(9):
                if i not in rows:
                    for v in values:
                        self._candidates.v[i, col] = self._candidates.v[i, col].replace(v, '')
        if values_in_row_or_col:
            for i in range(3):
                for j in range(3):
                    if (i, j) not in [ij_0, ij_1]:
                        for v in values:
                            self._candidates.v[bi * 3 + i, bj * 3 + j] = self._candidates.v[
                                bi * 3 + i, bj * 3 + j].replace(v,
                                                                '')
        return values_in_row_or_col

    def clean_couples(self, iteration=0, skip=[]):
        """
        Look for cells in a column or row of a block with same two candidates. If two cells in a row (or column) have the
        same unique two candidates, it means that those numbers can't be anywhere else on the block or on the row (column)
        :param iterations:
        :param skip:
        :return:
        """
        for bi in range(3):
            for bj in range(3):
                bopt = self._candidates.get_block(bi, bj)
                bin = {}
                for i, row in enumerate(bopt.v):
                    for j, el in enumerate(row):
                        if len(el) == 2:
                            if (bi, bj, i, j, el) not in skip:
                                if el in bin:
                                    skip.append((bi, bj, i, j, el))
                                    if self._clean_couples(bi, bj, bin[el], (i, j), el):
                                        self.add_step(step_type=StepType.NakedPair,
                                                      message=f"Found same 2 values [{el}] in a row or a col on a "
                                                              f"block [{3 * bi + bj}. Cleaned it")
                                        # return self.clean_couples(iteration + 1, skip)
                                        return 1
                                else:
                                    bin[el] = (i, j)
        return iteration

    def solve_clean(self) -> tuple:
        if self._scanned_candidates is False:
            self.scan_blocks()
        while True:
            if self.solved():
                self.valid_solutions.append(self.as_str())
                return Resolutions.solved, None
            if self.is_broken():
                return Resolutions.unsolvable, self.is_broken()

            if self.clean_candidates():
                self.solve_tree.append(0)
                continue
            else:
                if self.find_single_candidates():
                    self.solve_tree.append(1)
                    continue
                else:
                    if self.find_column_single_candidates():
                        self.solve_tree.append(2)
                        continue
                    else:
                        if self.find_row_single_candidates():
                            self.solve_tree.append(3)
                            continue
                        else:
                            if self.find_block_single_candidates():
                                self.solve_tree.append(4)
                                continue
                            else:
                                if self.clean_candidates_in_single_col_of_block():
                                    self.solve_tree.append(5)
                                    continue
                                else:
                                    if self.clean_candidates_in_single_row_of_block():
                                        self.solve_tree.append(6)
                                        continue
                                    else:
                                        if self.clean_couples():
                                            self.solve_tree.append(7)
                                            continue
                                        else:
                                            break
        return Resolutions.unsolved, None

    def solve_guessing(self, randomize=False, find_all=False) -> int:
        # Solve using logic
        self.solve_clean()
        if self.solved():
            return Status.SOLVED
        if self.is_broken():
            return Status.UNSOLVABLE
        # Solve using guessing
        i, j, els = self._guess_numbers(randomize)
        self.costs['guesses'] += 1
        for e in els:
            # print(f"Guessing {i, j, e}, empty seats: {self.num_of_missing_cells}")
            s = Sudoku(self._clues.as_str(), candidates_str=self._candidates.as_str(),
                       init_step=self.steps[-1])
            s.set_solution(e, i, j)
            s.add_step(step_type=StepType.Guessing,
                       message=f"Guessing value {e} on position [{i}, {j}]")
            status = s.solve_guessing(find_all=find_all)
            if status == Status.SOLVED:
                if find_all is False:
                    self.steps.extend(s.steps)
                    self._clues = s._clues
                    self._candidates = s._candidates
                    self.costs['guesses'] += s.costs['guesses']
                    self.costs['iterations'].extend(s.costs['iterations'])
                    return status
                else:
                    if s.solved():
                        self.valid_solutions.append(s)
                    else:
                        self.valid_solutions.extend(s.valid_solutions)
        if len(self.valid_solutions) > 0:
            return Status.SOLVED
        return Status.UNSOLVABLE

    def _guess_numbers(self, randomize=False):
        '''Return a cell and a possible number to guess. It will return cells with minimal amount of candidates to maximize probability
        of correct guessing.
        It would be better to  find repetead cells and guess them first.
        '''
        if randomize is False:
            candidates = [len(x) if x else 1000 for x in self._candidates.v.flatten()]
            min_candidates = np.min(candidates)
            if min_candidates == 1000:
                return False
            i = candidates.index(min_candidates)
            bi = i // len(self._symbols)
            bj = i % len(self._symbols)
            return bi, bj, self._candidates.v[bi, bj]
        random_empty_cell = random.choice(list(self.missing_cells))
        random_candidates = [x for x in self._candidates.v[random_empty_cell]]
        random.shuffle(random_candidates)
        return *random_empty_cell, random_candidates

    def is_broken(self):
        errors = self._candidates.is_broken()
        for i, row in enumerate(self._clues.v):
            for j, cell in enumerate(row):
                if cell == '-':
                    if len(self._candidates.v[i, j]) == 0:
                        errors.append(f"Cell [{i * 9 + j}] is empty")
        for i, row in enumerate(self._candidates.v):
            sols_row = self._clues.get_row(i)
            fl_opt_row = row.flatten()
            fl_sol_row = sols_row.flatten()
            for el in self._symbols:
                el_in_opt_row = [el in x for x in fl_opt_row]
                el_in_sol_row = [el in x for x in fl_sol_row]
                if np.count_nonzero(el_in_sol_row) + np.count_nonzero(el_in_opt_row) == 0:
                    errors.append(f"Row [{i}] doesn't contain '{el}' in solutions nor candidates")

        for i in range(len(self._symbols)):
            sols_col = self._clues.get_column(i)
            candidates_col = self._candidates.get_column(i)
            fl_opt_col = candidates_col.flatten()
            fl_sol_col = sols_col.flatten()
            for el in self._symbols:
                el_in_opt_col = [el in x for x in fl_opt_col]
                el_in_sol_col = [el in x for x in fl_sol_col]
                if np.count_nonzero(el_in_sol_col) + np.count_nonzero(el_in_opt_col) == 0:
                    errors.append(f"Column [{i}] doesn't contain '{el}' in solutions nor candidates")

        return errors

    def set_solution(self, value, row, column):
        if self._clues.v[row, column] == '-':
            self._clues.v[row, column] = value
            self._candidates.found_value_clean(row, column, value)
            return True
        return False

    def find_block_single_candidates(self, found=0) -> int:
        for bi in range(3):
            for bj in range(3):
                block = self._candidates.get_block(bi, bj)
                for s in self._symbols:
                    arr = [s in x for x in block.v.flatten()]
                    if np.count_nonzero(np.array(arr)) == 1:
                        ix = arr.index(True)
                        r = bi * 3 + ix // 3
                        c = bj * 3 + ix % 3
                        if self.set_solution(s, r, c):
                            self.add_step(step_type=StepType.SingleCandidate,
                                          message=f"Found only one candidate for value {s} on block {3 * bi + bj}: "
                                                  f"[{r}, {c}]")
                            return 1
        return found

    def find_single_candidates(self, found=0) -> int:
        for r in range(9):
            for c in range(9):
                if self._clues.v[r, c] == '-' and len(self._candidates.v[r, c]) == 1:
                    if self.set_solution(self._candidates.v[r, c], r, c):
                        self.add_step(step_type=StepType.SingleCandidate,
                                      message=f"Found single candidate on row {r} column {c}, with value "
                                              f"{self._candidates.v[r, c]}")
                        return 1
        return found

    def solved(self) -> bool:
        if '-' in self._clues.v.flatten():
            return False
        s = set(self._symbols)
        for r in range(9):
            if set(self._clues.get_row(r)) != s:
                return False
        for c in range(9):
            if set(self._clues.get_column(c)) != s:
                return False
        return True

    def find_row_single_candidates(self, found=0) -> int:
        for s in self._symbols:
            for rix, line in enumerate(self._candidates.v):
                arr = [s in x for x in line.flatten()]
                if np.count_nonzero(np.array(arr)) == 1:
                    cix = arr.index(True)
                    if self.set_solution(s, rix, cix):
                        self.add_step(step_type=StepType.SingleCandidate,
                                      message=f"Found single candidate for value {s} on row {rix} [column {cix}]")
                        # return self.find_row_single_candidates(found + 1)
                        return 1
        return found

    def find_column_single_candidates(self, found=0) -> int:
        for c in range(9):
            line = self._candidates.get_column(c)
            for s in self._symbols:
                arr = [s in x for x in line]
                if np.count_nonzero(np.array(arr)) == 1:
                    rix = arr.index(True)
                    if self.set_solution(s, rix, c):
                        self.add_step(step_type=StepType.SingleCandidate,
                                      message=f"Found single candidate for value {s} on column {c} [row {rix}]")
                        # return self.find_column_single_candidates(found + 1)
                        return 1
        return found

    def scan_blocks(self):
        for bi in range(3):
            for bj in range(3):
                block = self._clues.get_block(bi, bj)
                for r in range(bi * 3, bi * 3 + 3):
                    for c in range(bj * 3, bj * 3 + 3):
                        if self._clues.v[r, c] == '-':
                            for s in self._symbols:
                                if s in block.v or s in self._clues.get_row(r) or s in self._clues.get_column(c):
                                    continue
                                if s not in self._candidates.v[r, c]:
                                    self._candidates.v[r, c] = self._candidates.v[r, c] + s
        self.add_step(step_type=StepType.ScanCandidates,
                      message=f"Scanned blocks setting candidates",
                      resolution=10)
        self._scanned_candidates = True


if __name__ == '__main__':
    # init_str = '53--7----6--195----98----6-8---6---34--8-3--17---2---6-6----28----419--5----8--79'
    # Beginner
    # init_str = '8593674-113-5-869--62149385-8593176-----74-582-185-934---49-813-13--254-54-6-3279'
    # Easy
    # init_str = '56---438---218-56---85-2-9-983--1-------38-16-17-2-8398-4-5-97-756819-4-12--47---'
    # Medium
    # init_str = '9-62---5---1-----9--------61-94-68---32--5-6-76413-59-6---73-4--9-----7-4--6-2915'
    # Hard
    # init_str = '-8--9-65--7-4------51------6----3---1-9--8--2-4---7-9-----1--6---2---8-5---3-21--'
    # Hard
    # init_str = '4-8---9--9---4-7----6----48-8---1-7---5--------18-24-6-3-----5-81-3--------98---7'
    # init_str = '--1----9----915-------784-26-----7--5----2--14--8------4--2-8-7--349------7------'
    # Extreme (easy)
    # init_str = '----8---9-6-9--18---4-3----1--5-4--6---3-754----------5-7----1-84---3-----9---7-2'
    # Extreme
    # init_str = '--5-----8---18---7-----412---9-----2-4-3--5--5-6--7-8-6---9---1-2---5----9-6--7--'
    init_str = '235----7---8----------23-4-864--------7--6-85----72----5--67-18--1------9--1---23'

    # others
    init_str = '-86---1-----9-6---5---1--27-24-3--5----6-4----6--9-24-89--5---3---7-9-----1---97-'
    # init_str = '7-3-----6-1---9----961---3-5----79-4---81-2-----5-------24----8---------3-4----6-'
    # init_str = '---1--9---7-----6---2-4-5-18--65--4---7--8-1--6-4------59-1---23---2-7-5---------'
    # init_str = '--9--7-4--71-2---5-4-----39-----8------46------219-8---6----4---9-2865--5--------'
    # init_str = '346----5-5-----6----73------7-6-84-----9-2---2-3--5---4-5-632-7-----7-48-3---1-69'
    # init_str = '236-8-41--8-1--62-1--2---39-62--7--1-73---56-41---2---794-2-------9--28--2------6'
    # init_str = '----1-78-58---7--41--83---68-2-4196-----62-----1--8-2----1-3-----528---33---9-87-'
    s = Sudoku(init_str)
    print(f"missing cells: {s.num_of_missing_cells}")
    s.solve_guessing(find_all=True)
    # print(len(s.valid_solutions))
    # for x in s.valid_solutions:
    #     print(x.as_str())
    print(s.init_str, s.init_str.count('-'), s.cost, s.solved(), len(s.valid_solutions), '|'.join([x.step_type for x in s.steps]))
    # print(s.solve_tree)
    from pygame_sudoku import SudokuVisualize

    sv = SudokuVisualize(steps=s.steps)
    sv.run()
    sv.quit()
