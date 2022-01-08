import numpy as np
import random

from numpy.core.arrayprint import format_float_positional
from pysudoku.sudoku import Sudoku
from pysudoku.common import Resolutions, Status
from pysudoku.solvers.human_helpers import Step, StepResolution, StepType


class HumanSolver(Sudoku):

    def __init__(self, init_values: str, candidates_str: str = None,
                 init_step: Step = None):
        super(HumanSolver, self).__init__(init_values=init_values, candidates_str=candidates_str)
        self.solve_tree = []
        self.errors = []
        self.iterations = 0
        self.costs = {'iterations': [],
                      'guesses': 0}
        if init_step is None:
            self.steps = [Step(self._clues.as_str(), self._candidates.as_str(), step_type=StepType.InitState,
                               msg='Initial State', prev_step=None)]
        else:
            self.steps = [init_step]

    @property
    def cost(self):
        cost = 0
        for ste in self.steps:
            cost += ste.cost
        return cost

    def add_step(self, step_type: str, message):
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
            self.add_step(step_type=StepType.CleanCandidates, message='Cleaned candidates')
            return True
        return False

    def set_solution(self, value, row, column):
        if self._clues.v[row, column] == '-':
            self._clues.v[row, column] = value
            self._candidates.found_value_clean(row, column, value)
            return True
        return False

    def set_candidates(self):
        for bi in range(3):
            for bj in range(3):
                block = self._clues.get_block(bi, bj)
                for r in range(bi * 3, bi * 3 + 3):
                    for c in range(bj * 3, bj * 3 + 3):
                        self._candidates.v[r, c] = ''
                        if self._clues.v[r, c] == '-':
                            for s in self._symbols:
                                if s in block.v or s in self._clues.get_row(r) or s in self._clues.get_column(c):
                                    continue
                                if s not in self._candidates.v[r, c]:
                                    self._candidates.v[r, c] = self._candidates.v[r, c] + s
        self.add_step(step_type=StepType.ScanCandidates,
                      message=f"Scanned blocks setting candidates")
        self._scanned_candidates = True




    def find_single_position(self) -> int:
        # It is the same case where you only get a single candidate but with only a position empty in a row, col or block
        # Just to differentitate between this case and the general case of single candidate, in case we want to give a different cost
        for _ix in range(9):
            ## check row
            _row = self.clues.get_row(_ix)
            _row_miss_sols = {'1','2','3','4','5','6','7','8','9'} - set(_row)
            if len(_row_miss_sols) == 1:
                _col_ix = np.where(_row == "-")[0][0]
                _value = list(_row_miss_sols)[0]
                if self.set_solution(_value, _ix, _col_ix):
                    self.add_step(step_type=StepType.SinglePosition,
                                    message=f"Found single position in row {_ix}, column {_col_ix}, value {_value}\n")
                    return 1
            ## check column
            _col = self.clues.get_column(_ix)
            _col_miss_sols = {'1','2','3','4','5','6','7','8','9'} - set(_col)
            if len(_col_miss_sols) == 1:
                _row_ix = np.where(_col == "-")[0][0]
                _value = list(_col_miss_sols)[0]
                if self.set_solution(_value, _row_ix, _ix):
                    self.add_step(step_type=StepType.SinglePosition,
                                    message=f"Found single position in column:  row {_row_ix}, column {_ix}, value {_value}\n")
                    return 1
            ## check block
            _block_num = (_ix//3, _ix%3)
            _block = self.clues.get_block(*_block_num).v.flatten()
            _block_miss_sols = {'1','2','3','4','5','6','7','8','9'} - set(_block)
            if len(_block_miss_sols) == 1:
                _block_ix = np.where(_block == "-")[0][0]
                _value = list(_block_miss_sols)[0]
                if self.set_solution(_value, 3*_block_num[0]+_block_ix // 3, 3*_block_num[1] + _block_ix % 3):
                    self.add_step(step_type=StepType.SinglePosition,
                                    message=f"Found single position in block: row {3*_block_num[0]+_block_ix // 3}, "
                                            f"column {3*_block_num[1] + _block_ix % 3}, value {_value}\n")
                    return 1
            # check block candidates for elements only in a cell
            # should check if this one is the only necessary step and we can get rid of the others previous 3
            _block_cand = self._candidates.get_block(*_block_num).v.flatten()
            for _v in _block_miss_sols:
                _v_in_cand = [_v in _block_cand[x] for x in range(9)]
                positions = np.where(_v_in_cand)[0]
                if len(positions) == 1:
                    if self.set_solution(_v, 3*_block_num[0] + positions[0] // 3, 3*_block_num[1] + positions[0] % 3):
                        self.add_step(step_type=StepType.SinglePosition,
                                      message=f"Found single position in block: row {3*_block_num[0] + positions[0] // 3}, "
                                              f"column {3*_block_num[1] + positions[0] % 3}, value {_v}\n")
                        return 1

        return 0

    def find_single_candidates(self) -> int:
        for r in range(9):
            for c in range(9):
                if self._clues.v[r, c] == '-' and len(self._candidates.v[r, c]) == 1:
                    val = self._candidates.v[r, c]
                    if self.set_solution(val, r, c):
                        self.add_step(step_type=StepType.SingleCandidate,
                                      message=f"Found single candidate on row {r} column {c}, with value "
                                              f"{val}")
                        return 1
        return 0

    def process_candidate_lines(self,) -> int:
        if self.clean_candidates_in_single_row_of_block():
            return 1
        if self.clean_candidates_in_single_col_of_block():
            return 1
        return 0

    def clean_candidates_in_single_row_of_block(self) -> int:
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
                        if changes:
                            self.add_step(step_type=StepType.CandidateLines,
                                            message=f"Found candidate value {s} only in a row of block {bi * 3 + bj} "
                                                    f"and cleaned outside it.\n")
                            return 1
        return 0

    def clean_candidates_in_single_col_of_block(self) -> int:
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
                                    self._candidates.v[i, bj * 3 + bcix] = self._candidates.v[i, bj * 3 + bcix].replace(
                                        s, '')
                        if changes:
                            self.add_step(
                                step_type=StepType.CandidateLines,
                                message=f"Found candidate value {s} only in a column of block {bi * 3 + bj} and "
                                        f"cleaned outside it.\n")
                            return 1
        return 0
               
    def process_double_pair(self):
        if self._process_double_pairs_columns():
            return 1
        return 0
    
    def _process_double_pairs_columns(self) -> int:
        for bj in range(3):
            for bi in range(2):
                _cb = self._candidates.get_block(bi, bj)
                # elements of each column of the block as set (no repetitions)
                col_els = [set(x[0]+x[1]+x[2]) for x in [_cb.get_column(i) for i in range(3)]]
                # for each pair of columns
                for c in [(0,1,2), (0,2,1), (1,2,0)]: # (first col to check, second col to check, col to clear if need to)
                    # elements only present on selected columns and not in the other one
                    # elements in selected columns - elements common to 3 columns -> elements only in selected columns
                    _comm_els = set.difference(set.intersection(col_els[c[0]], col_els[c[1]]), set.intersection(*col_els))
                    if _comm_els:
                        # Check on the same columns on the other blocks (in the same column of blocks)
                        for _bi in range(bi+1, 3):
                            __cb = self._candidates.get_block(_bi, bj)
                            _rows_els = [set(x[0]+x[1]+x[2]) for x in [__cb.get_column(i) for i in range(3)]]
                            __com_els = set.difference(set.intersection(_rows_els[c[0]], _rows_els[c[1]]), set.intersection(*_rows_els))
                            coincidences = set.intersection(_comm_els, __com_els)
                            if coincidences:
                                # There are coincidences on another block on same column of the common elements
                                found = False
                                __bi = list(set.difference({0,1,2}, {bi, _bi}))[0]
                                for _c in c[:1]:
                                    for i in range(3):
                                        _v = list(coincidences)[0]
                                        if _v in self._candidates.v[__bi*3+i, bj * 3 + _c]:
                                            found = True
                                            self._candidates.v[__bi*3+i, bj * 3 + _c] = self._candidates.v[__bi*3+i, bj * 3 + _c].replace(_v, '')
                                if found:
                                    self.add_step(
                                        step_type=StepType.DoublePairs,
                                        message=f"Found double pairs in blocks ({(bi, bj), (_bi, bj)}) at columns {c[0], c[1]}. "
                                                f"Cleaned value {_v} in column {c[2]} in block ({(__bi, bj)})\n")
                                    return 1
        return 0

                





    def _clean_naked_pairs(self, bi, bj, ij_0, ij_1, values) -> int:
        # (i,j) -> row and column inside the block (0 <= bi,bj <3)
        # (bi, bj) -> row, column of the block (0 <= bi,bj <3)
        changes = 0
        values_in_row_or_col = False
        # check if same row and clean the row outside the block
        if ij_0[0] == ij_1[0]:
            row = bi * 3 + ij_0[0]
            cols = (bj * 3 + ij_0[1], bj * 3 + ij_1[1])
            for j in range(9):
                if j not in cols:
                    for v in values:
                        if v in self._candidates.v[row, j]:
                            changes += 1
                            self._candidates.v[row, j] = self._candidates.v[row, j].replace(v, '')
        # Check if same column and clean column if it is the case
        elif ij_0[1] == ij_1[1]:
            rows = (bi * 3 + ij_0[0], bi * 3 + ij_1[0])
            col = bj * 3 + ij_0[1]
            for i in range(9):
                if i not in rows:
                    for v in values:
                        if v in self._candidates.v[i, col]:
                            self._candidates.v[i, col] = self._candidates.v[i, col].replace(v, '')
                            changes += 1
        # clean block
        for i in range(3):
            for j in range(3):
                if (i, j) not in [ij_0, ij_1]:
                    for v in values:
                        if v in self._candidates.v[bi * 3 + i, bj * 3 + j]:
                            self._candidates.v[bi * 3 + i, bj * 3 + j] = \
                                self._candidates.v[bi * 3 + i, bj * 3 + j].replace(v, '')
                            changes += 1
        return changes

    def process_naked_pairs(self) -> int:
        """
        Look for cells in a column or row of a block with same two candidates. If two cells in a row (or column) have the
        same unique two candidates, it means that those numbers can't be anywhere else on the block or on the row (column)
        """
        for bi in range(3):
            for bj in range(3):
                bloc_cand = self._candidates.get_block(bi, bj)
                bin = {}
                for i, row in enumerate(bloc_cand.v):
                    for j, el in enumerate(row):
                        if len(el) == 2:
                            if el in bin:
                                # We've found at least 2 cells whith the same 2 candidates and only those candidates. We can clean the block
                                # and if the cells are in a row/column we can clean candidates from the others blocks in the same row/column
                                if self._clean_naked_pairs(bi, bj, bin[el], (i, j), el):
                                    self.add_step(step_type=StepType.NakedPair,
                                                    message=f"Found same 2 values [{el}] in a row or a col on a "
                                                            f"block [{bi, bj}]. Cleaned it")
                                    return 1
                            else:
                                bin[el] = (i, j)
        return 0

    def solve_clean(self) -> tuple:
        if self._scanned_candidates is False:
            self.set_candidates()
        while True:
            if self.solved():
                self.valid_solutions.append(self.as_str())
                return Resolutions.solved, None
            isb = self.is_broken()
            if isb:
                self.errors.extend(isb)
                return Resolutions.unsolvable, isb
            
            if self.find_single_position():
                self.solve_tree.append(StepType.SinglePosition)
                continue
            elif self.clean_candidates():
                self.solve_tree.append(StepType.CleanCandidates)
                continue
            elif self.find_single_candidates():
                self.solve_tree.append(StepType.SingleCandidate)
                continue
            elif self.process_double_pair():
                self.solve_tree.append(StepType.DoublePairs)
                continue
            elif self.clean_candidates_in_single_col_of_block():
                self.solve_tree.append(5)
                continue
            elif self.clean_candidates_in_single_row_of_block():
                self.solve_tree.append(6)
                continue
            elif self.process_naked_pairs():
                self.solve_tree.append(StepType.NakedPair)
                continue
            else:
                return Resolutions.unsolved, None

    def solve_guessing(self, randomize=False, find_all=False) -> int:
        # Solve using logic
        self.solve_clean()
        if self.solved():
            return Status.SOLVED
        if self.errors:
            return Status.UNSOLVABLE
        # Solve using guessing
        i, j, els = self._guess_numbers(randomize)
        self.costs['guesses'] += 1
        for e in els:
            # print(f"Guessing {i, j, e}, empty seats: {self.num_of_missing_cells}")
            s = HumanSolver(self._clues.as_str(), candidates_str=self._candidates.as_str(),
                            init_step=self.steps[-1])
            s.set_solution(e, i, j)
            s.add_step(step_type=StepType.Guessing,
                       message=f"Guessing value {e} on position [{i}, {j}]")
            status = s.solve_guessing(find_all=find_all)
            if status == Status.SOLVED:
                if find_all is False:
                    self.steps.extend(s.steps)
                    self.solve_tree.extend(s.solve_tree)
                    self._clues = s._clues
                    self._candidates = s._candidates
                    self.costs['guesses'] += s.costs['guesses']
                    self.costs['iterations'].extend(s.costs['iterations'])
                    return status
                else:
                    if s.solved():
                        self.steps.extend(s.steps)
                        self.solve_tree.extend(s.solve_tree)
                        self._clues = s._clues
                        self._candidates = s._candidates
                        self.costs['guesses'] += s.costs['guesses']
                        self.costs['iterations'].extend(s.costs['iterations'])
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
    # init_str = '-86---1-----9-6---5---1--27-24-3--5----6-4----6--9-24-89--5---3---7-9-----1---97-'
    # init_str = '7-3-----6-1---9----961---3-5----79-4---81-2-----5-------24----8---------3-4----6-'
    # init_str = '---1--9---7-----6---2-4-5-18--65--4---7--8-1--6-4------59-1---23---2-7-5---------'
    # init_str = '--9--7-4--71-2---5-4-----39-----8------46------219-8---6----4---9-2865--5--------'
    # init_str = '346----5-5-----6----73------7-6-84-----9-2---2-3--5---4-5-632-7-----7-48-3---1-69'
    # init_str = '236-8-41--8-1--62-1--2---39-62--7--1-73---56-41---2---794-2-------9--28--2------6'
    # init_str = '----1-78-58---7--41--83---68-2-4196-----62-----1--8-2----1-3-----528---33---9-87-'
    init_str = '----1-78-58---7--41--83---68-2-4196-----62-----1--8-2----1-3-----528---33---9-87-'
    # Evil from 3485 paper
    # init_str = '---------5-8---------2----1---5---9------1--69-6---4---31--6--7---72-8---82-9---3'
    # sudokuoftheday.com - Diabolical
    init_str = '--2-95----4-6-8-----9-3--4---4----93-93-6-58-85----6---2--5-7-----8-9-3----71-4--'
    # sudokuoftheday.com - Fiendish
    init_str = '-----64---72----3-5--78--1-1--8-3----6-----9----1-9--6-9--48--1-4----25---13-----'

    # mygen
    init_str = '2--7----1-8-1---7-14-2---9---2--8---8---4---9---3--7---2---5-16-6---3-2-3----1--8'
    s = HumanSolver(init_str)
    print(f"missing cells: {s.num_of_missing_cells}")
    s.solve_guessing(find_all=True)
    # print(len(s.valid_solutions))
    # for x in s.valid_solutions:
    #     print(x.as_str())
    print(s.init_str, s.init_str.count('-'), s.cost, s.solved(), len(s.valid_solutions),
          '|'.join([x.step_type for x in s.steps]))
    print(s.solve_tree)
    if s.errors:
        print(f"Some errors found:\n {'n'.join([f'  - {x}' for x in s.errors])}")
    print(f"cost: {s.cost}")
    from pysudoku.ui.pygame_sudoku import SudokuVisualize

    sv = SudokuVisualize(steps=s.steps)
    sv.run()
    sv.quit()
