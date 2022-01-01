#!/usr/bin/env python

import numpy as np

from pysudoku.common import SudokuMatrix, CandidatesMatrix


DEBUG = False



class Sudoku:

    def __init__(self, init_values: str, candidates_str: str = None) -> None:
        if len(init_values) != 81:
            raise ValueError("Invalid Sudoku input")
        self.init_str = init_values
        self._clues = SudokuMatrix(init_values)

        self._symbols = []
        self.fill_symbols()

        if candidates_str is None:
            self._candidates = CandidatesMatrix(symbols=self._symbols, list_values=None)
            self._scanned_candidates = False
        else:
            self._candidates = CandidatesMatrix.from_str(string_repr=candidates_str)
            self._scanned_candidates = True

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

