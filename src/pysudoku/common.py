import numpy as np
import random

DEBUG = False


class Status:
    UNKNOWN = 0
    SOLVED = 1
    UNSOLVED = 2
    UNSOLVABLE = 3


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


class Matrix:
    def __init__(self, v, block=False):
        self.v = v
        self._block = block

    def get_column(self, index):
        return self.v[:, index]

    def get_row(self, index):
        return self.v[index, :]

    def __str__(self):
        tmp = ''
        if self._block:
            for i, row in enumerate(self.v):
                tmp += '|'.join([el if el else '-' for el in row])
                tmp += '\n'

        else:
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
        return Matrix(self.v[i * 3:i * 3 + 3, j * 3:j * 3 + 3], block=True)


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

