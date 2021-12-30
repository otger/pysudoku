from sudoku2 import Sudoku
import random


def generate():
    s = Sudoku(init_values='-' * 81)
    s.solve_guessing(randomize=True)
    current = Sudoku(init_values=s.as_str()[0])
    while True:
        prev = current
        init_list = list(current.init_str)
        non_empty_indexes = [i for i, x in enumerate(init_list) if x != '-']
        selected_ix = random.choice(non_empty_indexes)

        init_list[selected_ix] = '-'
        # We remove the mirror position. If selected_ix is row r and col c (r,c), we also remove (8-r, 8-c) (0 based)
        init_list[80-selected_ix] = '-'
        init_str = ''.join(init_list)
        current = Sudoku(init_values=init_str)
        current.solve_guessing(find_all=True)
        if len(current.valid_solutions) > 1:
            return prev


if __name__ == "__main__":
    i = 0
    with open("sudokus4.csv", "a") as fp:
        while i < 10000:
            s = generate()
            s = Sudoku(s.init_str)
            s.solve_guessing(find_all=True)
            t = f"{s.init_str}, {s.init_str.count('-')}, {s.cost}, {s.solved()}, {len(s.valid_solutions)}, {'|'.join([x.step_type for x in s.steps])}"
            fp.write(f"{t}\n")
            print(t)
            fp.flush()
            i += 1
