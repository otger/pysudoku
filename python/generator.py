from sudoku import Sudoku
import random


def generate():
    s = Sudoku(init_values='-' * 81)
    s.solve_guessing(randomize=True)
    current = Sudoku(init_values=s.as_str()[0])
    while True:
        prev = current
        init_list = list(current.init_str)
        non_empty_indexes = [i for i, x in enumerate(init_list) if x != '-']
        init_list[random.choice(non_empty_indexes)] = '-'
        init_str = ''.join(init_list)
        current = Sudoku(init_values=init_str)
        current.solve_guessing(find_all=True)
        if len(current.valid_solutions) > 1:
            return prev


if __name__ == "__main__":
    i = 0
    with open("sudokus.csv", "a") as fp:
        while i < 10000:
            s = generate()
            s = Sudoku(s.init_str)
            s.solve_guessing()
            fp.write(f"{s.init_str}, {s.init_str.count('-')}, {s.costs['guesses']}, {s.solved()}, {'|'.join([str(x) for x in s.costs['iterations']])}\n")
            print(f"{s.init_str}, {s.init_str.count('-')}, {s.costs['guesses']}, {s.solved()}, {'|'.join([str(x) for x in s.costs['iterations']])}")
            fp.flush()
            i += 1
