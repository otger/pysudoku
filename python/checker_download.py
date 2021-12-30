from sudoku2 import Sudoku

"""
Solver and checker of the sudokus found at: https://www.kaggle.com/bryanpark/sudoku/version/3
"""


def check(init_str):
    s = Sudoku(init_str)
    s.solve_guessing(find_all=True)
    s = Sudoku(init_str)
    s.solve_guessing()
    return s.init_str, s.init_str.count('-'), s.cost, s.solved(), len(s.valid_solutions), '|'.join([x.step_type for x in s.steps])


if __name__ == "__main__":
    i = 0
    with open("downloaded_sudokus.csv", "r") as downloaded:
        with open("downloaded_sudokus_proc2.csv", "a") as fp:
            while True:
                try:
                    line = downloaded.readline().split(',')[0].replace('0', '-')
                    s = check(line)
                    if s:
                        tmp = ', '.join([str(x) for x in s])
                        fp.write(f"{tmp}\n")
                        print(tmp)
                    else:
                        i += 1
                        print(f"Multiple solutions: {i}")
                except:
                    break
