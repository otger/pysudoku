from sudoku import Sudoku

"""
Solver and checker of the sudokus found at: https://www.kaggle.com/bryanpark/sudoku/version/3
"""


def check(init_str):
    s = Sudoku(init_str)
    s.solve_guessing(find_all=True)
    if len(s.valid_solutions) > 1:
        return False
    s = Sudoku(init_str)
    s.solve_guessing()
    return s.init_str, s.init_str.count('-'), s.costs['guesses'], s.solved(), s.costs['iterations']


if __name__ == "__main__":
    i = 0
    with open("downloaded_sudokus.csv", "r") as downloaded:
        with open("downloaded_sudokus_proc.csv", "a") as fp:
            while True:
                try:
                    line = downloaded.readline().split(',')[0].replace('0', '-')
                    s = check(line)
                    if s:
                        fp.write(f"{s[0]}, {s[1]}, {s[2]}, {s[3]}, {'|'.join([str(x) for x in s[4]])}\n")
                    else:
                        i += 1
                        print(f"Multiple solutions: {i}")
                except:
                    break
