from pysudoku.solvers.human import HumanSolver

"""
Solver and checker of the sudokus found 
downloaded_sudokus.csv: https://www.kaggle.com/bryanpark/sudoku/version/3
    - this are very simple sudokus. A lot of them, but all of them can be solved with very simple strategies
sudoku17.txt: https://web.archive.org/web/20130922164440if_/http://school.maths.uwa.edu.au/~gordon/sudoku17
    This are a collection of 17 clues sudokus elaborated by Gordon Royle and The University of Western Australia.
"""


def check(init_str):
    s = HumanSolver(init_str)
    s.solve_guessing(find_all=True)
    step_names = [x.step_type for x in s.steps]
    return s.init_str, s.init_str.count('-'), s.cost, s.solved(), len(s.valid_solutions), step_names.count('gsg'),'|'.join(step_names)


if __name__ == "__main__":
    i = 0
    import os
    FILE = "downloaded_sudokus.csv"
    #FILE = "sudoku17.txt"
    current_folder = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(current_folder, FILE), "r") as downloaded:
        with open(os.path.join(current_folder, f"{FILE.split('.')[0]}_solved.csv"), "a") as fp:
            fp.write("#init values, empty cells, cost, solved, valid solutions, num of guessings, steps to solve\n")
            while True:
                try:
                    line = downloaded.readline().split(',')[0].replace('0', '-')
                    s = check(line)
                    if s:
                        tmp = ', '.join([str(x) for x in s])
                        fp.write(f"{tmp}\n")
                        print(tmp)
                except:
                    break
