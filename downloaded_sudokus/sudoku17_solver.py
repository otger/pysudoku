from pysudoku.solvers.human import HumanSolver

"""
Solver and checker of the sudokus found at: https://web.archive.org/web/20130922164440if_/http://school.maths.uwa.edu.au/~gordon/sudoku17
This are a collection of 17 clues sudokus elaborated by Gordon Royle and The University of Western Australia.
"""


def check(init_str):
    s = HumanSolver(init_str)
    s.solve_guessing(find_all=True)
    return s.init_str, s.init_str.count('-'), s.cost, s.solved(), len(s.valid_solutions),'|'.join([x.step_type for x in s.steps])


if __name__ == "__main__":
    import os
    current_folder = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(current_folder, "sudoku17.txt"), "r") as downloaded:
        with open(os.path.join(current_folder, "sudoku17_solved.csv"), "w") as fp:
            fp.write("#init values, empty cells, cost, solved, valid solutions, steps to solve\n")
            while True:
                try:
                    line = downloaded.readline().split(',')[0].replace('0', '-').replace('\n', '')
                    s = check(line)
                    if s:
                        tmp = ', '.join([str(x) for x in s])
                        fp.write(f"{tmp}\n")
                        print(tmp)
                except:
                    break
