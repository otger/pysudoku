from pysudoku.solvers.human import HumanSolver

"""
Solver and checker of the sudokus copied from apps, second field is the level assigned by the app where the sudoku comes from
"""


def check(init_str):
    s = HumanSolver(init_str)
    s.solve_guessing(find_all=True)
    step_names = [x.step_type for x in s.steps]
    return s.init_str, s.init_str.count('-'), s.cost, s.solved(), len(s.valid_solutions),len(step_names), step_names.count('gsg'),'|'.join(sorted(list(set(step_names))))


if __name__ == "__main__":
    i = 0
    import os
    FILE = "sudokus_from_app.csv"
    current_folder = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(current_folder, FILE), "r") as downloaded:
        with open(os.path.join(current_folder, f"{FILE.split('.')[0]}_solved.csv"), "w") as fp:
            fp.write("#init values, app_level, empty cells, cost, solved, valid solutions, num of steps, num of guessings, step types\n")
            while True:
                try:
                    fields = downloaded.readline().split(',')
                    if fields[0].startswith('#'):
                        continue
                    print(fields)
                    s = check(fields[1].strip())
                    if s:
                        tmp = ', '.join([str(x) for x in s[1:]])
                        fp.write(f"{s[0]}, {fields[0]}, {tmp}\n")
                        print(tmp)
                except:
                    break
