from pysudoku.solvers.human import HumanSolver
import random

from pysudoku.solvers.human_helpers import StepType


def generate(max_trials=1, mirrored=False, reset_trials_on_success=False, max_guesses=None, max_empty_cells = None):
    s = HumanSolver(init_values='-' * 81)
    s.solve_guessing(randomize=True, find_all=False)
    current = HumanSolver(init_values=s.as_str()[0])
    trials = 0
    while True:
        prev = current
        init_list = list(current.init_str)
        non_empty_indexes = [i for i, x in enumerate(init_list) if x != '-']
        if max_empty_cells:
            if 81-len(non_empty_indexes) == max_empty_cells:
                return prev
        selected_ix = random.choice(non_empty_indexes)

        init_list[selected_ix] = '-'
        if mirrored:
            # We remove the mirror position. If selected_ix is row r and col c (r,c), we also remove (8-r, 8-c) (0 based)
            init_list[80-selected_ix] = '-'

        init_str = ''.join(init_list)
        current = HumanSolver(init_values=init_str)
        current.solve_guessing(find_all=True)
        if len(current.valid_solutions) > 1:
            trials += 1
            if trials >= max_trials:
                return prev
            current = prev
        elif reset_trials_on_success:
            trials = 0
        if max_guesses:
            if current.solve_tree.count(StepType.Guessing) > max_guesses:
                return prev

if __name__ == "__main__":
    MAX_TRIALS = 64
    RESET_TRIALS_ON_SUCCESS = True
    MIRRORED = False
    # MAX_EMPTY_CELLS = 10
    # class LevelsCosts:
    #     beginner = 1500
    #     easy = 3000
    #     medium = 4000
    #     hard = 10000
    #     extreme = 2000000

    MAX_EMPTY_CELLS = [15,18,21,24,26,28,29, 31,33,35,37,39, 52,53,54, 55,56,57,58,59,60,61,62,63,64,65]
    GEN_PUZZLES_PER_LOOP_EMPTY_CELLS_VALUE = 75
    OUTPUT_FILENAME = "20220207.csv"
    import os
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output'))
    output_file = os.path.join(output_folder, OUTPUT_FILENAME)
    write_header = True
    if os.path.exists(output_file):
        write_header = False
    with open(output_file, "a") as fp:
        if write_header:
            # fp.write("# Max empty cells, Max Trials, Reset Trials, Mirrored, Sudoku Str, Empty cells, Cost, Solved, Valid Solutions, Number of guesses, Steps\n")
            fp.write("#init values, max empty cells, max trials, reset_trials, Mirrored, empty cells, cost, solved, valid solutions, num of steps, num of guessings, step types\n")
        for max_empty_cells in MAX_EMPTY_CELLS:
            i = 0
            while i < GEN_PUZZLES_PER_LOOP_EMPTY_CELLS_VALUE:
                s = generate(max_trials=MAX_TRIALS, reset_trials_on_success=RESET_TRIALS_ON_SUCCESS, max_guesses=2, max_empty_cells=max_empty_cells)
                s = HumanSolver(s.init_str)
                s.solve_guessing(find_all=True)
                step_names = [x.step_type for x in s.steps]
                t = ', '.join([str(x) for x in [s.init_str, max_empty_cells, MAX_TRIALS, RESET_TRIALS_ON_SUCCESS, MIRRORED,
                    s.init_str.count('-'), s.cost, s.solved(), len(s.valid_solutions),len(step_names), 
                    step_names.count('gsg'),'|'.join(sorted(list(set(step_names))))]])
                # t = f"{max_empty_cells}, {MAX_TRIALS}, {RESET_TRIALS_ON_SUCCESS}, {MIRRORED}, {s.init_str}, {s.init_str.count('-')}, {s.cost}, " \
                #     f"{s.solved()}, {len(s.valid_solutions)}, " \
                #     f"{step_names.count('gsg')}, {'|'.join(step_names)}"
                fp.write(f"{t}\n")
                print(t)
                fp.flush()
                i += 1
