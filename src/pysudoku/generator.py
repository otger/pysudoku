from pysudoku.solvers.human import HumanSolver
import random


def generate(max_trials=1, reset_trials_on_success=False):
    s = HumanSolver(init_values='-' * 81)
    s.solve_guessing(randomize=True, find_all=False)
    current = HumanSolver(init_values=s.as_str()[0])
    trials = 0
    while True:
        prev = current
        init_list = list(current.init_str)
        non_empty_indexes = [i for i, x in enumerate(init_list) if x != '-']
        selected_ix = random.choice(non_empty_indexes)

        init_list[selected_ix] = '-'
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


if __name__ == "__main__":
    MAX_TRIALS = 10
    RESET_TRIALS_ON_SUCCESS = True
    import os
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output'))
    output_file = os.path.join(output_folder, "sudokus5.csv")
    i = 0
    with open(output_file, "a") as fp:
        while i < 10000:
            s = generate(max_trials=MAX_TRIALS)
            s = HumanSolver(s.init_str)
            s.solve_guessing(find_all=True)
            t = f"{s.init_str}, {s.init_str.count('-')}, {s.cost}, {s.solved()}, {len(s.valid_solutions)}, " \
                f"{MAX_TRIALS}, {RESET_TRIALS_ON_SUCCESS}, {'|'.join([x.step_type for x in s.steps])}"
            fp.write(f"{t}\n")
            print(t)
            fp.flush()
            i += 1
