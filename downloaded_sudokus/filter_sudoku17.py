#!/usr/bin/python3
import os

FILE_TO_PROC = 'sudoku17_solved.csv'
OUTPUT_FILENAME = 'sudoku17_solved_filtered'
MAX_GUESSINGS = 2
MIN_COST = 6401

# "#init values, empty cells, cost, solved, valid solutions, steps to solve\n"
current_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_folder, FILE_TO_PROC), 'r') as fp17:
    with open(os.path.join(current_folder, f"{OUTPUT_FILENAME}.csv"), 'w') as fp17csv:
        fp17csv.write('#init values, empty cells, cost, guessings, steps\n')
        while True:
            try:
                l = fp17.readline()
            except:
                break
            if l == '':
                break
            if l.startswith('#'):
                continue
            if l.count(',') != 5:
                continue
            init_values, empty_cells, cost, solved, valid_solutions, steps = l.replace('\n', '').split(',')
            print(init_values)
            if solved == 'False':
                continue
            if int(valid_solutions) != 1:
                continue
            step_names = steps.split('|')
            guessings = step_names.count('gsg')
            if guessings > MAX_GUESSINGS:
                continue
            if int(cost) < MIN_COST:
                continue
            fp17csv.write(f"{init_values}, {empty_cells}, {cost}, {guessings}, {'|'.join(sorted(list(set(step_names))))}\n")
