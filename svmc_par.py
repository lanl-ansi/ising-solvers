#!/usr/bin/env python3

### Requirements ###
# bqpjson v0.5.3 - pip install bqpjson
# spin-vector-monte-carlo - private repository

import sys, os, argparse, json, random, tempfile
import shutil
import time
import numpy as np
import subprocess

from io import StringIO
from subprocess import Popen
from subprocess import PIPE
from collections import namedtuple

import bqpjson


def write_bqpjson_as_txt(directory, data):
    output_path = os.path.join(directory, "input.txt")

    with open(output_path, "w") as output:
        for lt in data['linear_terms']:
            output.write('1\t{:d}\t{:.6f}\n'.format(lt['id'], lt['coeff']))
    
        for qt in data['quadratic_terms']:
            i, j = qt['id_tail'], qt['id_head']
            output.write('2\t{:d}\t{:d}\t{:.6f}\n'.format(i, j, qt['coeff']))

    return output_path


def parse_result(result_path):
    with open(result_path, "r") as output:
        all_lines = output.readlines()
        summary_line = all_lines[0].strip()
        summary_line = " ".join(summary_line.split())

        num_sweeps = int(float(summary_line.split(' ')[0]))
        solve_time = float(summary_line.split(' ')[1])
        minimum_energy = float(summary_line.split(' ')[2])
        best_solution = all_lines[1].replace('0', '')
        best_solution = " ".join(best_solution.split()).strip()
       
    return num_sweeps, solve_time, minimum_energy, best_solution


def main(args):
    if args.input_file == None:
        data = json.load(sys.stdin)
    else:
        with open(args.input_file) as file:
            data = json.load(file)

    bqpjson.validate(data)

    if data['variable_domain'] != 'spin':
        print('only spin domains are supported. Given %s' % data['variable_domain'])
        quit()

    tmp_directory = tempfile.mkdtemp()
    input_path = write_bqpjson_as_txt(tmp_directory, data)
    input_name = os.path.splitext(os.path.basename(input_path))[0]

    # Must be divisible by the number of points in the annealing schedule.
    assert int(args.runtime_limit) % 1000 == 0

    # Get the effective local field setting.
    elf_setting = 1 if not args.effective_local_field else 2

    # Get the directory of the SVMC executable.
    svmc_dir = os.path.dirname(shutil.which('svmc'))
    annealing_schedule_path = os.path.join(svmc_dir, 'annealing_schedule.txt')
    new_annealing_schedule_path = os.path.join(tmp_directory, 'annealing_schedule.txt')
    shutil.copy(annealing_schedule_path, new_annealing_schedule_path)

    # Run the algorithm and measure the elapsed time.
    start_time = time.time()
    subprocess.call(["mpiexec", "-np", "8", "svmc", "12", \
        str(args.runtime_limit), "8", "0", str(elf_setting), input_name], \
        stdout = subprocess.DEVNULL, cwd = tmp_directory)
    elapsed_time = time.time() - start_time

    # Get the path to the result.
    result_path = os.path.join(tmp_directory, "SVMC_LowestEnergyFound_input.dat")

    # Get the best energy, corresponding solution, and other metadata.
    num_sweeps, solve_time, energy, solution = parse_result(result_path)
    nodes = len(data['variable_ids'])
    edges = len(data['quadratic_terms'])

    # Estimate lower bound from the problem data. 
    lt_lb = -sum(abs(lt['coeff']) for lt in data['linear_terms'])
    qt_lb = -sum(abs(qt['coeff']) for qt in data['quadratic_terms'])
    lower_bound = lt_lb + qt_lb
    scaled_lower_bound = data['scale'] * (lower_bound + data['offset'])

    # Print solution data.
    print()

    if args.show_solution:
        bqp_solution = solution.replace(' ', ', ')
        print('BQP_SOLUTION, %d, %d, %f, %f, %s' % \
            (nodes, edges, energy, solve_time, bqp_solution))

    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % \
        (nodes, edges, energy, scaled_lower_bound, energy, lower_bound, solve_time, 0, num_sweeps))

    # Clean up.
    shutil.rmtree(tmp_directory)


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help='the data file to operate on (.json)')
    parser.add_argument('-ss', '--show-solution', help='prints the a solution data line', action='store_true', default=False)
    parser.add_argument('-rtl', '--runtime-limit', help='runtime limit (sweeps)', type=int)
    parser.add_argument('-elf', '--effective-local-field', help='whether or not to use the effective local field in updates', action='store_true', default=False)
    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())
