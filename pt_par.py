#!/usr/bin/env python3

### Requirements ###
# bqpjson v0.5.3 - pip install bqpjson
# parallel-tempering - private repository

import sys, os, argparse, json, random, tempfile
import shutil
import time
import numpy as np

from io import StringIO

import subprocess
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
        energy_lines = output.readlines()[1:-1]
        energies = [float(x.split('\t')[1]) for x in energy_lines]
        minimum_energy_index = np.argmin(energies, axis = 0)
        minimum_energy = energies[minimum_energy_index]
        best_solution = energy_lines[minimum_energy_index].split(' ')[-1]

    return minimum_energy, best_solution


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

    start_time = time.time()
    subprocess.call(["mpiexec", "-np", "8", "parallel-tempering", "64", \
        "2", "0.1", "5.0", "0", "0", "1", str(args.runtime_limit), "8", \
        "-999999.9", "0", input_name], stdout = subprocess.DEVNULL, cwd = tmp_directory)
    elapsed_time = time.time() - start_time

    result_path = os.path.join(tmp_directory, "EnergiesFound_input.dat")

    # Get the best energy, corresponding solution, and metadata.
    energy, solution = parse_result(result_path)
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
        bqp_solution = ', '.join(["-1" if solution[i] == "0" else "1" for i in range(0, nodes)])
        print('BQP_SOLUTION, %d, %d, %f, %f, %s' % \
            (nodes, edges, energy, elapsed_time, bqp_solution))

    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % \
        (nodes, edges, energy, scaled_lower_bound, energy, lower_bound, elapsed_time, 0, 0))

    # Clean up.
    shutil.rmtree(tmp_directory)

def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help='the data file to operate on (.json)')
    parser.add_argument('-ss', '--show-solution', help='prints the a solution data line', action='store_true', default=False)
    parser.add_argument('-rtl', '--runtime-limit', help='runtime limit (iters.)', type=int)
    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())