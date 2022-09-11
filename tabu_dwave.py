#!/usr/bin/env python3

### Requirements ###
# bqpjson v0.5 - pip install bqpjson
# dwave-tabu v0.4 - pip install dwave-tabu


import argparse
import bqpjson
import tabu
import json
import sys
import time


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

    h = {}
    for lt in data['linear_terms']:
        i = lt['id']
        assert(not i in h)
        h[i] = lt['coeff']

    J = {}
    for qt in data['quadratic_terms']:
        i = qt['id_tail']
        j = qt['id_head']
        assert(not (i, j) in J)
        J[(i, j)] = qt['coeff']

    # Set up the tabu sampler.
    sampler = tabu.TabuSampler()

    # Sample the tabu solver `args.runtime_limit` number of times.
    start_time = time.time()
    sample_set = sampler.sample_ising(h, J, num_reads = args.runtime_limit, timeout = 1000.0)
    time_elapsed = time.time() - start_time

    # Get the sample with the lowest energy.
    best_sample = sample_set.first
    best_objective = best_sample.energy
    best_solution = ', '.join([str(best_sample.sample[vid]) for vid in data['variable_ids']])
    best_runtime = time_elapsed

    # Get lower bound estimate.
    lt_lb = -sum(abs(lt['coeff']) for lt in data['linear_terms'])
    qt_lb = -sum(abs(qt['coeff']) for qt in data['quadratic_terms']) 
    lower_bound = lt_lb + qt_lb

    # Get scaled objective information.
    scaled_objective = data['scale'] * (best_objective + data['offset'])
    scaled_lower_bound = data['scale'] * (lower_bound + data['offset'])

    # Get other metadata that will be printed.
    nodes = len(data['variable_ids'])
    edges = len(data['quadratic_terms'])

    print()
    if args.show_solution:
        print('BQP_SOLUTION, %d, %d, %f, %f, %s' % (nodes, edges, scaled_objective, best_runtime, best_solution))
    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % (nodes, edges, scaled_objective, scaled_lower_bound, best_objective, lower_bound, best_runtime, 0, args.runtime_limit))


def build_cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--input-file', help = 'the data file to operate on (.json)')
    parser.add_argument('-ss', '--show-solution', help = 'prints the a solution data line', action='store_true', default = False)
    parser.add_argument('-rtl', '--runtime-limit', help = 'tabu solver runtime limit (number of reads)', type = int, default = 1)

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())

