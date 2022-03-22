#!/usr/bin/env python3

### Requirements ###
# bqpjson v0.5 - pip install bqpjson
# cplex v22.1.0 - see CPLEX installation instructions


### NOTE ###
# these are good articles to reference when using this solver

# @article{1306.1202,
#   Author = {Sanjeeb Dash},
#   Title = {A note on QUBO instances defined on Chimera graphs},
#   Year = {2013},
#   Eprint = {arXiv:1306.1202},
#   url = {https://arxiv.org/abs/1612.05024}
# }

# @Article{Billionnet2007,
#   author="Billionnet, Alain and Elloumi, Sourour",
#   title="Using a Mixed Integer Quadratic Programming Solver for the Unconstrained Quadratic 0-1 Problem",
#   journal="Mathematical Programming",
#   year="2007",
#   volume="109",
#   number="1",
#   pages="55--68",
#   issn="1436-4646",
#   doi="10.1007/s10107-005-0637-9",
#   url="http://dx.doi.org/10.1007/s10107-005-0637-9"
# }


import argparse
import bqpjson
import json
import sys

from docplex.mp.model import Model
from docplex.util.environment import get_environment

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

    data = bqpjson.spin_to_bool(data)

    variable_ids = set(data['variable_ids'])
    variable_product_ids = set([(qt['id_tail'], qt['id_head']) for qt in data['quadratic_terms']])

    m = Model()

    if args.runtime_limit != None:
        m.set_time_limit(args.runtime_limit)

    m.parameters.threads.set(args.thread_limit)

    variable_lookup = {}

    for vid in variable_ids:
        variable_lookup[vid] = m.binary_var(name = 'site_%04d' % vid)

    spin_data = bqpjson.core.swap_variable_domain(data)

    if len(spin_data['linear_terms']) <= 0 or all(lt['coeff'] == 0.0 for lt in spin_data['linear_terms']):
        print('detected spin symmetry, adding symmetry breaking constraint')
        v1 = data['variable_ids'][0]
        m.add_constraint(variable_lookup[v1] == 0.0)

    obj = 0.0

    for lt in data['linear_terms']:
        i = lt['id']
        obj += lt['coeff'] * variable_lookup[i]

    for qt in data['quadratic_terms']:
        i = qt['id_tail']
        j = qt['id_head']
        obj += qt['coeff'] * variable_lookup[i] * variable_lookup[j]

    m.minimize(obj)
    m.print_information()
    solution = m.solve()

    lower_bound = m.solve_details.best_bound
    # lower_bound = m.solve_details.mip_relative_gap * solution.objective_value + solution.objective_value
    scaled_objective = data['scale'] * (solution.objective_value + data['offset'])
    scaled_lower_bound = data['scale'] * (lower_bound + data['offset'])
    best_solution = ', '.join(["-1" if variable_lookup[vid].solution_value <= 0.5 else "1" for vid in data['variable_ids']])

    print()
    if args.show_solution:
        print('BQP_SOLUTION, %d, %d, %f, %f, %s' % (len(variable_ids), len(variable_product_ids), scaled_objective, m.solve_details.time, best_solution))
    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % (len(variable_ids), len(variable_product_ids), scaled_objective, \
        scaled_lower_bound, solution.objective_value, lower_bound, m.solve_details.time, 0, m.solve_details.nb_nodes_processed))


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help = 'the data file to operate on (.json)')
    parser.add_argument('-ss', '--show-solution', help = 'prints the a solution data line', action = 'store_true', default = False)
    parser.add_argument('-rtl', '--runtime-limit', help = 'CPLEX runtime limit (seconds)', type = float)
    parser.add_argument('-tl', '--thread-limit', help = 'CPLEX thread limit', type = int, default = 1)
    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())

