#!/usr/bin/env python3

### Requirements ###
# bqpjson v0.5 - pip install bqpjson
# gurobi v7.0 - http://www.gurobi.com/
#

### NOTE ###
# these are good articles to reference when using this solver
#
# @article{1306.1202,
#   Author = {Sanjeeb Dash},
#   Title = {A note on QUBO instances defined on Chimera graphs},
#   Year = {2013},
#   Eprint = {arXiv:1306.1202},
#   url = {https://arxiv.org/abs/1612.05024}
# }
#
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
#

import argparse, json, sys

from gurobipy import *

import bqpjson

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
        m.setParam('TimeLimit', args.runtime_limit)

    m.setParam('Threads', args.thread_limit)

    if args.cuts != None:
        m.setParam('Cuts', args.cuts)

    #m.setParam('Presolve', 2)
    #m.setParam('MIPFocus', 1)
    #m.setParam('MIPFocus', 2)

    variable_lookup = {}
    for vid in variable_ids:
        variable_lookup[vid] = m.addVar(lb=0, ub=1, vtype = GRB.BINARY, name='site_%04d' % vid)
    m.update()

    spin_data = bqpjson.core.swap_variable_domain(data)
    if len(spin_data['linear_terms']) <= 0 or all(lt['coeff'] == 0.0 for lt in spin_data['linear_terms']):
        print('detected spin symmetry, adding symmetry breaking constraint')
        v1 = data['variable_ids'][0]
        m.addConstr(variable_lookup[(v1,v1)] == 0)

    obj = 0.0
    for lt in data['linear_terms']:
        i = lt['id']
        obj += lt['coeff']*variable_lookup[i]

    for qt in data['quadratic_terms']:
        i = qt['id_tail']
        j = qt['id_head']
        obj += qt['coeff']*variable_lookup[i]*variable_lookup[j]

    m.setObjective(obj, GRB.MINIMIZE)

    m.update()

    m._cut_count = 0
    m.optimize(cut_counter)

    if args.show_solution:
        print('')
        for k,v in variable_lookup.items():
            print('{:<18}: {}'.format(v.VarName, v.X))

    lower_bound = m.MIPGap*m.ObjVal + m.ObjVal
    scaled_objective = data['scale']*(m.ObjVal+data['offset'])
    scaled_lower_bound = data['scale']*(lower_bound+data['offset'])

    print('')
    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % (len(variable_ids), len(variable_product_ids), scaled_objective, scaled_lower_bound, m.ObjVal, lower_bound, m.Runtime, m._cut_count, m.NodeCount))


def cut_counter(model, where):
    cut_names = {
        'Clique:', 'Cover:', 'Flow cover:', 'Flow path:', 'Gomory:', 
        'GUB cover:', 'Inf proof:', 'Implied bound:', 'Lazy constraints:', 
        'Learned:', 'MIR:', 'Mod-K:', 'Network:', 'Projected Implied bound:', 
        'StrongCG:', 'User:', 'Zero half:'}
    if where == GRB.Callback.MESSAGE:
        # Message callback
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        if any(name in msg for name in cut_names):
            model._cut_count += int(msg.split(':')[1])


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help='the data file to operate on (.json)')

    parser.add_argument('-ss', '--show-solution', help='print the solution', action='store_true', default=False)
    parser.add_argument('-rtl', '--runtime-limit', help='gurobi runtime limit (sec.)', type=float)
    parser.add_argument('-tl', '--thread-limit', help='gurobi thread limit', type=int, default=1)
    parser.add_argument('-cuts', help='gurobi cuts parameter', type=int)

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())



