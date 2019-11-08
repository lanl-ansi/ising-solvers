#!/usr/bin/env python3

import argparse, json, random, time, math
from collections import namedtuple

import bqpjson


Model = namedtuple('Model', ['variables', 'linear', 'quadratic', 'linear_list','adjacent'])


def load_model(data):
    variables = data['variable_ids']
    linear = {lt['id']:lt['coeff'] for lt in data['linear_terms']}
    quadratic = {(qt['id_tail'],qt['id_head']):qt['coeff'] for qt in data['quadratic_terms']}

    # prepare for faster evaluation of flip_delta
    linear_list = [0.0] * (max(variables) + 1) # list is faster than dict
    for var, coeff in linear.items():
        linear_list[var] = coeff
    adjacent = [None] * (max(variables) + 1) # list is faster than dict
    for qt in data['quadratic_terms']:
        i, j, coeff = qt['id_tail'], qt['id_head'], qt['coeff']
        if adjacent[i] is None:
            adjacent[i] = []
        adjacent[i].append((j, coeff))
        if adjacent[j] is None:
            adjacent[j] = []
        adjacent[j].append((i, coeff))

    return Model(variables, linear, quadratic, linear_list, adjacent)


def make_random_assignemnt(model):
    assignment = [None] * (max(model.variables) + 1) # list is faster than dict
    for var in model.variables:
        assignment[var] = random.choice([-1.0,1.0])
    return assignment


def make_all_up_assignemnt(model):
    assignment = [None] * (max(model.variables) + 1) # list is faster than dict
    for var in model.variables:
        assignment[var] = 1.0
    return assignment


def make_all_down_assignemnt(model):
    assignment = [None] * (max(model.variables) + 1) # list is faster than dict
    for var in model.variables:
        assignment[var] = -1.0
    return assignment


def evaluate(model, assignment):
    objective = 0.0
    for var, coeff in model.linear.items():
        objective += coeff * assignment[var]
    for (var1, var2), coeff in model.quadratic.items():
        objective += coeff * assignment[var1] * assignment[var2]
    return objective


def flip(assignment, variable):
    assignment[variable] = -1.0*assignment[variable]


def flip_delta(model, assignment, variable):
    delta = 0.0
    delta += -1.0 * model.linear_list[variable] * assignment[variable]
    for i, coeff in model.adjacent[variable]:
        delta += -1.0 * coeff * assignment[i] * assignment[variable]
    return 2.0*delta


def step(model, assignment, objective):
    best_var = None
    best_delta = math.inf
    for var in model.variables:
        delta = flip_delta(model, assignment, var)
        if delta < best_delta:
            best_var = var
            best_delta = delta
    if best_delta >= 0.0:
        return None
    else:
        flip(assignment, best_var)
        return best_delta


def main(args):
    with open(args.input_file) as input_file:
        data = json.load(input_file)

    bqpjson.validate(data)

    if data['variable_domain'] != 'spin':
        raise Exception('only spin domains are supported. Given {}'.format(data['variable_domain']))

    if args.initial_assignment == 'ran':
        make_restart_assignment = make_random_assignemnt
    elif args.initial_assignment == 'ones':
        make_restart_assignment = make_all_up_assignemnt
    elif args.initial_assignment == 'zeros':
        make_restart_assignment = make_all_down_assignemnt
    else:
        assert False

    if args.seed is not None:
        random.seed(args.seed)

    model = load_model(data)
    scale, offset = data['scale'], data['offset']

    assignment = make_restart_assignment(model)
    objective = evaluate(model, assignment)
    iterations = 1
    restarts = 0

    best_assignment = [i for i in assignment]
    best_objective = objective
    start_time = time.process_time()
    end_time = start_time + args.runtime_limit

    while time.process_time() < end_time:
        result = step(model, assignment, objective)
        if result is None: # restart
            assignment = make_restart_assignment(model)
            objective = evaluate(model, assignment)
            restarts += 1
        else: # move downward
            objective += result
            iterations += 1
        if objective < best_objective:
            best_objective = objective
            best_assignment = [i for i in assignment]
            #variable_up = sum(assignment[i] > 0 for i in model.variables)
            #variable_down = sum(assignment[i] <= 0 for i in model.variables)
            #print(objective, variable_up, variable_down)

        if args.show_objectives:
            print('objective:',  objective)
        if args.show_scaled_objectives:
            print('scaled objective:', scale * (objective + offset))

    objective = evaluate(model, best_assignment)
    if not math.isclose(objective,best_objective):
        raise Exception('final objective values do not match, incremental objective {}, true objective {}'.format(best_objective, objective))


    runtime = time.process_time() - start_time
    nodes = len(model.variables)
    edges = len(model.quadratic)
    objective = best_objective
    lower_bound = - sum(abs(lt['coeff']) for lt in data['linear_terms']) - sum(abs(qt['coeff']) for qt in data['quadratic_terms']) 
    scaled_objective = scale * (objective + offset)
    scaled_lower_bound = scale * (lower_bound + offset)
    best_solution = ', '.join([str(int(best_assignment[vid])) for vid in data['variable_ids']])
    cut_count = 0
    node_count = iterations

    print()
    print('iterations:', iterations)
    print('restarts:', restarts)
    print('best objective:', objective)
    print('best scaled objective:', scaled_objective)

    print()
    print('BQP_SOLUTION, %d, %d, %f, %f, %s' % (nodes, edges, scaled_objective, runtime, best_solution))
    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % (nodes, edges, scaled_objective, scaled_lower_bound, objective, lower_bound, runtime, cut_count, node_count))


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help='the data file to operate on (.json)')
    parser.add_argument('-so', '--show-objectives', help='print the objectives seen by the program', action='store_true', default=False)
    parser.add_argument('-sso', '--show-scaled-objectives', help='print the scaled objectives seen by the program', action='store_true', default=False)
    parser.add_argument('-rtl', '--runtime-limit', help='runtime limit (sec.)', type=float, default=10)
    parser.add_argument('-ia', '--initial_assignment', help='initial assignment when restarting', choices=['ran', 'ones', 'zeros'], default='ran')
    parser.add_argument('-s', '--seed', help='random seed', type=int)
    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())
