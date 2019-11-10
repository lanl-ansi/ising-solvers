#!/usr/bin/env python3

import argparse, json, time, math
from collections import namedtuple

import bqpjson


Model = namedtuple('Model', ['variables', 'linear', 'quadratic', 'linear_list', 'adjacent', 'quadratic_sums'])


def load_model(data):
    variables = data['variable_ids']
    linear = {lt['id']:lt['coeff'] for lt in data['linear_terms']}
    quadratic = {(qt['id_tail'],qt['id_head']):qt['coeff'] for qt in data['quadratic_terms']}

    linear_list = [0.0] * (max(variables) + 1) # list is faster than dict
    for var, coeff in linear.items():
        linear_list[var] = coeff
    adjacent = [None] * (max(variables) + 1) # list is faster than dict
    quadratic_sums = [0.0] * (max(variables) + 1) # list is faster than dict
    for qt in data['quadratic_terms']:
        i, j, coeff = qt['id_tail'], qt['id_head'], qt['coeff']
        if adjacent[i] is None:
            adjacent[i] = []
        adjacent[i].append((j, coeff))
        if adjacent[j] is None:
            adjacent[j] = []
        adjacent[j].append((i, coeff))
        quadratic_sums[i] += coeff
        quadratic_sums[j] += coeff

    return Model(variables, linear, quadratic, linear_list, adjacent, quadratic_sums)


def evaluate(model, assignment):
    objective = 0.0
    for var, coeff in model.linear.items():
        objective += coeff * assignment[var]
    for (var1, var2), coeff in model.quadratic.items():
        objective += coeff * assignment[var1] * assignment[var2]
    return objective


def sign(x):
    return 1.0 if x > 0 else -1.0


def f(x, y):
    return min(x + y, 0) - min(-x + y, 0) - x


def make_zero_messages(model):
    return {(i,j): 0.0 for i in model.variables for j, _ in model.adjacent[i]}


def update_messages(model, messages, swap, incomings):
    for i in model.variables:
        for j, coeff in model.adjacent[i]:
            swap[i, j] = f(2 * coeff, 2 * model.linear_list[i] + incomings[i] - messages[j, i])
    return swap, messages


def compute_incomings(model, messages):
    incomings = [0.0] * len(model.linear_list)
    for i in model.variables:
        for j, _ in model.adjacent[i]:
            incomings[j] += messages[i, j]
    return incomings


def update_assignment(model, messages, assignment):
    incomings = compute_incomings(model, messages)
    for i in model.variables:
        assignment[i] = -sign(f(2 * model.quadratic_sums[i], 2 * model.linear_list[i] + incomings[i]))
    return incomings


def main(args):
    with open(args.input_file) as input_file:
        data = json.load(input_file)

    bqpjson.validate(data)

    if data['variable_domain'] != 'spin':
        raise Exception('only spin domains are supported. Given {}'.format(data['variable_domain']))

    model = load_model(data)
    scale, offset = data['scale'], data['offset']

    messages = make_zero_messages(model)
    swap = make_zero_messages(model) # swap space when updating messages
    assignment = [None] * len(model.linear_list)
    incomings = update_assignment(model, messages, assignment)
    objective = evaluate(model, assignment)
    iterations = 1

    best_assignment = [i for i in assignment]
    best_objective = objective
    start_time = time.process_time()
    end_time = start_time + args.runtime_limit

    while time.process_time() < end_time:
        messages, swap = update_messages(model, messages, swap, incomings)
        incommings = update_assignment(model, messages, assignment)
        objective = evaluate(model, assignment)
        if objective < best_objective:
            best_objective = objective
            best_assignment = [i for i in assignment]
        iterations += 1

        if args.show_objectives:
            print('objective:',  objective)
        if args.show_scaled_objectives:
            print('scaled objective:', scale * (objective + offset))

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
    print('best objective:', objective)
    print('best scaled objective:', scaled_objective)

    print()
    if args.show_solution:
        print('BQP_SOLUTION, %d, %d, %f, %f, %s' % (nodes, edges, scaled_objective, runtime, best_solution))
    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % (nodes, edges, scaled_objective, scaled_lower_bound, objective, lower_bound, runtime, cut_count, node_count))


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help='the data file to operate on (.json)')
    parser.add_argument('-ss', '--show-solution', help='prints the a solution data line', action='store_true', default=False)

    parser.add_argument('-so', '--show-objectives', help='print the objectives seen by the program', action='store_true', default=False)
    parser.add_argument('-sso', '--show-scaled-objectives', help='print the scaled objectives seen by the program', action='store_true', default=False)
    parser.add_argument('-rtl', '--runtime-limit', help='runtime limit (sec.)', type=float, default=10)
    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())
