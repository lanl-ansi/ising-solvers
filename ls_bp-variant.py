#!/usr/bin/env python3

import argparse, json, time, math, random

import bqpjson


class Model:
    def __init__(self, variables, linear, quadratic, linear_list, adjacent, offset):
        self.variables = variables
        self.linear = linear
        self.quadratic = quadratic
        self.linear_list = linear_list
        self.adjacent = adjacent
        self.offset = offset


def load_model(data):
    variables = set(data['variable_ids'])
    linear = {lt['id']:lt['coeff'] for lt in data['linear_terms']}
    quadratic = {(qt['id_tail'],qt['id_head']):qt['coeff'] for qt in data['quadratic_terms']}

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
    
    return Model(variables, linear, quadratic, linear_list, adjacent, 0.0)


def evaluate(model, assignment):
    objective = model.offset
    for var, coeff in model.linear.items():
        objective += coeff * assignment[var]
    for (var1, var2), coeff in model.quadratic.items():
        objective += coeff * assignment[var1] * assignment[var2]
    return objective


def sign(x):
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0


def f(x, y):
    return min(x + y, 0) - min(-x + y, 0) - x


def make_zero_messages(model):
    return [[0.0] * len(model.linear_list) for _ in model.linear_list]


def update_messages(model, messages, scratch, incomings, threshold):
    '''write updated messages to scratch and swap scratch and messages'''
    max_change = 0.0
    for i in model.variables:
        for j, coeff in model.adjacent[i]:
            scratch[i][j] = f(2 * coeff, 2 * model.linear_list[i] + incomings[i] - messages[j][i])
            max_change = max(max_change, abs(scratch[i][j] - messages[i][j]))
    converged = max_change < threshold
    return scratch, messages, converged


def compute_incomings(model, messages):
    '''sum of incoming messages for each spin'''
    incomings = [0.0] * len(model.linear_list)
    for i in model.variables:
        for j, _ in model.adjacent[i]:
            incomings[j] += messages[i][j]
    return incomings


def update_assignment(model, messages, assignment):
    '''update the assignment in place, and return the incomings to save later computation'''
    incomings = compute_incomings(model, messages)
    for i in model.variables:
        assignment[i] = -sign(2 * model.linear_list[i] + incomings[i])
        if math.isclose(assignment[i], 0.0):
            if random.random() < 0.5:
                assignment[i] = -1
            else:
                assignment[i] = 1
    return incomings


def update_assignment_and_fix_one(model, messages, assignment):
    incomings = compute_incomings(model, messages)
    # update assignment and find out the variable to fix
    weights = {}
    for i in model.variables:
        assignment[i] = -sign(2 * model.linear_list[i] + incomings[i])
        weights[i] = abs(2 * model.linear_list[i] + incomings[i])
        if math.isclose(assignment[i], 0.0):
            assignment[i] = -1.0 if random.random() < 0.5 else 1.0
    var, _ = max(weights.items(), key=lambda pair: pair[1])
    # modify the model
    model.variables.remove(var)
    model.offset += model.linear.get(var, 0.0) * assignment[var]
    model.linear.pop(var, None)
    for i, coeff in model.adjacent[var]:
        model.linear.setdefault(i, 0.0)
        model.linear[i] += coeff * assignment[var]
        model.linear_list[i] += coeff * assignment[var]
    model.quadratic = {(i,j):coeff for (i,j),coeff in model.quadratic.items() if i != var and j != var}
    model.adjacent = [[] for _ in range(len(model.linear_list))]
    for (i,j), coeff in model.quadratic.items():
        if model.adjacent[i] is None: model.adjacent[i] = []
        model.adjacent[i].append((j, coeff))
        if model.adjacent[j] is None: model.adjacent[j] = []
        model.adjacent[j].append((i, coeff))
    return incomings, var


def main(args):
    with open(args.input_file) as input_file:
        data = json.load(input_file)

    bqpjson.validate(data)

    if data['variable_domain'] != 'spin':
        raise Exception('only spin domains are supported. Given {}'.format(data['variable_domain']))

    model = load_model(data)
    scale, offset = data['scale'], data['offset']
    coeff_sum = max(*(abs(coeff) for coeff in model.linear.values()), *(abs(coeff) for coeff in model.quadratic.values()))
    threshold = coeff_sum * args.relative_threshold

    messages = make_zero_messages(model)
    scratch = make_zero_messages(model) # swap space when updating messages
    assignment = [None] * len(model.linear_list)
    incomings = update_assignment(model, messages, assignment)
    objective = evaluate(model, assignment)
    iterations = 1

    best_assignment = [i for i in assignment]
    best_objective = objective
    start_time = time.process_time()
    end_time = start_time + args.runtime_limit

    while time.process_time() < end_time:
        messages, scratch, converged = update_messages(model, messages, scratch, incomings, threshold)
        if converged:
            incomings, var = update_assignment_and_fix_one(model, messages, assignment)
            if not model.variables: break
            if args.show_fixed_variables: print('fix variable {} = {}'.format(var, assignment[var]))
        else:
            incomings = update_assignment(model, messages, assignment)
        objective = evaluate(model, assignment)
        if objective < best_objective:
            best_objective = objective
            best_assignment = [i for i in assignment]
        iterations += 1

        if args.show_objectives:
            print('objective:',  objective)
        if args.show_scaled_objectives:
            print('scaled objective:', scale * (objective + offset))
    
    original_model = load_model(data)
    true_objective = evaluate(original_model, best_assignment)
    if not math.isclose(true_objective, best_objective):
        raise Exception('final objective values do not match, incremental objective {}, true objective {}'.format(best_objective, true_objective)) 

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

    parser.add_argument('-sfv', '--show-fixed-variables', help='print the varibles being fixed', action='store_true', default=False)
    parser.add_argument('-rt', '--relative-threshold', help='relative threshold of message passing consensus with respect to the sum of all coefficients', type=float, default=6.0)
    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())
