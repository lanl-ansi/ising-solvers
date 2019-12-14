#!/usr/bin/env python3

import argparse, json, random, time, math
from collections import namedtuple

import bqpjson


VariableAssignment = namedtuple('Assignment', ['variable', 'value', 'energy'])

def evaluate(data, assignment):
    objective = 0.0
    for lt in data['linear_terms']:
        objective += lt['coeff'] * assignment[lt['id']]
    for qt in data['quadratic_terms']:
        objective += qt['coeff'] * assignment[qt['id_tail']] * assignment[qt['id_head']]
    return objective

def evaluate_neighbors(vid, linear_coeff, neighbors, assignment):
    objective = 0.0
    objective += linear_coeff * assignment[vid]
    for qt in neighbors:
        objective += qt['coeff'] * assignment[qt['id_tail']] * assignment[qt['id_head']]
    return objective


def main(args):
    with open(args.input_file) as input_file:
        data = json.load(input_file)

    bqpjson.validate(data)

    if data['variable_domain'] != 'spin':
        raise Exception('only spin domains are supported. Given {}'.format(data['variable_domain']))

    if args.seed is not None:
        random.seed(args.seed)

    scale, offset = data['scale'], data['offset']

    variable_ids = set(data['variable_ids'])
    variable_product_ids = set([(qt['id_tail'], qt['id_head']) for qt in data['quadratic_terms']])

    linear_coeff = {vid:0.0 for vid in data['variable_ids']}
    for lt in data['linear_terms']:
        linear_coeff[lt['id']] = lt['coeff']

    neighbors = {vid:[] for vid in data['variable_ids']}
    for qt in data['quadratic_terms']:
        neighbors[qt['id_tail']].append(qt)
        neighbors[qt['id_head']].append(qt)


    variable_order = [i for i in variable_ids]

    objective = float('inf')
    iterations = 0
    restarts = 0

    best_assignment = {vid:0 for vid in variable_ids}
    best_objective = objective
    start_time = time.process_time()
    end_time = start_time + args.runtime_limit

    while time.process_time() < end_time:
        random.shuffle(variable_order)
        assignment = {vid:0 for vid in variable_ids}
        assignment_value = 0
        unassigned = set(vid for vid in variable_ids)

        for i in range(len(variable_order)):
            iterations += 1
            assignments = []
            for vid in variable_order:
                if vid in unassigned:
                    #print(vid)
                    assignment[vid] = 1
                    #eval_up = evaluate(data, assignment)
                    eval_up = evaluate_neighbors(vid, linear_coeff[vid], neighbors[vid], assignment)
                    assignments.append(VariableAssignment(vid,  1, eval_up))

                    assignment[vid] = -1
                    #eval_down = evaluate(data, assignment)
                    eval_down = evaluate_neighbors(vid, linear_coeff[vid], neighbors[vid], assignment)
                    assignments.append(VariableAssignment(vid, -1, eval_down))

                    assignment[vid] = 0

            assignments.sort(key=lambda x: x.energy)
            #print(assignments)
            min_energy = assignments[0].energy
            min_assignments = []
            for assign in assignments:
                if assign.energy <= min_energy:
                    min_assignments.append(assign)
                else:
                    break
            assign = random.choice(min_assignments)
            assignment[assign.variable] = assign.value
            unassigned.remove(assign.variable)
            #print(assign.variable, assign.value)
            print('.', end = '')
            if time.process_time() > end_time:
                break

        if len(unassigned) > 0:
            for vid in unassigned:
                assignment[vid] = random.choice([-1,1])

        #print(assignment)

        objective = evaluate(data, assignment)
        if objective < best_objective:
            best_assignment = assignment
            best_objective = objective
            print('')
            print("best solution: {}, {}".format(best_objective, time.process_time() - start_time))

        if time.process_time() < end_time:
            restarts += 1
            print('R', end = '')
        #break

    runtime = time.process_time() - start_time

    objective = evaluate(data, best_assignment)
    if not math.isclose(objective, best_objective):
        raise Exception('final objective values do not match, incremental objective {}, true objective {}'.format(best_objective, objective))

    nodes = len(variable_ids)
    edges = len(variable_product_ids)
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
    if args.show_solution:
        print('BQP_SOLUTION, %d, %d, %f, %f, %s' % (nodes, edges, scaled_objective, runtime, best_solution))
    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % (nodes, edges, scaled_objective, scaled_lower_bound, objective, lower_bound, runtime, cut_count, node_count))


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help='the data file to operate on (.json)')
    parser.add_argument('-ss', '--show-solution', help='prints the a solution data line', action='store_true', default=False)

    parser.add_argument('-rtl', '--runtime-limit', help='runtime limit (sec.)', type=float, default=10)
    parser.add_argument('-s', '--seed', help='random seed', type=int)
    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())
