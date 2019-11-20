#!/usr/bin/env python3

### Requirements ###
# bqpjson v0.5 - pip install bqpjson
# gurobi v7.0 - http://www.gurobi.com/
#

import argparse, json, sys, random, time

from gurobipy import *

import bqpjson

int_tol = 1e-6

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

    if args.seed is not None:
        random.seed(args.seed)

    variable_ids = set(data['variable_ids'])
    variable_product_ids = set([(qt['id_tail'], qt['id_head']) for qt in data['quadratic_terms']])

    #print(data['linear_terms'])
    #print(data['quadratic_terms'])

    objective_best = float("Inf")
    solution_best = {}
    lp_solves = 0

    start_time = time.process_time()
    end_time = start_time + args.runtime_limit

    while time.process_time() < end_time:
        m = Model()
        #if args.runtime_limit != None:
        #    m.setParam('TimeLimit', args.runtime_limit)

        m.setParam('OutputFlag', 0)
        m.setParam('Threads', args.thread_limit)

        #m.setParam('Method', 2)
        #m.setParam('Crossover', 0)
        #m.setParam('Presolve', 2)
        #m.setParam('MIPFocus', 1)
        #m.setParam('MIPFocus', 2)

        variable_lookup = {}
        for vid in variable_ids:
            variable_lookup[(vid,vid)] = m.addVar(lb=-1, ub=1, vtype = GRB.CONTINUOUS, name='site_%04d' % vid)
        for pair in variable_product_ids:
            variable_lookup[pair] = m.addVar(lb=-1, ub=1, vtype = GRB.CONTINUOUS, name='product_%04d_%04d' % (pair[0], pair[1]))
        m.update()

        for i,j in variable_product_ids:
            #m.addConstr(variable_lookup[(i,i)]*variable_lookup[(j,j)] >= variable_lookup[(i,j)]*variable_lookup[(i,j)])
            m.addConstr(variable_lookup[(i,j)] >= -1*variable_lookup[(j,j)] + -1*variable_lookup[(i,i)] - 1)
            m.addConstr(variable_lookup[(i,j)] >=  1*variable_lookup[(j,j)] +  1*variable_lookup[(i,i)] - 1)
            m.addConstr(variable_lookup[(i,j)] <= -1*variable_lookup[(j,j)] +  1*variable_lookup[(i,i)] + 1)
            m.addConstr(variable_lookup[(i,j)] <=  1*variable_lookup[(j,j)] + -1*variable_lookup[(i,i)] + 1)
            #m.addGenConstrAnd(variable_lookup[(i,j)], [variable_lookup[(i,i)], variable_lookup[(j,j)]])

        if len(data['linear_terms']) <= 0 or all(lt['coeff'] == 0.0 for lt in data['linear_terms']):
            print('detected spin symmetry, adding symmetry breaking constraint')
            v1 = data['variable_ids'][0]
            m.addConstr(variable_lookup[(v1,v1)] == -1)

        obj = 0.0
        for lt in data['linear_terms']:
            i = lt['id']
            obj += lt['coeff']*variable_lookup[(i,i)]

        for qt in data['quadratic_terms']:
            i = qt['id_tail']
            j = qt['id_head']
            obj += qt['coeff']*variable_lookup[(i,j)]

        #print(obj)
        m.setObjective(obj, GRB.MINIMIZE)

        m.update()
        m.optimize()
        lp_solves += 1
        #print(m.Runtime)

        lower_bound = m.ObjVal

        remaining_vars = {i for i in data['variable_ids']}
        var_values = {vid:variable_lookup[(vid,vid)].X for vid in remaining_vars}
        #print(var_values)
        #for pair in variable_product_ids:
        #    print(pair, variable_lookup[pair].X)
        #break

        while any( abs(val) <= (1.0 - int_tol) for (vid, val) in var_values.items() ):
            var_values_order = sorted(var_values.items(), key=lambda x: abs(x[1]), reverse=True)

            #print(var_values_order)

            largest_value = 0.0
            largest_ids = []
            for (vid, val) in var_values.items():
                if abs(val) > largest_value:
                    largest_value = val
                    largest_ids = [vid]
                else:
                    if math.isclose(abs(val), largest_value):
                        largest_ids.append(vid)
                    else:
                        assert(abs(val) < largest_value)
                        break

            fixes = {}
            if largest_value < 1.0:
                vid_fix = random.choice(largest_ids)

                value_fix = random.choice([-1, 1])
                if largest_value > 0.0:
                    value_fix = 1
                if largest_value < 0.0:
                    value_fix = -1

                fixes[vid_fix] = value_fix
            else:
                for vid in largest_ids:
                    fixes[vid] = var_values[vid]

            #print(vid_fix, value_fix)

            for (vid, val) in fixes.items():
                m.addConstr(variable_lookup[(vid,vid)] == val)
                remaining_vars.remove(vid)

            m.update()
            m.optimize()
            lp_solves += 1
            #print("%f, %d" % (m.Runtime, len(remaining_vars)))

            #print(m.ObjVal)

            var_values = {vid:variable_lookup[(vid,vid)].X for vid in remaining_vars}
            #print(var_values)

        objective = m.ObjVal
        solution = {vid:variable_lookup[(vid,vid)].X for vid in data['variable_ids']}

        if objective < objective_best:
            print("")
            print("objective: %f" % objective)
            objective_best = objective
            solution_best = solution

        if objective_best <= lower_bound:
            print("")
            print("optimal solution found")
            break

        print("R", end = '')


    # if args.show_solution:
    #     print('')
    #     for k,v in variable_lookup.items():
    #         print('{:<18}: {}'.format(v.VarName, v.X))

    runtime = time.process_time() - start_time
    scaled_objective = data['scale']*(objective_best+data['offset'])
    scaled_lower_bound = data['scale']*(lower_bound+data['offset'])
    solution_best_str = ', '.join(["-1" if solution_best[vid] <= 0.5 else "1" for vid in data['variable_ids']])


    print()
    if args.show_solution:
        print('BQP_SOLUTION, %d, %d, %f, %f, %s' % (len(variable_ids), len(variable_product_ids), scaled_objective, runtime, solution_best_str))
    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % (len(variable_ids), len(variable_product_ids), scaled_objective, scaled_lower_bound, objective_best, lower_bound, runtime, 0, lp_solves))


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help='the data file to operate on (.json)')
    parser.add_argument('-ss', '--show-solution', help='prints the a solution data line', action='store_true', default=False)

    parser.add_argument('-rtl', '--runtime-limit', help='runtime limit (sec.)', type=float, default=10)
    parser.add_argument('-s', '--seed', help='random seed', type=int)

    parser.add_argument('-tl', '--thread-limit', help='gurobi thread limit', type=int, default=1)

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())



