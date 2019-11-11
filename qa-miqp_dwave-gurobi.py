#!/usr/bin/env python3

### Requirements ###
# bqpjson v0.5 - pip install bqpjson
# dwave-cloud-client v0.5.4 - pip install dwave-cloud-client
# gurobi v7.0 - http://www.gurobi.com/
#


import argparse, json, sys, time

import dwave.cloud as dc

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


    dw_config = dc.config.load_config(os.getenv("HOME")+"/dwave.conf", profile=args.profile)
    dw_chip_id = None

    if 'dw_endpoint' in data['metadata'] and not args.ignore_solver_metadata:
        dw_config['endpoint'] = data['metadata']['dw_endpoint']
        print('using d-wave endpoint provided in data file: %s' % dw_config['endpoint'])

    if 'dw_solver_name' in data['metadata'] and not args.ignore_solver_metadata:
        dw_config['solver'] = data['metadata']['dw_solver_name']
        print('using d-wave solver name provided in data file: %s' % dw_config['solver'])

    if 'dw_chip_id' in data['metadata'] and not args.ignore_solver_metadata:
        dw_chip_id = data['metadata']['dw_chip_id']
        print('found d-wave chip id in data file: %s' % dw_chip_id)

    client = dc.Client.from_config(**dw_config)
    solver = client.get_solver()

    if not dw_chip_id is None:
        if solver.properties['chip_id'] != dw_chip_id:
            print('WARNING: qpu chip ids do not match.  data: %s  hardware: %s' % (dw_chip_id, solver.properties['chip_id']))

    couplers = solver.properties['couplers']
    sites = solver.properties['qubits']

    site_range = tuple(solver.properties['h_range'])
    coupler_range = tuple(solver.properties['j_range'])

    h = {}
    #obj = data['offset']
    for lt in data['linear_terms']:
        i = lt['id']
        assert(not i in h)
        h[i] = lt['coeff']

    J = {}
    for qt in data['quadratic_terms']:
        i = qt['id_tail']
        j = qt['id_head']
        assert(not (i,j) in J)
        J[(i,j)] = qt['coeff']

    params = {
        'auto_scale': False,
        'num_reads': args.num_reads,
        'num_spin_reversal_transforms': int(args.num_reads/args.spin_reversal_transform_rate),
        'annealing_time': args.annealing_time
    }

    print('d-wave parameters:')
    for k,v in params.items():
        print('  {} - {}'.format(k,v))

    t0 = time.time()
    answers = solver.sample_ising(h, J, **params)
    solve_time = time.time() - t0

    client.close()

    for i in range(len(answers['energies'])):
        print('%f - %d' % (answers['energies'][i], answers['num_occurrences'][i]))
        if i > 50:
            print('showed 50 of %d' % len(answers['energies']))
            break

    if args.compute_hamming_distance:
        min_energy = min(e for e in answers['energies'])
        min_energy_states = []
        for i in range(len(answers['energies'])):
            if math.isclose(answers['energies'][i], min_energy):
                sol = answers['solutions'][i]
                min_energy_states.append([sol[vid] for vid in data['variable_ids']])

        for i in range(len(answers['energies'])):
            sol = answers['solutions'][i]
            state = [sol[vid] for vid in data['variable_ids']]
            min_dist = len(data['variable_ids'])

            for min_state in min_energy_states:
                dist = sum(min_state[i] != state[i] for i in range(len(data['variable_ids'])))
                if dist < min_dist:
                    min_dist = dist
            print('BQP_ENERGY, %d, %d, %f, %f, %d, %d' % (len(data['variable_ids']), len(data['quadratic_terms']), min_energy, answers['energies'][i], answers['num_occurrences'][i], min_dist))

    #print(answers['solutions'][0])
    qa_solution = answers['solutions'][0]

    nodes = len(data['variable_ids'])
    edges = len(data['quadratic_terms'])
    
    lt_lb = -sum(abs(lt['coeff']) for lt in data['linear_terms'])
    qt_lb = -sum(abs(qt['coeff']) for qt in data['quadratic_terms']) 
    lower_bound = lt_lb+qt_lb

    #best_objective = answers['energies'][0]
    #best_nodes = args.num_reads
    qpu_runtime = answers['timing']['total_real_time']/1000000.0
    #scaled_objective = data['scale']*(best_objective+data['offset'])
    #scaled_lower_bound = data['scale']*(lower_bound+data['offset'])

    #return

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
        variable_lookup[vid].start = (0 if qa_solution[vid] <= 0 else 1)
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

    # if args.show_solution:
    #     print('')
    #     for k,v in variable_lookup.items():
    #         print('{:<18}: {}'.format(v.VarName, v.X))

    lower_bound = m.MIPGap*m.ObjVal + m.ObjVal
    scaled_objective = data['scale']*(m.ObjVal+data['offset'])
    scaled_lower_bound = data['scale']*(lower_bound+data['offset'])
    best_solution = ', '.join(["-1" if variable_lookup[vid].X <= 0.5 else "1" for vid in data['variable_ids']])

    print('')
    if args.show_solution:
        print('BQP_SOLUTION, %d, %d, %f, %f, %s' % (len(variable_ids), len(variable_product_ids), scaled_objective, m.Runtime, best_solution))
    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % (len(variable_ids), len(variable_product_ids), scaled_objective, scaled_lower_bound, m.ObjVal, lower_bound, m.Runtime+qpu_runtime, m._cut_count, m.NodeCount))


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
    parser.add_argument('-ss', '--show-solution', help='prints the a solution data line', action='store_true', default=False)

    parser.add_argument('-p', '--profile', help='connection details to load from dwave.conf', default=None)
    parser.add_argument('-ism', '--ignore-solver-metadata', help='connection details to load from dwave.conf', action='store_true', default=False)

    parser.add_argument('-chd', '--compute-hamming-distance', help='computes the hamming distance from the best solution', action='store_true', default=False)
    parser.add_argument('-nr', '--num-reads', help='the number of reads to take from the d-wave hardware', type=int, default=200)
    parser.add_argument('-at', '--annealing-time', help='the annealing time of each d-wave sample', type=int, default=5)
    parser.add_argument('-srtr', '--spin-reversal-transform-rate', help='the number of reads to take before each spin reversal transform', type=int, default=101)

    parser.add_argument('-rtl', '--runtime-limit', help='gurobi runtime limit (sec.)', type=float)
    parser.add_argument('-tl', '--thread-limit', help='gurobi thread limit', type=int, default=1)
    parser.add_argument('-cuts', help='gurobi cuts parameter', type=int)

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())



