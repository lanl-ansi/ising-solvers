#!/usr/bin/env python3

### Requirements ###
# bqpjson v0.5 - pip install bqpjson
# dwave-cloud-client v0.5.4 - pip install dwave-cloud-client

import argparse, json, math, time, os, sys

import dwave.cloud as dc

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

    # A core assumption of this solver is that the given B-QP will magically be compatable with the given D-Wave QPU
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
        'num_spin_reversal_transforms': int(math.ceil(args.num_reads/args.spin_reversal_transform_rate))-1,
        'annealing_time': args.annealing_time
    }

    if args.postprocess_optimization:
        params['postprocess'] = 'optimization'

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


    nodes = len(data['variable_ids'])
    edges = len(data['quadratic_terms'])

    lt_lb = -sum(abs(lt['coeff']) for lt in data['linear_terms'])
    qt_lb = -sum(abs(qt['coeff']) for qt in data['quadratic_terms']) 
    lower_bound = lt_lb+qt_lb

    best_objective = answers['energies'][0]
    best_solution = ', '.join([str(answers['solutions'][0][vid]) for vid in data['variable_ids']])
    best_nodes = args.num_reads
    best_runtime = answers['timing']['total_real_time']/1000000.0
    scaled_objective = data['scale']*(best_objective+data['offset'])
    scaled_lower_bound = data['scale']*(lower_bound+data['offset'])

    print()
    if args.show_solution:
        print('BQP_SOLUTION, %d, %d, %f, %f, %s' % (nodes, edges, scaled_objective, best_runtime, best_solution))
    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % (nodes, edges, scaled_objective, scaled_lower_bound, best_objective, lower_bound, best_runtime, 0, best_nodes))


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help='the data file to operate on (.json)')
    parser.add_argument('-ss', '--show-solution', help='prints the a solution data line', action='store_true', default=False)

    parser.add_argument('-p', '--profile', help='connection details to load from dwave.conf', default=None)
    parser.add_argument('-ism', '--ignore-solver-metadata', help='connection details to load from dwave.conf', action='store_true', default=False)

    parser.add_argument('-chd', '--compute-hamming-distance', help='computes the hamming distance from the best solution', action='store_true', default=False)

    parser.add_argument('-nr', '--num-reads', help='the number of reads to take from the d-wave hardware', type=int, default=10000)
    parser.add_argument('-at', '--annealing-time', help='the annealing time of each d-wave sample', type=int, default=5)
    parser.add_argument('-srtr', '--spin-reversal-transform-rate', help='the number of reads to take before each spin reversal transform', type=int, default=100)
    parser.add_argument('-ppo', '--postprocess-optimization', help='set the postprocess parameter to optimization', action='store_true', default=False)


    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())





