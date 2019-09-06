#!/usr/bin/env python3

### Requirements ###
# bqpjson v0.5 - pip install bqpjson
# dwave-cloud-client v0.5.4 - pip install dwave-cloud-client

# refrence https://arxiv.org/abs/1708.03049

import argparse, json, time, os, sys, math

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


    t0 = time.perf_counter()

    anneal_offset_ranges = solver.properties['anneal_offset_ranges']
    anneal_offset_step = solver.properties['anneal_offset_step']
    offsets = [0.0 for v in range(0, max(solver.nodes)+1)]

    params = {
        'auto_scale': False,
        'num_reads': args.num_reads,
        'num_spin_reversal_transforms': int(args.num_reads/args.spin_reversal_transform_rate),
        'annealing_time': args.annealing_time,
        'anneal_offsets': offsets
    }

    print('d-wave parameters:')
    for k,v in params.items():
        print('  {} - {}'.format(k,v))

    for i in range(0, args.max_iterations):
        print('iteration: ', i)
        answers = solver.sample_ising(h, J, **params)

        samples = float(sum(answers['num_occurrences']))
        mean_energy = sum(answers['num_occurrences'][i] * answers['energies'][i] for i in range(0, len(answers['num_occurrences']))) / samples
        print('mean energy {}'.format(mean_energy))

        #f = floppyness_h1(answers, data['variable_ids'])
        f = floppyness(answers, data['variable_ids'])
        #print(f)

        #for v in data['variable_ids']:
        #    offsets[v] = offsets[v] + (args.alpha*f[v] - offsets[v])/(1 + math.sqrt(i))
        #print([offsets[v] for v in data['variable_ids']])

        #for v in data['variable_ids']:
        #    if f[v] > 0.0:
        #        offsets[v] = offsets[v] + anneal_offset_ranges[v][1]/10.0

        #for (i,v) in sorted(f.items(), key=lambda x: abs(x[1]), reverse=True):
        #    print('{} - {}'.format(i,v))

        # seems to work
        # for v in data['variable_ids']:
        #     if abs(f[v]) > 0.01:
        #         offsets[v] = offsets[v] + anneal_offset_ranges[v][0]/20.0
        #     if abs(f[v]) < 0.005:
        #         offsets[v] = offsets[v] + anneal_offset_ranges[v][1]/20.0


        max_flop = max(abs(v) for v in f.values())
        min_flop = min(abs(v) for v in f.values())
        print('flop range {} {}'.format(min_flop, max_flop))
        mid_flop = (max_flop+min_flop)/2.0
        for v in data['variable_ids']:
            if abs(f[v]) > 0.1:
                #offsets[v] = offsets[v] + anneal_offset_ranges[v][0]/float(args.max_iterations)
                offsets[v] = offsets[v] - 3*(anneal_offset_step)
            if abs(f[v]) < 0.01:
                #offsets[v] = offsets[v] + anneal_offset_ranges[v][1]/float(args.max_iterations)
                offsets[v] = offsets[v] + anneal_offset_step
            offsets[v] = max(offsets[v], anneal_offset_ranges[v][0])
            offsets[v] = min(offsets[v], anneal_offset_ranges[v][1])

        #print([offsets[v] for v in data['variable_ids']])

    client.close()
    solve_time = time.perf_counter() - t0


    for i in range(len(answers['energies'])):
        print('%f - %d' % (answers['energies'][i], answers['num_occurrences'][i]))
    samples = float(sum(answers['num_occurrences']))
    mean_energy = sum(answers['num_occurrences'][i] * answers['energies'][i] for i in range(0, len(answers['num_occurrences']))) / samples
    sd_energy = sum(answers['num_occurrences'][i] * (answers['energies'][i] - mean_energy)**2 for i in range(0, len(answers['num_occurrences']))) / samples
    sd_energy = math.sqrt(sd_energy/samples)
    print('energy mean / sd: {} {}'.format(mean_energy, sd_energy))

    nodes = len(data['variable_ids'])
    edges = len(data['quadratic_terms'])
    
    lt_lb = -sum(abs(lt['coeff']) for lt in data['linear_terms'])
    qt_lb = -sum(abs(qt['coeff']) for qt in data['quadratic_terms']) 
    lower_bound = lt_lb+qt_lb

    best_objective = answers['energies'][0]
    best_nodes = args.num_reads
    best_runtime = answers['timing']['total_real_time']/1000000.0
    scaled_objective = data['scale']*(best_objective+data['offset'])
    scaled_lower_bound = data['scale']*(lower_bound+data['offset'])

    print('ISING_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d, %f' % (nodes, edges, scaled_objective, scaled_lower_bound, best_objective, lower_bound, best_runtime, 0, best_nodes, solve_time))


def floppyness_h1(answers, variable_ids):
    print('compute floppyness')
    samples = sum(answers['num_occurrences'])
    flop = {v:0 for v in variable_ids}
    for a1 in answers['samples']:
        for a2 in answers['samples']:
            hamming_dist = sum(a1[v] != a2[v] for v in variable_ids)
            if hamming_dist == 1:
                for v in variable_ids:
                    if a1[v] != a2[v]:
                        flop[v] = flop[v] + 1
                        break
    for v in variable_ids:
        assert(flop[v] <= samples)
    return {v:flop[v]/samples for v in variable_ids}


def floppyness(answers, variable_ids):
    print('compute floppyness')
    samples = sum(answers['num_occurrences'])
    flop = {v:0 for v in variable_ids}
    for a in answers['samples']:
        for v in variable_ids:
            flop[v] = flop[v] + a[v]

    return {v:flop[v]/samples for v in variable_ids}


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help='the data file to operate on (.json)')

    parser.add_argument('-p', '--profile', help='connection details to load from dwave.conf', default=None)
    parser.add_argument('-ism', '--ignore-solver-metadata', help='connection details to load from dwave.conf', action='store_true', default=False)

    parser.add_argument('-nr', '--num-reads', help='the number of reads to take from the d-wave hardware', type=int, default=10000)
    parser.add_argument('-at', '--annealing-time', help='the annealing time of each d-wave sample', type=int, default=5)
    parser.add_argument('-srtr', '--spin-reversal-transform-rate', help='the number of reads to take before each spin reversal transform', type=int, default=100)

    parser.add_argument('-mi', '--max-iterations', help='the maximum number of iterations before convergence is reached', type=int, default=10)
    parser.add_argument('-a', '--alpha', help='annealing offset update parameter', type=float, default=0.04)

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())





