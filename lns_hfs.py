#!/usr/bin/env python3

### Requirements ###
# bqpjson v0.5.3 - pip install bqpjson
# qubo (cli) - https://github.com/lanl-ansi/HFS-algorithm 
# docker and the hfs_alg container are required for docker-based execution
#

### Note ###
# these are good articles to reference when using this solver
#
# @misc{1409.3934,
#   Author = {Alex Selby},
#   Title = {Efficient subgraph-based sampling of Ising-type models with frustration},
#   Year = {2014},
#   Eprint = {arXiv:1409.3934},
#   url = {https://arxiv.org/abs/1409.3934},
# }
#
# @inproceedings{Hamze:2004:FT:1036843.1036873,
#  author = {Hamze, Firas and de Freitas, Nando},
#  title = {From Fields to Trees},
#  booktitle = {Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence},
#  series = {UAI '04},
#  year = {2004},
#  isbn = {0-9749039-0-6},
#  location = {Banff, Canada},
#  pages = {243--250},
#  numpages = {8},
#  url = {http://dl.acm.org/citation.cfm?id=1036843.1036873},
#  acmid = {1036873},
#  publisher = {AUAI Press},
#  address = {Arlington, Virginia, United States},
# }
#

import sys, os, argparse, json, random, tempfile

from io import StringIO

from subprocess import Popen
from subprocess import PIPE

from collections import namedtuple

import bqpjson

HFS_DIR = 'hfs'
Result = namedtuple('Result', ['nodes', 'objective', 'runtime'])

# NOTE: this code assumes the HFS solver (i.e. "qubo") is in available in the local path
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

    if not os.path.exists(HFS_DIR):
        os.makedirs(HFS_DIR)

    data = bqpjson.spin_to_bool(data)

    hfs_data = StringIO()
    hfs_scale, hfs_offset = bqpjson.bqpjson_to_hfs(data, hfs_data, precision=args.precision)

    hfs_data = hfs_data.getvalue()

    if args.show_input:
        print('INFO: hfs solver input', file=sys.stderr)
        print(hfs_data, file=sys.stderr)

    first_line = hfs_data.split('\n', 1)[0]
    chimera_degree_effective = int(first_line.split()[0])
    print('INFO: found effective chimera degree {}'.format(chimera_degree_effective), file=sys.stderr)

    tmp_hfs_file = create_tmp_file(prefix='hfs_')
    tmp_sol_file = create_tmp_file(prefix='sol_')

    #print('INFO: hfs temp input file {}'.format(tmp_hfs_file))

    print('INFO: hfs temp solution file {}'.format(tmp_sol_file))
    print('INFO: writing data to {}'.format(tmp_hfs_file), file=sys.stderr)
    with open(tmp_hfs_file, 'w') as hfs_file:
        hfs_file.write(hfs_data)

    # print(err.getvalue())

    if args.docker_run:
        # assume that the hfs_alg container is available
        volume_map = '{}:/{}'.format(os.path.abspath(HFS_DIR), HFS_DIR)
        cmd = ['docker', 'run', '-v', volume_map, 'hfs_alg']
    else:
        # assume that the qubo executable is natively accessible
        cmd = ['qubo']

    # s - seed
    # m0 - mode of operation, try to find minimum value by heuristic search
    # N - size of Chimera graph 
    cmd.extend(['-s', str(args.seed), '-m0', '-N', str(chimera_degree_effective)])

    if args.runtime_limit != None:
        # t - min run time for some modes
        # T - max run time for some modes
        cmd.extend(['-t', str(args.runtime_limit), '-T', str(args.runtime_limit+10)])
    cmd.extend(['-O', tmp_sol_file])
    cmd.append(tmp_hfs_file)

    print('INFO: running command {}'.format(cmd), file=sys.stderr)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()

    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')

    print('INFO: qubo stderr', file=sys.stderr)
    print(stderr, file=sys.stderr)

    print('INFO: qubo stdout', file=sys.stderr)
    print(stdout, file=sys.stderr)

    results = []
    reading_results = False
    for line in stdout.split('\n'):
        if not reading_results:
            if 'Nodes' in line and 'bv' in line and 'nsol' in line:
                reading_results = True
        else:
            parts = line.split()
            if len(parts) == 3:
                parts = (int(parts[0]), int(parts[1]), float(parts[2]))
                results.append(Result(*parts))
            else:
                reading_results = False

    print('INFO: found {} result lines'.format(len(results)), file=sys.stderr)
    assert(len(results) > 0)

    if args.show_hfs_solution:
        print('INFO: qubo solution', file=sys.stderr)
        with open(tmp_sol_file) as f:
            print(f.read(), file=sys.stderr)

    nodes = len(data['variable_ids'])
    edges = len(data['quadratic_terms'])
    
    lt_lb = -sum(abs(lt['coeff']) for lt in data['linear_terms'])
    qt_lb = -sum(abs(qt['coeff']) for qt in data['quadratic_terms'])
    lower_bound = lt_lb + qt_lb
    scaled_lower_bound = data['scale'] * (lower_bound + data['offset'])

    best_nodes = results[-1].nodes
    best_runtime = results[-1].runtime

    best_hfs_objective = results[-1].objective
    scaled_hfs_objective = hfs_scale * (best_hfs_objective + hfs_offset)

    verify_hfs_solution(tmp_hfs_file, tmp_sol_file, best_hfs_objective)

    result = evaluate_solution_in_bqpjson(data, tmp_sol_file)
    if result is None:
        print("INFO: using objective evaluated in HFS data", file=sys.stderr)
        best_objective, scaled_objective = best_hfs_objective, scaled_hfs_objective
    else:
        print("INFO: using objective evaluated in bqpjson data", file=sys.stderr)
        best_objective, scaled_objective = result
        print()
        print("INFO: scaled HFS objective = {}".format(scaled_hfs_objective), file=sys.stderr)
        print("INFO: scaled bqpjson objective = {}".format(scaled_objective), file=sys.stderr)
        print("INFO: HFS error = {}".format(scaled_hfs_objective - scaled_objective), file=sys.stderr)
    print()

    print()
    if args.show_solution:
        hfs_solution = read_solution(tmp_sol_file)
        chimera_degree = data['metadata']['chimera_degree']
        chimera_cell_size = data['metadata']['chimera_cell_size']
        bqp_solution = ', '.join(["-1" if hfs_solution[hfs_site_idx(vid, chimera_degree, chimera_cell_size)] <= 0.5 else "1" for vid in data['variable_ids']])
        print('BQP_SOLUTION, %d, %d, %f, %f, %s' % (nodes, edges, scaled_objective, best_runtime, bqp_solution))
    print('BQP_DATA, %d, %d, %f, %f, %f, %f, %f, %d, %d' % (nodes, edges, scaled_objective, scaled_lower_bound, best_objective, lower_bound, best_runtime, 0, best_nodes))

    remove_tmp_file(tmp_hfs_file)
    remove_tmp_file(tmp_sol_file)


def create_tmp_file(prefix=None):
    fd, filename = tempfile.mkstemp(prefix=prefix, dir=HFS_DIR)
    os.close(fd)
    filename_parts = filename.split(os.sep+HFS_DIR+os.sep)
    filename = os.path.join(HFS_DIR, filename_parts[1])
    return filename


def remove_tmp_file(filename):
    print('INFO: removing file {}'.format(filename), file=sys.stderr)
    try:
        os.remove(filename)
    except:
        print('WARNING: removing file {} failed'.format(filename), file=sys.stderr)


def verify_hfs_solution(tmp_hfs_file, tmp_sol_file, hfs_objective):
    try:
        problem = read_hfs_problem(tmp_hfs_file)
    except:
        print('WARNING: failed to verify solution. Cannot read problem file', file=sys.stderr)
        return
    try:
        solution = read_solution(tmp_sol_file)
    except:
        print('WARNING: failed to verify solution. Cannot read solution file', file=sys.stderr)
        return
    energy = evaluate_energy(problem, solution)
    if energy == hfs_objective:
        print('INFO: HFS solution verified', file=sys.stderr)
    else:
        print("ERROR: solution and HFS objective do NOT match (solution's energy = {}, HFS objective = {})".format(energy, hfs_objective), file=sys.stderr)
        quit()


def evaluate_solution_in_bqpjson(bqpjson_data, tmp_sol_file):
    try:
        solution = read_solution(tmp_sol_file)
    except:
        print('WARNING: failed to evaluate solution in bqpjson data. Cannot read solution file', file=sys.stderr)
        return None
    problem = load_bqpjson_problem(bqpjson_data)
    energy = evaluate_energy(problem, solution)
    return energy, bqpjson_data['scale'] * (energy + bqpjson_data['offset'])


def read_hfs_problem(path):
    problem = {}
    with open(path) as f:
        next(f)
        for line in f:
            values = [int(w) for w in line.split()]
            i, j = sorted([tuple(values[0:4]), tuple(values[4:8])])
            weight = values[8]
            problem[i, j] = weight
    return problem


def load_bqpjson_problem(data):
    chimera_degree = data['metadata']['chimera_degree']
    chimera_cell_size = data['metadata']['chimera_cell_size']
    assert chimera_cell_size % 2 == 0
    problem = {}
    for lt in data['linear_terms']:
        i = hfs_site_idx(lt['id'], chimera_degree, chimera_cell_size)
        problem[i, i] = lt['coeff']
    for qt in data['quadratic_terms']:
        i, j = hfs_site_idx(qt['id_tail'], chimera_degree, chimera_cell_size), hfs_site_idx(qt['id_head'], chimera_degree, chimera_cell_size)
        problem[i, j] = qt['coeff']
    return problem


def hfs_site_idx(bqpjson_idx, chimera_degree, chimera_cell_size):
    cell_idx, site_idx = divmod(bqpjson_idx, chimera_cell_size)
    row, col = divmod(cell_idx, chimera_degree)
    a, b = divmod(site_idx, chimera_cell_size//2)
    return row, col, a, b


def read_solution(path):
    solution = {}
    with open(path) as f:
        for line in f:
            values = [int(w) for w in line.split()]
            site = tuple(values[0:4])
            assignment = values[4]
            solution[site] = assignment
    return solution


def evaluate_energy(problem, solution):
    energy = 0.0
    for (i, j), coeff in problem.items():
        if i == j:
            energy += coeff * solution[i]
        else:
            energy += coeff * solution[i] * solution[j]
    return energy


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input-file', help='the data file to operate on (.json)')
    parser.add_argument('-ss', '--show-solution', help='prints the a solution data line', action='store_true', default=False)

    parser.add_argument('-dr', '--docker-run', help='run in hfs_alg docker container', action='store_true', default=False)
    parser.add_argument('-shs', '--show-hfs-solution', help='prints the raw hfs solution data', action='store_true', default=False)

    parser.add_argument('-rtl', '--runtime-limit', help='runtime limit (sec.)', type=float)

    parser.add_argument('-s', '--seed', help='hfs solver seed', type=int, default=0)
    parser.add_argument('-p', '--precision', help='precision of transforming the problem into HFS format', type=int, default=3)

    parser.add_argument('-si', '--show-input', help='print the input file', action='store_true', default=False)

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())


