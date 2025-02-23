import argparse
import numpy as np
import os
import time
import copy
from datetime import timedelta
from scipy.spatial import distance_matrix
from utils import run_all_in_pool
from utils.data_utils import check_extension, load_dataset, save_dataset
from subprocess import check_call, check_output, CalledProcessError
from problems.vrp.vrp_baseline import get_lkh_executable
import torch
from tqdm import tqdm
import re
import math

def fn_destroy(loc, tour, method, tour_taken, n_removed=10,
               randomization_degree=8, new_requests=None):

    if new_requests is None:
        removed = []
    else:
        removed = new_requests

    if len(tour + new_requests) <= n_removed:
        return [], tour + new_requests

    if method == 0:
        ids = list(range(len(tour)))
        remove_ids = np.random.choice(ids, n_removed - len(removed),
                                      replace=False)
        removed = removed + [tour.pop(i) for i in sorted(remove_ids, reverse=True)]

    elif method == 1:
        length_tour = calc_tsp_length(loc, tour_taken + tour + new_requests)
        while len(removed) < n_removed:
            ids = list(range(len(tour)))
            cost_func = lambda x: length_tour - calc_tsp_length(loc,
                                                                tour_taken +
                                                                tour[:x] +
                                                                tour[x+1:],
                                                                subtour=True)
            subtour_costs = list(map(cost_func, ids))
            idx = 1 + int((np.random.random(1) ** randomization_degree)[0] * len(tour))
            remove_id = np.argsort(subtour_costs)[-idx]
            removed.append(tour.pop(remove_id))

    elif method == 2:
        while len(removed) < n_removed:
            ids = list(range(len(tour)))
            reference = tour[np.random.choice(ids, 1).item()]
            dist = np.linalg.norm(loc[reference] - loc[tour], axis=-1)
            idx = int((np.random.random(1) ** randomization_degree)[0] * len(tour))
            remove_id = np.argsort(dist)[idx]
            removed.append(tour.pop(remove_id))

    return tour, removed


def fn_repair(loc, partial_tour, removed, method, tour_taken=None,
              randomization=0.25):

    if tour_taken is None:
        tour_taken = []

    if method < 2:
        randomization = 0.0
    if method % 2 == 0:
        method = "regret"
    else:
        method = "cheapest"

    _, tour = run_insertion(loc, method, tour=partial_tour,
                            tour_taken=tour_taken, randomization=randomization)

    return tour


def destroy_and_repair(loc, tour, destroy, repair,
                       tour_taken=None, new_requests=None):

    if tour_taken is None:
        tour_taken = []

    if new_requests is None:
        new_requests = []

    loc = np.array(loc)
    partial_tour, removed = fn_destroy(loc, tour, destroy, tour_taken,
                                       new_requests=new_requests)
    new_tour = fn_repair(loc, partial_tour, removed, repair,
                         tour_taken=tour_taken)
    return new_tour


def anneal(length_best, length_current, length_new,
           acceptance_probability=0.5, degradation=0.05):
    temp = - (degradation / acceptance_probability) * length_best
    probability = np.exp((length_new - length_current) / temp)

    return (np.random.rand(1) < probability)[0]

def two_opt(loc, tour, tour_taken):
    best = tour
    improved = True

    while improved:
        improved = False

        for i in range(1, len(tour)-2):
            for j in range(i+1, len(tour)):
                if j-i == 1:
                    continue # changes nothing, skip then

                new_tour = tour[:]
                new_tour[i:j] = tour[j-1:i-1:-1] # this is the 2woptSwap

                if calc_tsp_length(loc, tour_taken + new_tour) < calc_tsp_length(loc, tour_taken + best):
                    best = new_tour
                    improved = True

        tour = best

    return best

def ALNS(loc, tour, tour_taken=None, new_requests=None,
         n_no_improvement=500, rho=0.1,
         score_best=1.0, score_better=0.4, score_accepted=0.25,
         freq_update_weights=200, freq_2_opt=200):
    if tour_taken is None:
        tour_taken = []

    if new_requests is None:
        new_requests = []

    weights = np.ones((3, 4))
    probabilities = (weights / weights.sum()).flatten()
    scores = np.zeros((3, 4))
    n_called = np.zeros((3, 4))

    tour_best = copy.copy(tour) + new_requests
    tour_current = copy.copy(tour)
    tour_full = tour_taken + tour_best
    length_best = calc_tsp_length(loc, tour_full)
    length_current = length_best

    iteration = 1
    non_iteration = 1

    choices = [(i, j) for i in range(3) for j in range(4)]

    while non_iteration < n_no_improvement:
        destroy, repair = choices[np.random.choice(12, 1, p=probabilities)[0]]

        tour_new = destroy_and_repair(loc, tour_current, destroy, repair,
                                    tour_taken=tour_taken,
                                    new_requests=new_requests)

        length_new = calc_tsp_length(loc, tour_taken + tour_new)

        new_requests = []

        iteration += 1
        non_iteration += 1

        if length_new < length_best:
            tour_best = copy.copy(tour_new)
            length_best = length_new

            tour_current = copy.copy(tour_new)
            length_current = length_new

            non_iteration = 1
            scores[destroy, repair] += score_best
            n_called[destroy, repair] += 1.0

        elif length_new < length_current:
            tour_current = copy.copy(tour_new)
            length_current = length_new

            scores[destroy, repair] += score_better
            n_called[destroy, repair] += 1.0

        elif anneal(length_best, length_current, length_new):
            tour_current = copy.copy(tour_new)
            length_current = length_new

            scores[destroy, repair] += score_accepted
            n_called[destroy, repair] += 1.0

        if iteration % freq_update_weights == 0:
            weights[scores != 0] = (scores[scores != 0] / n_called[scores != 0]) \
                * rho + (1.0 - rho) * weights[scores != 0]

            probabilities = (weights / weights.sum()).flatten()
            scores = np.zeros((3, 4))
            n_called = np.zeros((3, 4))

        if iteration % freq_2_opt == 0:
            if len(tour_taken + tour_current) < len(loc):
                tour_current = copy.copy(tour_best)
            tour_current = two_opt(loc, tour_current, tour_taken)

    return float(length_best), tour_taken + tour_best


def solve_ALNS(directory, name, data, seeds=None, disable_cache=False):
    if seeds is None:
        seeds = [0]

    try:
        problem_filename = os.path.join(directory, f"{name}.alns.pkl")

        if os.path.isfile(problem_filename) and not disable_cache:
            (cost, tour, duration) = load_dataset(problem_filename)
            return cost, tour, duration

        loc, revealed = data
        revealed = [int(x) for x in revealed]

        t_start = time.time()
        best_cost = np.inf
        best_tour = None
        for seed in seeds:
            np.random.seed(seed)
            _, tour = run_insertion(loc[:revealed[0]], 'cheapest')
            cost, tour = ALNS(loc[:revealed[0]], tour)

            reoptim_ids = [int(idx+1) for idx, i in enumerate(revealed)
                        if idx+1 < len(revealed) and i < revealed[idx+1]
                        ]

            tour_taken = []

            for idx in reoptim_ids:
                tour_taken = tour[:idx]
                tour_not_taken = tour[idx:]
                new_requests = list(np.arange(len(tour), revealed[idx]))
                cost, tour = ALNS(loc[:revealed[idx]], tour_not_taken,
                                tour_taken=tour_taken,
                                new_requests=new_requests)

            if cost < best_cost:
                best_cost = cost
                best_tour = copy.deepcopy(tour)

        duration = time.time() - t_start  # Measure clock time

        assert best_tour is not None

        save_dataset((best_cost, best_tour, duration), problem_filename)

        return best_cost, best_tour, duration
    except Exception as e:
        print("Exception occured")
        print(e)
        return None

def solve_gurobi(directory, name, loc, disable_cache=False, timeout=None, gap=None,
                 dynamic=False, re_opt=False, start=False):
    # Lazy import so we do not need to have gurobi installed to run this script
    from problems.tsp.tsp_gurobi import solve_euclidian_tsp as solve_euclidian_tsp_gurobi
    from problems.tsp.tsp_gurobi import solve_euclidian_tsp_dynamic as solve_euclidian_tsp_gurobi_dynamic

    try:
        problem_filename = os.path.join(directory, "{}.gurobi{}{}-{}-{}{}.pkl".format(
            name, "dynamic" if dynamic else "", "re_opt" if re_opt else None,
            start if start is not None else "no_start",
            "" if timeout is None else "t{}".format(timeout), "" if gap is None else "gap{}".format(gap)))

        if dynamic:
            if start:
                loc, revealed, start = loc
                if type(start[0]) == int:
                    start = list(zip(start, start[1:] + [start[0]]))
            else:
                loc, revealed = loc
                start = None

            revealed = [int(x) for x in revealed]

        if os.path.isfile(problem_filename) and not disable_cache:
            (cost, tour, duration) = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            t_start = time.time()

            if dynamic:
                if re_opt:
                    reoptim_ids = [int(idx+1) for idx, i in enumerate(revealed)
                                if idx+1 < len(revealed) and i < revealed[idx+1]
                                ]
                    tour_taken = []
                    cost, tour = solve_euclidian_tsp_gurobi(loc[:revealed[0]],
                                                            tour_taken,
                                                            threads=1,
                                                            timeout=timeout,
                                                            gap=gap
                                                            )
                    for idx in reoptim_ids:
                        tour_taken = tour[:idx]
                        cost, tour = solve_euclidian_tsp_gurobi(loc[:revealed[idx]],
                                                                tour_taken,
                                                                threads=1,
                                                                timeout=timeout,
                                                                gap=gap
                                                                )
                else:
                    cost, tour = solve_euclidian_tsp_gurobi_dynamic(loc, revealed, start=start, threads=1, timeout=timeout, gap=gap)
            else:
                cost, tour = solve_euclidian_tsp_gurobi(loc, [], threads=1, timeout=timeout, gap=gap)

            duration = time.time() - t_start  # Measure clock time

            save_dataset((cost, tour, duration), problem_filename)

        if dynamic:
            total_cost = calc_tsp_length(loc[:revealed[-1]], tour)
        else:
            total_cost = calc_tsp_length(loc, tour)

        assert abs(total_cost - cost) <= 1e-5, "Cost is incorrect"
        return total_cost, tour, duration

    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we can retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None


def solve_concorde_log(executable, directory, name, loc, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.tsp".format(name))
    tour_filename = os.path.join(directory, "{}.tour".format(name))
    output_filename = os.path.join(directory, "{}.concorde.pkl".format(name))
    log_filename = os.path.join(directory, "{}.log".format(name))

    # if True:
    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_tsplib(problem_filename, loc, name=name)

            with open(log_filename, 'w') as f:
                start = time.time()
                try:
                    # Concorde is weird, will leave traces of solution in current directory so call from target dir
                    check_call([executable, '-s', '1234', '-x', '-o',
                                os.path.abspath(tour_filename), os.path.abspath(problem_filename)],
                               stdout=f, stderr=f, cwd=directory)
                except CalledProcessError as e:
                    # Somehow Concorde returns 255
                    assert e.returncode == 255
                duration = time.time() - start

            tour = read_concorde_tour(tour_filename)
            save_dataset((tour, duration), output_filename)

        return calc_tsp_length(loc, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def solve_lkh_log(executable, directory, name, loc, runs=1, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.lkh{}.vrp".format(name, runs))
    tour_filename = os.path.join(directory, "{}.lkh{}.tour".format(name, runs))
    output_filename = os.path.join(directory, "{}.lkh{}.pkl".format(name, runs))
    param_filename = os.path.join(directory, "{}.lkh{}.par".format(name, runs))
    log_filename = os.path.join(directory, "{}.lkh{}.log".format(name, runs))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_tsplib(problem_filename, loc, name=name)

            params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": tour_filename, "RUNS": runs, "SEED": 1234}
            write_lkh_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, param_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_tsplib(tour_filename)
            save_dataset((tour, duration), output_filename)

        return calc_tsp_length(loc, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def write_tsplib(filename, loc, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "TSP"),
                ("DIMENSION", len(loc)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # tsplib does not take floats
            for i, (x, y) in enumerate(loc)
        ]))
        f.write("\n")
        f.write("EOF\n")


def read_concorde_tour(filename):
    with open(filename, 'r') as f:
        n = None
        tour = []
        for line in f:
            if n is None:
                n = int(line)
            else:
                tour.extend([int(node) for node in line.rstrip().split(" ")])
    assert len(tour) == n, "Unexpected tour length"
    return tour


def read_tsplib(filename):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    return tour.tolist()


def calc_tsp_length(loc, tour, subtour=False):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert len(tour) == len(loc) or subtour
    sorted_locs = np.array(loc)[np.concatenate((tour, [tour[0]]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def _calc_insert_cost(D, prv, nxt, ins, tour_taken=None, randomization=0.0):
    """
    Calculates insertion costs of inserting ins between prv and nxt
    :param D: distance matrix
    :param prv: node before inserted node, can be vector
    :param nxt: node after inserted node, can be vector
    :param ins: node to insert
    :return:
    """
    if tour_taken is None:
        tour_taken = []

    if len(tour_taken) > 0:
        prv = np.insert(prv, 0, tour_taken)
        nxt = np.roll(prv, -1)
        result = D[prv, ins] \
              + D[ins, nxt] \
              - D[prv, nxt] \
              + randomization * np.random.uniform(-1, 1) * D.max()
        return result[len(tour_taken)-1:]

    return (
        D[prv, ins]
        + D[ins, nxt]
        - D[prv, nxt]
        + randomization * np.random.uniform(-1, 1) * D.max()
    )


def run_insertion(loc, method, tour=None, tour_taken=None, randomization=0.0):
    n = len(loc)
    D = distance_matrix(loc, loc)

    if tour is None:
        tour = []

    if tour_taken is None:
        tour_taken = []

    mask = np.zeros(n, dtype=bool)
    mask[tour] = True
    mask[tour_taken] = True
    tour = tour  # np.empty((0, ), dtype=int)
    for i in range(n - len(tour) - len(tour_taken)):
        feas = mask == 0
        feas_ind = np.flatnonzero(mask == 0)
        if method == 'random':
            # Order of instance is random so do in order for deterministic results
            a = feas_ind[i]
        elif method == 'nearest':
            if len(tour) == 0:
                a = feas_ind[0]  # order does not matter so first is random
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmin()] # node nearest to any in tour
        elif method == 'farthest':
            if len(tour) == 0:
                a = D.max(1).argmax()  # Node with farthest distance to any other node
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmax()]  # node which has closest node in tour farthest

        elif method == 'cheapest':
            if len(tour) == 0:
                a = feas_ind[0]
            elif len(tour) == 1:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmax()]  # node which has closest node in tour farthest
            else:
                best_cost = math.inf
                best_ind = -1
                for idx in feas_ind:
                    insert_cost = np.min(
                        _calc_insert_cost(
                            D,
                            tour,
                            np.roll(tour, -1),
                            idx,
                            tour_taken=tour_taken,
                            randomization=randomization
                        )
                    )
                    if insert_cost < best_cost:
                        best_ind = idx
                        best_cost = insert_cost
                a = best_ind
        elif method == 'regret':
            if len(tour) == 0:
                a = feas_ind[0]
            elif len(tour) == 1:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmax()]  # node which has closest node in tour farthest
            else:
                best_cost = -math.inf
                best_ind = -1
                for idx in feas_ind:
                    regret_costs = _calc_insert_cost(
                            D,
                            tour,
                            np.roll(tour, -1),
                            idx,
                            tour_taken=tour_taken,
                            randomization=randomization
                        )
                    regret_costs.sort()
                    if len(tour) > 2:
                        regret_cost = regret_costs[2] + \
                            regret_costs[1] - \
                            2 * regret_costs[0]
                    else:
                        regret_cost = regret_costs[1] - regret_costs[0]

                    if regret_cost > best_cost:
                        best_ind = idx
                        best_cost = regret_cost

                a = best_ind

        mask[a] = True

        if len(tour) == 0:
            tour = [a]
        else:
            # Find index with least insert cost
            ind_insert = np.argmin(
                _calc_insert_cost(
                    D,
                    tour,
                    np.roll(tour, -1),
                    a,
                    tour_taken=tour_taken
                )
            )
            tour.insert(ind_insert + 1, a)

    cost = D[tour, np.roll(tour, -1)].sum()
    return cost, tour


def solve_insertion(directory, name, loc, method='random'):
    start = time.time()
    cost, tour = run_insertion(loc, method)
    duration = time.time() - start
    return cost, tour, duration


def calc_batch_pdist(dataset):
    diff = (dataset[:, :, None, :] - dataset[:, None, :, :])
    return torch.matmul(diff[:, :, :, None, :], diff[:, :, :, :, None]).squeeze(-1).squeeze(-1).sqrt()


def nearest_neighbour(dataset, start='first'):
    dist = calc_batch_pdist(dataset)

    batch_size, graph_size, _ = dataset.size()

    total_dist = dataset.new(batch_size).zero_()

    if not isinstance(start, torch.Tensor):
        if start == 'random':
            start = dataset.new().long().new(batch_size).zero_().random_(0, graph_size)
        elif start == 'first':
            start = dataset.new().long().new(batch_size).zero_()
        elif start == 'center':
            _, start = dist.mean(2).min(1)  # Minimum total distance to others
        else:
            assert False, "Unknown start: {}".format(start)

    current = start
    dist_to_startnode = torch.gather(dist, 2, current.view(-1, 1, 1).expand(batch_size, graph_size, 1)).squeeze(2)
    tour = [current]

    for i in range(graph_size - 1):
        # Mark out current node as option
        dist.scatter_(2, current.view(-1, 1, 1).expand(batch_size, graph_size, 1), np.inf)
        nn_dist = torch.gather(dist, 1, current.view(-1, 1, 1).expand(batch_size, 1, graph_size)).squeeze(1)

        min_nn_dist, current = nn_dist.min(1)
        total_dist += min_nn_dist
        tour.append(current)

    total_dist += torch.gather(dist_to_startnode, 1, current.view(-1, 1)).squeeze(1)

    return total_dist, torch.stack(tour, dim=1)


def solve_all_nn(dataset_path, eval_batch_size=1024, no_cuda=False, dataset_n=None, progress_bar_mininterval=0.1):
    import torch
    from torch.utils.data import DataLoader
    from problems import TSP
    from utils import move_to

    dataloader = DataLoader(
        TSP.make_dataset(filename=dataset_path, num_samples=dataset_n if dataset_n is not None else 1000000),
        batch_size=eval_batch_size
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu")
    results = []
    for batch in tqdm(dataloader, mininterval=progress_bar_mininterval):
        start = time.time()
        batch = move_to(batch, device)

        lengths, tours = nearest_neighbour(batch)
        lengths_check, _ = TSP.get_costs(batch, tours)

        assert (torch.abs(lengths - lengths_check.data) < 1e-5).all()

        duration = time.time() - start
        results.extend(
            [(cost.item(), np.trim_zeros(pi.cpu().numpy(), 'b'), duration) for cost, pi in zip(lengths, tours)])

    return results, eval_batch_size


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("method",
                        help="Name of the method to evaluate, 'nn',  'alns', 'gurobi', 'concorde' or '(nearest|random|farthest)_insertion'")
    parser.add_argument("--dynamic", action='store_true', help="Specifies if the problem is dynamic")
    parser.add_argument("--re_opt", action='store_true', help="Specifies if the problem should be reoptimized")
    parser.add_argument('--start', default=None, help="Name of the start directory")
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA (only for Tsiligirides)')
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--max_calc_batch_size', type=int, default=1000, help='Size for subbatches')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, help="Number of instances to process")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")

    opts = parser.parse_args()

    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:

        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"

        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])

        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "tsp", dataset_basename)
            os.makedirs(results_dir, exist_ok=True)

            out_file = os.path.join(results_dir, "{}{}{}-{}{}-{}{}".format(
                dataset_basename,
                "offs{}".format(opts.offset) if opts.offset is not None else "",
                "n{}".format(opts.n) if opts.n is not None else "",
                "dynamic" if opts.dynamic else "",
                "re_opt" if opts.re_opt else "",
                opts.method, ext
            ))
        else:
            out_file = opts.o

        assert opts.f or not os.path.isfile(
            out_file), "File already exists! Try running with -f option to overwrite."

        match = re.match(r'^([a-z_]+)(\d*)$', opts.method)
        assert match
        method = match[1]
        runs = 1 if match[2] == '' else int(match[2])

        if method == "nn":
            assert opts.offset is None, "Offset not supported for nearest neighbor"

            eval_batch_size = opts.max_calc_batch_size

            results, parallelism = solve_all_nn(
                dataset_path, eval_batch_size, opts.no_cuda, opts.n,
                opts.progress_bar_mininterval
            )
        elif method in ("gurobi", "gurobigap", "gurobit", "concorde", "lkh", "alns") or method[-9:] == 'insertion':

            target_dir = os.path.join(results_dir, "{}-{}-{}".format(
                dataset_basename,
                opts.method,
                "dynamic" if opts.dynamic else "static"
            ))
            assert opts.f or not os.path.isdir(target_dir), \
                "Target dir already exists! Try running with -f option to overwrite."

            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)

            if opts.dynamic:
                instances, revealed = load_dataset(dataset_path)

                if opts.start is not None:
                    assert os.path.isfile(check_extension(opts.start)), "Start file does not exist!"
                    start = [x[1] for x in load_dataset(opts.start)[0]]
                    dataset = [(x, ) for x in zip(instances, revealed, start)]
                else:
                    dataset = [(x, ) for x in zip(instances, revealed)]

            else:
                # TSP contains single loc array rather than tuple
                dataset = [(instance, ) for instance in load_dataset(dataset_path)]

            if method == "alns":
                assert opts.dynamic, "ALNS can only be used for dynamic problems"
                use_multiprocessing = True  # We run one thread per instance

                def run_func(args):
                    return solve_ALNS(*args, seeds=range(32))

            elif method == "concorde":
                use_multiprocessing = False
                executable = os.path.abspath(os.path.join('problems', 'tsp', 'concorde', 'concorde', 'TSP', 'concorde'))

                def run_func(args):
                    return solve_concorde_log(executable, *args, disable_cache=opts.disable_cache)

            elif method == "lkh":
                use_multiprocessing = False
                executable = get_lkh_executable()

                def run_func(args):
                    return solve_lkh_log(executable, *args, runs=runs, disable_cache=opts.disable_cache)

            elif method[:6] == "gurobi":
                use_multiprocessing = True  # We run one thread per instance

                def run_func(args):
                    return solve_gurobi(*args, disable_cache=opts.disable_cache,
                                        timeout=runs if method[6:] == "t" else None,
                                        gap=float(runs) if method[6:] == "gap" else None,
                                        dynamic=opts.dynamic, re_opt=opts.re_opt,
                                        start=True if opts.start is not None else False)
            else:
                assert method[-9:] == "insertion"
                use_multiprocessing = True

                def run_func(args):
                    return solve_insertion(*args, opts.method.split("_")[0])

            results, parallelism = run_all_in_pool(
                run_func,
                target_dir, dataset, opts, use_multiprocessing=use_multiprocessing
            )

        else:
            assert False, "Unknown method: {}".format(opts.method)

        costs, tours, durations = zip(*results)  # Not really costs since they should be negative
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(
            np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        save_dataset((results, parallelism), out_file)
