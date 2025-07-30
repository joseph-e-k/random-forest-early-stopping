import argparse
import time

import numpy as np

from .optimization import make_optimal_stopping_problem
from .qcp import make_and_time_qcp
from .utils.figures import create_independent_plots_grid, save_drawing
from .utils.logging import configure_logging, get_module_logger
from .utils.misc import get_output_path
from .utils.multiprocessing import parallelize_to_array
from .utils.caching import memoize


_logger = get_module_logger()


def time_os_lp_solution(n, alpha, random_seed=None):
    problem, *_ = make_optimal_stopping_problem(n, alpha)
    solution = problem.solve_with_soplex(random_seed=random_seed)
    return solution.seconds_to_solve


def time_os_qcp_solution(n, alpha, random_seed=None):
    return make_and_time_qcp(n, alpha)


@memoize()
def get_os_solution_times(ensemble_sizes, adrs, n_reps, nonce):
    random_seeds = [hash(nonce) + i for i in range(n_reps)]

    return parallelize_to_array(
        [time_os_lp_solution, time_os_qcp_solution],
        argses_to_combine=(ensemble_sizes, adrs, random_seeds)
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-lower", "-l", type=int, default=1)
    parser.add_argument("--n-upper", "-u", type=int)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--adrs", "--alphas", "-a", type=float, nargs="+", default=(0.0001, 0.001, 0.01, 0.05))
    parser.add_argument("--reps", "-r", type=int, default=3)
    parser.add_argument("--nonce", type=int, default=None)

    parser.add_argument("--min", action="store_const", const=np.min, dest="aggregator")
    parser.add_argument("--max", action="store_const", const=np.max, dest="aggregator")
    parser.add_argument("--mean", action="store_const", const=np.mean, dest="aggregator")
    parser.add_argument("--med", "--median", action="store_const", const=np.median, dest="aggregator")

    return parser.parse_args(args)


def main(args=None):
    configure_logging()

    args = parse_args(args)
    ensemble_sizes = list(range(args.n_lower, args.n_upper + 1, args.n_step))
    nonce = args.nonce or time.time_ns()

    _logger.info(f"{ensemble_sizes=}, adrs={args.adrs}, n_reps={args.reps}, {nonce=}")

    times = get_os_solution_times(
        ensemble_sizes=ensemble_sizes,
        adrs=args.adrs,
        n_reps=args.reps,
        nonce=nonce
    )

    _logger.info(f"{times=}")

    aggregator = args.aggregator or np.min

    agg_times = aggregator(times, axis=2) + 1e-3

    fig, axs = create_independent_plots_grid(len(args.adrs), n_rows=len(args.adrs), figsize=(7, 4))

    for i_adr, adr in enumerate(args.adrs):
        ax = axs[i_adr, 0]

        ax.set_yscale("log")
        ax.scatter(ensemble_sizes, agg_times[0, :, i_adr], marker="o")
        ax.scatter(ensemble_sizes, agg_times[1, :, i_adr], marker="x")
        ax.set_title(f"ADR = {adr}")
        ax.set_xlabel("N")
        ax.set_ylabel("Time (sec)")
    
    output_path = get_output_path(f"timing_combined_nonce_{nonce}")
    save_drawing(fig, output_path)


if __name__ == "__main__":
    main()
