import argparse
import time

import numpy as np

from ste.optimization import make_optimal_stopping_problem
from ste.utils.figures import create_independent_plots_grid, save_drawing
from ste.utils.logging import configure_logging, get_module_logger
from ste.utils.misc import get_output_path
from ste.utils.multiprocessing import parallelize_to_array
from ste.utils.caching import memoize


_logger = get_module_logger()


def time_os_solution(n, alpha):
    problem, *_ = make_optimal_stopping_problem(n, alpha)
    solution = problem.solve_with_soplex()
    return solution.seconds_to_solve


@memoize()
def get_os_solution_times(ensemble_sizes, adrs, n_reps, nonce):
    return parallelize_to_array(
        time_os_solution,
        reps=n_reps,
        argses_to_combine=(ensemble_sizes, adrs)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-lower", "-l", type=int, default=1)
    parser.add_argument("--n-upper", "-u", type=int)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--adrs", "--alphas", "-a", type=float, nargs="+", default=(0.0001, 0.001, 0.01, 0.05))
    parser.add_argument("--reps", "-r", type=int, default=3)
    parser.add_argument("--nonce", type=int, default=None)
    return parser.parse_args()


def main():
    configure_logging()

    args = parse_args()
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

    min_times = times.min(axis=0)

    fig, axs = create_independent_plots_grid(len(args.adrs), n_rows=len(args.adrs), figsize=(4, 4))

    for i_adr, adr in enumerate(args.adrs):
        ax = axs[i_adr, 0]
        ax.plot(ensemble_sizes, min_times[:, i_adr])
        ax.set_title(f"ADR = {adr}")
        ax.set_xlabel("n")
        ax.set_ylabel("Time (sec)")
    
    output_path = get_output_path(f"timing")
    save_drawing(fig, output_path)


if __name__ == "__main__":
    main()
