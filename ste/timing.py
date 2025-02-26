import argparse
import time

import numpy as np

from ste.optimization import make_optimal_stopping_problem
from ste.qcp import make_and_time_qcp
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
def get_os_solution_times(ensemble_sizes, adrs, n_reps, nonce, use_qcp=False):
    return parallelize_to_array(
        make_and_time_qcp if use_qcp else time_os_solution,
        reps=n_reps,
        argses_to_combine=(ensemble_sizes, adrs),
        dummy=use_qcp
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-lower", "-l", type=int, default=1)
    parser.add_argument("--n-upper", "-u", type=int)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--adrs", "--alphas", "-a", type=float, nargs="+", default=(0.0001, 0.001, 0.01, 0.05))
    parser.add_argument("--reps", "-r", type=int, default=3)
    parser.add_argument("--nonce", type=int, default=None)
    parser.add_argument("--qcp", "-q", action="store_true")

    parser.add_argument("--min", action="store_const", const=np.min, dest="aggregator")
    parser.add_argument("--max", action="store_const", const=np.max, dest="aggregator")
    parser.add_argument("--mean", action="store_const", const=np.mean, dest="aggregator")
    parser.add_argument("--med", "--median", action="store_const", const=np.median, dest="aggregator")

    return parser.parse_args()


def main():
    configure_logging()

    args = parse_args()
    ensemble_sizes = list(range(args.n_lower, args.n_upper + 1, args.n_step))
    nonce = args.nonce or time.time_ns()

    _logger.info(f"{ensemble_sizes=}, adrs={args.adrs}, n_reps={args.reps}, {nonce=}, use_qcp={args.qcp}")

    times = get_os_solution_times(
        ensemble_sizes=ensemble_sizes,
        adrs=args.adrs,
        n_reps=args.reps,
        nonce=nonce,
        use_qcp=args.qcp
    )

    _logger.info(f"{times=}")

    aggregator = args.aggregator or np.min

    agg_times = aggregator(times, axis=0)

    fig, axs = create_independent_plots_grid(len(args.adrs), n_rows=len(args.adrs), figsize=(7, 4))

    for i_adr, adr in enumerate(args.adrs):
        ax = axs[i_adr, 0]

        title = f"ADR = {adr}"
        if args.qcp:
            title += " (QCP)"
            ax.set_yscale("log")

        ax.scatter(ensemble_sizes, agg_times[:, i_adr], marker="o")
        ax.set_title(title)
        ax.set_xlabel("N")
        ax.set_ylabel("Time (sec)")
    
    output_path = get_output_path(f"timing_{'q_' if args.qcp else ''}nonce_{nonce}")
    save_drawing(fig, output_path)


if __name__ == "__main__":
    main()
