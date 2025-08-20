import argparse
import time

import numpy as np

from .optimization import make_optimal_stopping_problem
from .qcp import make_and_time_qcp
from .utils.figures import MARKERS, create_independent_plots_grid, save_drawing
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
def get_os_solution_times(ensemble_sizes, adrs, n_reps, nonce, problem_kinds):
    random_seeds = [hash(nonce) + i for i in range(n_reps)]

    timers = [
        {"lp": time_os_lp_solution, "qp": time_os_qcp_solution}[pk]
        for pk in problem_kinds
    ]

    return parallelize_to_array(
        timers,
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

    parser.add_argument('--lp', dest='problem_kinds', action='append_const', const='lp', default=[])
    parser.add_argument('--qp', dest='problem_kinds', action='append_const', const='qp')

    parser.add_argument("--log", action="store_true")

    args = parser.parse_args(args)

    if not args.problem_kinds:
        parser.error("at least one of --lp and --qp must be specified")

    return args


def aggregate_and_plot_times(ensemble_sizes, times, adrs, problem_kinds, log, aggregator=None, fig_and_axs=None):
    aggregator = aggregator or np.min

    agg_times = aggregator(times, axis=2) + 1e-3

    if fig_and_axs is None:
        fig, axs = create_independent_plots_grid(len(adrs), n_rows=len(adrs), figsize=(7, 4))
    else:
        fig, axs = fig_and_axs
        if axs.shape != (len(adrs), 1):
            raise ValueError("axs must have shape (len(adrs), 1)")

    for i_adr, adr in enumerate(adrs):
        ax = axs[i_adr, 0]

        if log:
            ax.set_yscale("log")
        
        for (i_pk, pk), marker in zip(enumerate(problem_kinds), MARKERS):
            label = {
                "qp": "QCQP",
                "lp": "LP"
            }[pk]
            ax.scatter(ensemble_sizes, agg_times[i_pk, :, i_adr], marker=marker, label=label)
        
        if len(problem_kinds) > 1:
            ax.legend()
        ax.set_xlabel("N")
        ax.set_ylabel("Time (sec)")

    return fig, axs


def main(args=None):
    configure_logging()

    args = parse_args(args)
    ensemble_sizes = list(range(args.n_lower, args.n_upper + 1, args.n_step))
    nonce = args.nonce or time.time_ns()
    problem_kinds = tuple(args.problem_kinds)

    _logger.info(f"{ensemble_sizes=}, adrs={args.adrs}, n_reps={args.reps}, {nonce=}, {problem_kinds=}")

    times = get_os_solution_times(
        ensemble_sizes=ensemble_sizes,
        adrs=args.adrs,
        n_reps=args.reps,
        nonce=nonce,
        problem_kinds=problem_kinds
    )

    _logger.info(f"{times=}")

    fig, axs = aggregate_and_plot_times(
        ensemble_sizes,
        times,
        args.adrs,
        args.problem_kinds,
        args.log,
        args.aggregator
    )
    
    output_path = get_output_path(f"timing_combined_nonce_{nonce}")
    save_drawing(fig, output_path)


if __name__ == "__main__":
    main()
