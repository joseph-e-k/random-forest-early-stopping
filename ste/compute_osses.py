import argparse
import functools
import pickle

import numpy as np

from ste.utils.caching import memoize

from .utils.logging import configure_logging, get_module_logger
from .utils.multiprocessing import parallelize
from .utils.misc import TimerContext, get_output_path
from .empirical_performance import get_minimax_ss, get_minimean_flat_ss, get_minimixed_flat_ss


_logger = get_module_logger()


DEFAULT_AERS = (0.0, 1e-1, 1e-2, 5e-2, 1e-3, 1e-4, 1e-5, 1e-6)
SS_GETTERS = [get_minimax_ss, get_minimean_flat_ss, get_minimixed_flat_ss]


def compute_optimal_stopping_strategies(N_lower, N_upper, N_step, alphas, ss_getters, should_pickle=False):
    """Compute optimal stopping strategies for a range of ensemble sizes, approaches, and AERs.

    Args:
        N_lower (int): Lower bound for the ensemble size.
        N_upper (int): Upper bound for the ensemble size.
        N_step (int): Step size between ensemble sizes.
        alphas (Iterable[float]): List of allowable disagreement rates.
        ss_getters (Sequence[Callable]): List of functions to use to compute OSSes.
        should_pickle (bool, optional): If True, save computed OSSes to .pickle files as they are computed. Defaults to False.
    """
    values_of_N = range(N_lower, N_upper + 1, N_step)
    tasks = parallelize(
        [functools.partial(ss_getter, estimated_smopdis=None) for ss_getter in ss_getters],
        argses_to_combine=(alphas, values_of_N),
        reraise_exceptions=False
    )
    with TimerContext("total"):
        for task in tasks:
            alpha, N = task.args_or_kwargs
            ss_name = task.function.func.__name__
            if ss_name.startswith("get_"):
                ss_name = ss_name[len("get_"):]
            if task.exception is None:
                _logger.info(f"Success ({task.duration:.1f}s): {ss_name}, {N=}, {alpha=}")
                if should_pickle:
                    path = get_output_path(f"../precomputed_stopping_strategies/{ss_name}_N_{N}_alpha_{alpha}", ".pickle")
                    with open(path, "wb") as result_file:
                        pickle.dump(np.asarray(task.result, dtype=float), result_file)
            else:
                _logger.error(f"Error ({task.duration:.1f}s): {N=}, {alpha=}, {task.exception=}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N-lower", "-l", type=int, default=1)
    parser.add_argument("--N-upper", "-u", type=int, required=True)
    parser.add_argument("--N-step", "-s", type=int, default=1)
    parser.add_argument("--alphas", "-a", type=float, nargs="+", default=DEFAULT_AERS)
    parser.add_argument("--pickle", "-p", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    cmd_args = _parse_args()
    compute_optimal_stopping_strategies(
        cmd_args.N_lower,
        cmd_args.N_upper,
        cmd_args.N_step,
        cmd_args.alphas,
        SS_GETTERS,
        cmd_args.pickle
    )
