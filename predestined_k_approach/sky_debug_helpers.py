import argparse
import itertools
import multiprocessing as mp
import traceback
import warnings
from typing import Callable

from predestined_k_approach.Forest import Forest
from predestined_k_approach.optimization import *
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from predestined_k_approach.utils import TimerContext


DEFAULT_AERS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0)


def make_sky_from_truncated_theta(truncated_theta):
    n = truncated_theta.shape[0] - 1
    theta = np.ones((n + 1, n + 1))
    theta[:, :truncated_theta.shape[1]] = truncated_theta

    p = np.zeros_like(theta)

    theta_bar = 1 - theta

    p[0, 0] = 1
    for i in range(n):
        p[i + 1, 0] = p[i, 0] * theta_bar[i, 0]

        for j in range(i + 1):
            p[i + 1, j + 1] = (
                ((i - j) / (i + 1)) * p[i, j + 1] * theta_bar[i, j + 1]
                + ((j + 1) / (i + 1)) * p[i, j] * theta_bar[i, j]
            )

    return Sky(p, p * theta, p * theta_bar)


def get_expected_runtimes(n_total, aer=10**-6):
    n_positive_low = n_total // 2
    n_positive_high = n_positive_low + 1

    low_forest = Forest(n_total, n_positive_low)
    high_forest = Forest(n_total, n_positive_high)

    low_fwe = ForestWithEnvelope.create_greedy(n_total, n_positive_low, aer)
    high_fwe = ForestWithEnvelope(forest=high_forest, envelope=low_fwe.envelope)

    fwss_sky, theoretical_fwss_time = make_and_solve_optimal_stopping_problem(n_total, aer)
    optimal_stopping_strategy = make_theta_from_sky(fwss_sky)

    low_fwss = ForestWithGivenStoppingStrategy(low_forest, optimal_stopping_strategy)
    high_fwss = ForestWithGivenStoppingStrategy(high_forest, optimal_stopping_strategy)

    low_fwss_time = low_fwss.analyse().expected_runtime
    high_fwss_time = high_fwss.analyse().expected_runtime
    low_fwe_time = low_fwe.analyse().expected_runtime
    high_fwe_time = high_fwe.analyse().expected_runtime

    return theoretical_fwss_time, low_fwss_time, high_fwss_time, low_fwe_time, high_fwe_time

@dataclasses.dataclass(frozen=True)
class Worker:
    function: Callable

    def __call__(self, args):
        try:
            with warnings.catch_warnings(action="ignore"):
                result = self.function(*args)
        except Exception as e:
            traceback.print_exc()
            return args, False, e
        return args, True, result


def n_cycles(iterable, n):
    """Return the sequence elements n times."""
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))


def search_for_impossibilities(n_processes, low_n_total, high_n_total, repetitions=1, aers=DEFAULT_AERS):
    n_totals = range(low_n_total, high_n_total + 1, 2)
    with TimerContext("total"):
        with mp.Pool(n_processes) as pool:
            imap = pool.imap(
                Worker(get_expected_runtimes),
                n_cycles(itertools.product(n_totals, aers), repetitions)
            )
            for (args, success, result) in imap:
                n_total, aer = args

                if success:
                    theoretical_fwss_time, low_fwss_time, high_fwss_time, low_fwe_time, high_fwe_time = result
                    info = (f"{n_total=}, {aer=}, {theoretical_fwss_time=}, {low_fwss_time=}, {high_fwss_time=},"
                            f"{low_fwe_time=}, {high_fwe_time=}")
                    if low_fwss_time > low_fwe_time and high_fwss_time > high_fwe_time:
                        print(f"impossible: {info}")
                    else:
                        print(f"possible: {info}")

                else:
                    print(f"error: {n_total=}, {aer=}, error={result!r}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-lower-bound", "-l", type=int)
    parser.add_argument("--n-upper-bound", "-u", type=int)
    parser.add_argument("--worker-process-count", "-w", type=int)
    parser.add_argument("--repetition-count", "-r", type=int, default=1)
    parser.add_argument("--alphas", "-a", type=float, nargs="+", default=DEFAULT_AERS)
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = _parse_args()
    search_for_impossibilities(
        cmd_args.worker_process_count,
        cmd_args.n_lower_bound,
        cmd_args.n_upper_bound,
        cmd_args.repetition_count,
        cmd_args.alphas
    )
