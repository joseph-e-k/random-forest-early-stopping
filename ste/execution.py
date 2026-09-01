import dataclasses
import functools
import threading

import numpy as np
from joblib import effective_n_jobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.parallel import Parallel, delayed

from ste.theoretical_performance import estimate_smopdis
from ste.utils.misc import TimerContext
from ste.utils.multiprocessing import parallelize_to_array

from .utils.caching import memoize
from .utils.data import split_dataset


class SequentialTestingStopped(Exception):
    pass


# Copied verbatim from sklearn.ensemble._base
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _accumulate_vote(predict, X, out, lock, ss, stopping_dice):
    probs = predict(X, check_input=False)
    with lock:
        out[0][0][probs.argmax(axis=1)] += 1
        i = out[0].sum()
        j = out[0][0][1]
        if stopping_dice[i] < ss[i, j]:
            raise SequentialTestingStopped(i, j)


@dataclasses.dataclass(frozen=True)
class SequentiallyTestedRandomForestClassifier:
    rf: RandomForestClassifier
    ss: np.ndarray  # shape = (n + 1, n + 1), where n = rf.n_estimators

    def __post_init__(self):
        n = len(self.rf.estimators_)
        assert self.ss.shape == (n + 1, n + 1)
        assert (0 <= self.ss).all()
        assert (self.ss <= 1).all()

    def predict(self, X):
        check_is_fitted(self.rf)
        # Check data
        X = self.rf._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.rf.n_estimators, self.rf.n_jobs)

        # avoid storing the output of every estimator by summing them here
        votes = [
            np.zeros((X.shape[0], j), dtype=int)
            for j in np.atleast_1d(self.rf.n_classes_)
        ]
        lock = threading.Lock()
        stopping_dice = np.random.uniform(size=self.rf.n_estimators+1)
        estimators = list(self.rf.estimators_)
        np.random.shuffle(estimators)
        try:
            Parallel(n_jobs=n_jobs, verbose=self.rf.verbose, require="sharedmem")(
                delayed(_accumulate_vote)(e.predict_proba, X, votes, lock, self.ss, stopping_dice)
                for e in estimators
            )
        except SequentialTestingStopped as stop:
            i, j = stop.args
            return int(j > i / 2)
        else:
            raise Exception("This state should be unreachable")


def get_wall_clock_times_once(dataset, data_partition_ratios, n_trees, adr, stopping_strategy_getter):
    training_data, calibration_data, evaluation_data = split_dataset(dataset, data_partition_ratios)

    forest = RandomForestClassifier(n_trees, n_jobs=1)
    forest.fit(*training_data)

    smopdis_estimate = estimate_smopdis(forest, calibration_data)
    ss = stopping_strategy_getter(adr, n_trees, smopdis_estimate).astype(float)

    stf = SequentiallyTestedRandomForestClassifier(forest, ss)

    X, _ = evaluation_data

    with TimerContext(verbose=False) as stf_timer:
        for i in range(len(X)):
            x = X[i:i+1]
            stf.predict(x)

    with TimerContext(verbose=False) as full_forest_timer:
        for i in range(len(X)):
            x = X[i:i+1]
            forest.predict(x)

    return np.array([stf_timer.elapsed_time, full_forest_timer.elapsed_time])


@memoize()
def get_wall_clock_times(datasets, data_partition_ratios, n_trees, n_forests, adr, stopping_strategy_getter):
    times = parallelize_to_array(
        functools.partial(
            get_wall_clock_times_once,
            data_partition_ratios=data_partition_ratios,
            n_trees=n_trees,
            adr=adr,
            stopping_strategy_getter=stopping_strategy_getter
        ),
        argses_to_combine=[datasets],
        reps=n_forests
    )

    return times
