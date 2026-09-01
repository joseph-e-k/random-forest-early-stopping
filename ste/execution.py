import dataclasses
import threading

import numpy as np
from joblib import effective_n_jobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.parallel import Parallel, delayed


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


def probs_to_votes(probs):
    probs = np.asarray(probs)
    votes = np.zeros_like(probs, dtype=int)
    votes[np.arange(probs.shape[0]), probs.argmax(axis=1)] = 1
    return votes


def _accumulate_vote(predict, X, out, lock, ss, stopping_dice):
    prediction = predict(X, check_input=False)
    votes = probs_to_votes(prediction)
    with lock:
        out[0] += votes
        breakpoint()
        i = out[0].sum()
        j = out[0][0][1]
        breakpoint()
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
        try:
            Parallel(n_jobs=n_jobs, verbose=self.rf.verbose, require="sharedmem")(
                delayed(_accumulate_vote)(e.predict_proba, X, votes, lock, self.ss, stopping_dice)
                for e in self.rf.estimators_
            )
        except SequentialTestingStopped as stop:
            i, j = stop.args
            return int(j > i / 2)
        else:
            raise Exception("This state should be unreachable")
