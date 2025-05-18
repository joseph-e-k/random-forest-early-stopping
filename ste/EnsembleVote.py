from __future__ import annotations

import dataclasses
import random
import warnings

import numpy as np
from scipy.special import logsumexp

from .utils.misc import shift_array


@dataclasses.dataclass(frozen=True)
class EnsembleVote:
    """Represents a potential vote balance of an ensemble of binary classifiers
    
    Attributes:
        n_total (int): Total number of base classifiers.
        n_yes (int): Number of "yes" votes.
    """
    n_total: int
    n_yes: int

    @property
    def result(self) -> bool:
        return self.n_yes > self.n_total / 2

    @property
    def n_steps(self) -> int:
        return self.n_total + 1

    @property
    def n_values(self) -> int:
        return self.n_yes + 1


@dataclasses.dataclass
class EnsembleVoteWithStoppingStrategy:
    """An EnsembleVote along with a stopping strategy.
    Together, these fully specify the state transition graph of the voting process.
    
    Attributes:
        ensemble_vote (EnsembleVote): The total number of votes in the ensemble.
        stopping_strategy (np.ndarray): The stopping strategy, a 2D array of shape (n_total + 1, n_yes + 1).
                                        At initialization, a larger matrix may be passed; it will be cropped to exclude unreachable states.
    """
    ensemble_vote: EnsembleVote
    stopping_strategy: np.ndarray

    n_total = property(lambda self: self.ensemble_vote.n_total)
    n_yes = property(lambda self: self.ensemble_vote.n_yes)
    result = property(lambda self: self.ensemble_vote.result)
    n_steps = property(lambda self: self.ensemble_vote.n_steps)
    prob_see_yes = property(lambda self: np.exp(self._log_prob_see_yes))
    prob_see_no = property(lambda self: np.exp(self._log_prob_see_no))

    def __post_init__(self):
        self.stopping_strategy = self.stopping_strategy[:self.n_total + 1, :self.n_yes + 1]

        self._n_steps = self.n_total + 1
        self._n_values = self.n_yes + 1

        self._n_no = self.n_total - self.n_yes

        self._n_seen = np.vstack([np.full(self._n_values, i_step) for i_step in range(self._n_steps)])
        self._n_seen_yes = np.column_stack([np.full(self._n_steps, i_value) for i_value in range(self._n_values)])

        self._n_unseen = self.n_total - self._n_seen
        self._n_seen_no = self._n_seen - self._n_seen_yes
        self._n_unseen_yes = np.maximum(self.n_yes - self._n_seen_yes, 0)
        self._n_unseen_no = np.maximum(self._n_no - self._n_seen_no, 0)

        with warnings.catch_warnings(category=RuntimeWarning, action="ignore"):
            self._log_prob_see_yes = np.log(self._n_unseen_yes / self._n_unseen)
            self._log_prob_see_no = np.log(self._n_unseen_no / self._n_unseen)

        self._compute_state_probabilities()

    def _compute_state_probabilities(self):
        with warnings.catch_warnings(category=RuntimeWarning, action="ignore"):
            self._log_prob_stop_if_reach = np.log(np.asarray(self.stopping_strategy, dtype=float))

        self._log_prob_reach = np.full(shape=(self._n_steps, self._n_values), fill_value=-np.inf)
        self._log_prob_reach[0, 0] = np.log(1)

        for i_step in range(1, self._n_steps):
            prev_log_prob_stop_if_reach = self._log_prob_stop_if_reach[i_step - 1, :]
            prev_log_prob_continue_if_reach = logsumexp(
                a=np.vstack([np.zeros_like(prev_log_prob_stop_if_reach), prev_log_prob_stop_if_reach]),
                b=np.vstack([np.ones_like(prev_log_prob_stop_if_reach), np.full_like(prev_log_prob_stop_if_reach, -1)]),
                axis=0
            )
            prev_log_prob_reach_and_continue = self._log_prob_reach[i_step - 1, :] + prev_log_prob_continue_if_reach
            log_prob_reach_by_no = prev_log_prob_reach_and_continue + self._log_prob_see_no[i_step - 1, :]
            log_prob_reach_by_yes = shift_array(
                prev_log_prob_reach_and_continue + self._log_prob_see_yes[i_step - 1, :],
                1,
                fill_value=-np.inf
            )
            np.logaddexp(
                log_prob_reach_by_no,
                log_prob_reach_by_yes,
                out=self._log_prob_reach[i_step, :]
            )

            if np.any(np.isnan(self._log_prob_reach[i_step, :])):
                raise ValueError("NaN appeared in log-probability computation")

            # Floating-point errors may occasionally cause us to compute "log-probabilities" that are slightly greater
            # than 0. We just clamp these, as allowing them to persist will NaN-poison our entire program.
            np.clip(self._log_prob_reach[i_step, :], None, 0, out=self._log_prob_reach[i_step, :])

    def get_log_state_probability(self, n_seen, n_seen_yes):
        if any([
            n_seen < 0,
            n_seen_yes < 0,
            n_seen > self.n_total,
            n_seen_yes > self.n_yes,
            n_seen_yes > n_seen
        ]):
            return -np.inf

        return self._log_prob_reach[n_seen, n_seen_yes]

    @staticmethod
    def get_state_result(n_seen, n_seen_yes):
        return n_seen_yes > n_seen / 2

    def analyse(self) -> EnsembleVoteAnalysis:
        log_prob_reach_and_stop = self._log_prob_reach + self._log_prob_stop_if_reach

        is_state_disagreement = (self._n_seen_yes > self._n_seen / 2) != self.result

        log_prob_disagreement = logsumexp(log_prob_reach_and_stop, b=is_state_disagreement)
        log_expected_runtime = logsumexp(log_prob_reach_and_stop, b=self._n_seen)

        return EnsembleVoteAnalysis(
            prob_disagreement=np.exp(log_prob_disagreement),
            expected_runtime=np.exp(log_expected_runtime)
        )


@dataclasses.dataclass(frozen=True)
class EnsembleVoteAnalysis:
    prob_disagreement: float
    expected_runtime: float
