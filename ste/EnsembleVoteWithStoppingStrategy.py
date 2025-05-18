from __future__ import annotations

import dataclasses
import random
import warnings

import numpy as np
from scipy.special import logsumexp

from .EnsembleVote import EnsembleVote
from .utils.misc import shift_array


class EnsembleVoteExecutionError(Exception):
    pass


@dataclasses.dataclass
class EnsembleVoteWithStoppingStrategy:
    ensemble_vote: EnsembleVote
    stopping_strategy: np.ndarray

    n_total = property(lambda self: self.ensemble_vote.n_total)
    n_yes = property(lambda self: self.ensemble_vote.n_yes)
    result = property(lambda self: self.ensemble_vote.result)
    n_steps = property(lambda self: self.ensemble_vote.n_steps)
    prob_see_yes = property(lambda self: np.exp(self._log_prob_see_yes))
    prob_see_no = property(lambda self: np.exp(self._log_prob_see_no))

    # TODO: Consistent naming style: log_prob_thing vs thing_log_prob vs log_thing_prob
    def _get_log_ss(self):
        with warnings.catch_warnings(category=RuntimeWarning, action="ignore"):
            return np.log(np.asarray(self.stopping_strategy, dtype=float))
        
    def get_prob_stop(self):
        return self.stopping_strategy

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

        self._log_prob_reach = np.empty(shape=(self._n_steps, self._n_values))
        with warnings.catch_warnings(category=RuntimeWarning, action="ignore"):
            self._log_prob_reach[0, :] = np.log(np.zeros(shape=self._n_values))

        self._compute_state_probabilities()

    def _compute_state_probabilities(self):
        self._log_prob_reach[0, 0] = np.log(1)
        self._log_prob_stop_if_reach = self._get_log_ss()

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

    def get_lowest_finite_log_probability(self):
        return np.min(self._log_prob_reach[np.isfinite(self._log_prob_reach)])

    @staticmethod
    def get_state_result(n_seen, n_seen_yes):
        return n_seen_yes > n_seen / 2

    def analyse(self) -> EnsembleVoteAnalysis:
        log_prob_reach_state_and_stop = self._log_prob_reach + self._log_prob_stop_if_reach

        is_state_disagreement = (self._n_seen_yes > self._n_seen / 2) != self.result

        log_prob_disagreement = logsumexp(log_prob_reach_state_and_stop, b=is_state_disagreement)
        log_expected_runtime = logsumexp(log_prob_reach_state_and_stop, b=self._n_seen)

        return EnsembleVoteAnalysis(
            prob_disagreement=np.exp(log_prob_disagreement),
            expected_runtime=np.exp(log_expected_runtime)
        )

    def simulate(self, rng=None) -> tuple[int, bool]:
        if rng is None:
            rng = random.Random()

        votes = np.zeros(self.n_total, dtype=int)
        which_yes = rng.sample(range(self.n_total), self.n_yes)
        votes[which_yes] = 1

        if rng.random() < np.exp(self._log_prob_stop_if_reach[0, 0]):
            return 0, False

        n_yes_seen = 0

        for i_vote, vote in enumerate(votes):
            n_seen = i_vote + 1
            n_yes_seen += vote

            if rng.random() < np.exp(self._log_prob_stop_if_reach[n_seen, n_yes_seen]):
                return n_seen, (n_yes_seen > n_seen / 2)

        raise EnsembleVoteExecutionError("Fell off the end of an ensemble vote while executing. This should be impossible.", self)

    def get_pi_and_pi_bar(self) -> tuple[np.ndarray, np.ndarray]:
        theta = np.exp(self._get_log_ss())
        theta_bar = 1 - theta
        pi = np.empty_like(theta)
        pi_bar = np.empty_like(theta)

        for n_seen_yes in range(theta.shape[1]):
            pi[0, n_seen_yes] = theta[0, n_seen_yes]
            pi_bar[0, n_seen_yes] = theta_bar[0, n_seen_yes]

        for n_seen in range(1, theta.shape[0]):
            for n_seen_yes in range(theta.shape[1]):
                combinatoric_factor = (n_seen_yes / n_seen * pi_bar[n_seen - 1, n_seen_yes - 1] + (n_seen - n_seen_yes) / n_seen * pi_bar[n_seen - 1, n_seen_yes])
                pi[n_seen, n_seen_yes] = theta[n_seen, n_seen_yes] * combinatoric_factor
                pi_bar[n_seen, n_seen_yes] = theta_bar[n_seen, n_seen_yes] * combinatoric_factor

        return pi, pi_bar


@dataclasses.dataclass(frozen=True)
class EnsembleVoteAnalysis:
    prob_disagreement: float
    expected_runtime: float
