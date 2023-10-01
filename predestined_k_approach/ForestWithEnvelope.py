from __future__ import annotations

import dataclasses
import random

import numpy as np
from scipy.special import logsumexp

from .Forest import Forest
from .envelopes import Envelope, get_null_envelope, add_increment_to_envelope
from .utils import shift_array


@dataclasses.dataclass
class ForestWithEnvelope:
    forest: Forest
    envelope: Envelope

    n_total = property(lambda self: self.forest.n_total)
    n_total_positive = property(lambda self: self.forest.n_total_positive)
    result = property(lambda self: self.forest.result)
    n_steps = property(lambda self: self.forest.n_steps)

    def __post_init__(self):
        self._n_steps = self.n_total + 1
        self._n_values = self.n_total_positive + 1

        self._n_good = self.n_total_positive
        self._n_bad = self.n_total - self._n_good

        self._n_seen = np.row_stack([np.full(self._n_values, i_step) for i_step in range(self._n_steps)])
        self._n_seen_good = np.column_stack([np.full(self._n_steps, i_value) for i_value in range(self._n_values)])

        self._n_unseen = self.n_total - self._n_seen
        self._n_seen_bad = self._n_seen - self._n_seen_good
        self._n_unseen_good = np.maximum(self._n_good - self._n_seen_good, 0)
        self._n_unseen_bad = np.maximum(self._n_bad - self._n_seen_bad, 0)

        self._log_prob_see_good = np.log(self._n_unseen_good / self._n_unseen)
        self._log_prob_see_bad = np.log(self._n_unseen_bad / self._n_unseen)

        self._log_state_probabilities = np.empty(shape=(self._n_steps, self._n_values))
        self._log_state_probabilities[0, :] = np.log(np.zeros(shape=self._n_values))
        self._log_state_probabilities[0, 0] = np.log(1)
        self._i_last_valid_state_probabilities = 0

    @classmethod
    def create(cls, n_total, n_total_positive, envelope=None):
        if envelope is None:
            envelope = get_null_envelope(n_total)

        forest = Forest(n_total, n_total_positive)
        return cls(forest, envelope)

    def _invalidate_state_probabilities(self, start_index=0):
        self._i_last_valid_state_probabilities = start_index

    def _recompute_state_probabilities(self, end_index=None):
        start_index = self._i_last_valid_state_probabilities + 1

        if end_index is None:
            end_index = self._n_steps - 1

        for i_step in range(start_index, end_index + 1):
            prev_is_nonterminal = self._get_mask_for_bounds(self._n_values, *self.envelope[i_step - 1])
            log_prev_is_nonterminal = np.log(prev_is_nonterminal)
            nonterminal_prev_log_prob = self._log_state_probabilities[i_step - 1, :] + log_prev_is_nonterminal
            log_prob_by_bad_observation = nonterminal_prev_log_prob + self._log_prob_see_bad[i_step - 1, :]
            log_prob_by_good_observation = shift_array(
                nonterminal_prev_log_prob + self._log_prob_see_good[i_step - 1, :],
                1,
                fill_value=-np.inf
            )
            np.logaddexp(
                log_prob_by_bad_observation,
                log_prob_by_good_observation,
                out=self._log_state_probabilities[i_step, :]
            )

        self._i_last_valid_state_probabilities = end_index

    @staticmethod
    def _get_mask_for_bounds(size, lower, upper):
        effective_lower = np.clip(lower, 0, size)
        lower_mask = np.concatenate([
            np.zeros(effective_lower),
            np.ones(size - effective_lower)
        ])

        effective_upper = np.clip(upper, -1, size - 1)
        upper_mask = np.concatenate([
            np.ones(effective_upper + 1),
            np.zeros(size - effective_upper - 1)
        ])

        return np.logical_and(lower_mask, upper_mask)

    def get_log_state_probability(self, n_seen, n_seen_good):
        if any([
            n_seen < 0,
            n_seen_good < 0,
            n_seen > self.n_total,
            n_seen_good > self.n_total_positive,
            n_seen_good > n_seen
        ]):
            return -np.inf

        self._recompute_state_probabilities(n_seen)
        return self._log_state_probabilities[n_seen, n_seen_good]

    def get_lowest_finite_log_probability(self):
        self._recompute_state_probabilities()
        return np.min(self._log_state_probabilities[np.isfinite(self._log_state_probabilities)])

    @staticmethod
    def get_state_result(n_seen, n_seen_good):
        return n_seen_good > n_seen / 2

    def analyse(self) -> ForestAnalysis:
        self._recompute_state_probabilities()

        state_weights_runtime = np.zeros_like(self._log_state_probabilities)
        state_weights_error = np.zeros_like(self._log_state_probabilities)

        for n_seen in range(self.n_steps - 1):
            lower_bound, upper_bound = self.envelope[n_seen]
            terminal_values = [lower_bound - 1, upper_bound + 1]

            for n_seen_positive in terminal_values:
                if 0 <= n_seen_positive <= self.n_total_positive:
                    state_weights_runtime[n_seen, n_seen_positive] = n_seen

                    if self.get_state_result(n_seen, n_seen_positive) != self.result:
                        state_weights_error[n_seen, n_seen_positive] = 1

        state_weights_runtime[self.n_total, :] = self.n_total

        log_prob_error = logsumexp(self._log_state_probabilities, b=state_weights_error)
        log_expected_runtime = logsumexp(self._log_state_probabilities, b=state_weights_runtime)

        return ForestAnalysis(
            prob_error=np.exp(log_prob_error),
            expected_runtime=np.exp(log_expected_runtime)
        )

    def get_score(self, allowable_error: float) -> float:
        analysis = self.analyse()
        error_weight = (1 / allowable_error) - 1
        expected_points_per_run = (1 - analysis.prob_error) - error_weight * analysis.prob_error
        return expected_points_per_run / analysis.expected_runtime

    def add_increment_to_envelope(self, increment_index: int):
        add_increment_to_envelope(self.envelope, increment_index)
        self._invalidate_state_probabilities(start_index=increment_index)

    def simulate(self) -> tuple[int, bool]:
        trees = np.zeros(self.n_total)
        which_positive = random.sample(range(self.n_total), self.n_total_positive)
        trees[which_positive] = 1

        n_positive_seen = 0

        for n_seen, next_tree in enumerate(trees):
            lower_bound, upper_bound = self.envelope[n_seen]
            if n_positive_seen < lower_bound:
                return n_seen, False
            if n_positive_seen > upper_bound:
                return n_seen, True

            n_positive_seen += next_tree

        return self.n_total, (n_positive_seen > self.n_total / 2)


@dataclasses.dataclass(frozen=True)
class ForestAnalysis:
    prob_error: float
    expected_runtime: float
