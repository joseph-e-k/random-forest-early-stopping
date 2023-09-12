from __future__ import annotations

import dataclasses
import random

import numpy as np

from .Forest import Forest
from .envelopes import Envelope, get_null_envelope
from .utils import shift_array


@dataclasses.dataclass
class ForestWithEnvelope:
    forest: Forest
    envelope: Envelope

    _state_probabilities: list[np.ndarray] = dataclasses.field(init=False, repr=False, compare=False, hash=False)

    n_total = property(lambda self: self.forest.n_total)
    n_total_positive = property(lambda self: self.forest.n_total_positive)
    result = property(lambda self: self.forest.result)
    n_steps = property(lambda self: self.forest.n_steps)

    def __post_init__(self):
        self._n_steps = self.n_total + 1
        self._n_values = self.n_total + 1

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

        self._log_state_probabilities = [np.log(np.zeros(shape=self.n_total + 1))]
        self._log_state_probabilities[0][0] = np.log(1)

        self._recompute_state_probabilities()

    @classmethod
    def create(cls, n_total, n_total_positive, envelope=None):
        forest = Forest(n_total, n_total_positive)

        if envelope is None:
            envelope = get_null_envelope(n_total)

        return cls(forest, envelope)

    def _invalidate_state_probabilities(self, start_index=0):
        del self._log_state_probabilities[start_index + 1:]

    def _recompute_state_probabilities(self, end_index=None):
        start_index = len(self._log_state_probabilities)

        if end_index is None:
            end_index = self._n_steps

        for i_step in range(start_index, end_index):
            prev_lower_bound, prev_upper_bound = self.envelope[i_step-1]
            log_prev_is_nonterminal = np.array(
                [np.log(int(prev_lower_bound <= v <= prev_upper_bound)) for v in range(self._n_values)]
            )
            nonterminal_prev_log_prob = self._log_state_probabilities[-1] + log_prev_is_nonterminal
            log_prob_by_bad_observation = nonterminal_prev_log_prob + self._log_prob_see_bad[i_step - 1, :]
            log_prob_by_good_observation = shift_array(
                nonterminal_prev_log_prob + self._log_prob_see_good[i_step - 1, :],
                1,
                fill_value=-np.inf
            )
            next_log_state_probabilities = np.logaddexp(log_prob_by_bad_observation, log_prob_by_good_observation)
            self._log_state_probabilities.append(next_log_state_probabilities)

    def get_state_probability(self, n_seen, n_seen_good):
        self._recompute_state_probabilities(n_seen + 1)
        return np.exp(self._log_state_probabilities[n_seen][n_seen_good])

    @staticmethod
    def get_state_result(n_seen, n_seen_good):
        return n_seen_good > n_seen / 2

    def analyse(self) -> ForestAnalysis:
        self._recompute_state_probabilities()

        prob_error = 0
        expected_runtime = 0

        for n_seen in range(self.n_steps - 1):
            lower_bound, upper_bound = self.envelope[n_seen]
            terminal_values = []
            if lower_bound > 0:
                terminal_values.append(lower_bound - 1)
            if upper_bound < self.n_steps:
                terminal_values.append(upper_bound + 1)

            for n_seen_positive in terminal_values:
                prob_state = np.exp(self._log_state_probabilities[n_seen][n_seen_positive])
                expected_runtime += n_seen * prob_state

                if self.get_state_result(n_seen, n_seen_positive) != self.result:
                    prob_error += prob_state

        for n_seen_positive in range(self.n_total_positive + 1):
            expected_runtime += self.n_total * np.exp(self._log_state_probabilities[self.n_total][n_seen_positive])

        return ForestAnalysis(
            prob_error=prob_error,
            expected_runtime=expected_runtime
        )

    def get_score(self, allowable_error: float) -> float:
        analysis = self.analyse()
        error_weight = (1 / allowable_error) - 1
        expected_points_per_run = (1 - analysis.prob_error) - error_weight * analysis.prob_error
        return expected_points_per_run / analysis.expected_runtime

    def update_envelope_suffix(self, envelope_suffix: Envelope):
        update_start_index = len(self.envelope) - len(envelope_suffix)
        self.envelope[update_start_index:] = envelope_suffix
        self._invalidate_state_probabilities(start_index=update_start_index)

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
