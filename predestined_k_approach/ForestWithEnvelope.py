from __future__ import annotations

import dataclasses
import math
import random

import numpy as np
from scipy.special import logsumexp

from .Forest import Forest
from .envelopes import Envelope, get_null_envelope, add_increment_to_envelope
from .utils import shift_array


class ForestExecutionError(Exception):
    pass


@dataclasses.dataclass
class ForestWithStoppingStrategy:
    forest: Forest

    n_total = property(lambda self: self.forest.n_total)
    n_total_positive = property(lambda self: self.forest.n_total_positive)
    result = property(lambda self: self.forest.result)
    n_steps = property(lambda self: self.forest.n_steps)

    # TODO: Consistent naming style: log_prob_thing vs thing_log_prob vs log_thing_prob
    def _get_log_prob_stop(self):
        raise NotImplementedError()

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

        self._log_prob_stop = self._get_log_prob_stop()

        self._i_last_valid_state_probabilities = 0

    def _recompute_state_probabilities(self, end_index=None):
        start_index = self._i_last_valid_state_probabilities + 1

        if end_index is None:
            end_index = self._n_steps - 1

        for i_step in range(start_index, end_index + 1):
            prev_log_prob_stop = self._log_prob_stop[i_step - 1, :]
            prev_log_prob_no_stop = logsumexp(
                a=np.row_stack([np.zeros_like(prev_log_prob_stop), prev_log_prob_stop]),
                b=np.row_stack([np.ones_like(prev_log_prob_stop), np.full_like(prev_log_prob_stop, -1)]),
                axis=0
            )
            prev_log_prob_arrive_and_continue = self._log_state_probabilities[i_step - 1, :] + prev_log_prob_no_stop
            log_prob_by_bad_observation = prev_log_prob_arrive_and_continue + self._log_prob_see_bad[i_step - 1, :]
            log_prob_by_good_observation = shift_array(
                prev_log_prob_arrive_and_continue + self._log_prob_see_good[i_step - 1, :],
                1,
                fill_value=-np.inf
            )
            np.logaddexp(
                log_prob_by_bad_observation,
                log_prob_by_good_observation,
                out=self._log_state_probabilities[i_step, :]
            )

        self._i_last_valid_state_probabilities = end_index

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

        log_prob_reach_state_and_stop = self._log_state_probabilities + self._log_prob_stop

        is_state_error = (self._n_seen_good > self._n_seen / 2) != self.result

        log_prob_error = logsumexp(log_prob_reach_state_and_stop, b=is_state_error)
        log_expected_runtime = logsumexp(log_prob_reach_state_and_stop, b=self._n_seen)

        return ForestAnalysis(
            prob_error=np.exp(log_prob_error),
            expected_runtime=np.exp(log_expected_runtime)
        )

    def simulate(self, rng=None) -> tuple[int, bool]:
        if rng is None:
            rng = random.Random()

        trees = np.zeros(self.n_total, dtype=int)
        which_positive = rng.sample(range(self.n_total), self.n_total_positive)
        trees[which_positive] = 1

        if rng.random() < np.exp(self._log_prob_stop[0, 0]):
            return 0, False

        n_positive_seen = 0

        for i_tree, tree_conclusion in enumerate(trees):
            n_seen = i_tree + 1
            n_positive_seen += tree_conclusion

            if rng.random() < np.exp(self._log_prob_stop[n_seen, n_positive_seen]):
                return n_seen, (n_positive_seen > n_seen / 2)

        raise ForestExecutionError("Fell off the end of a forest while executing. This should be impossible.", self)


@dataclasses.dataclass
class ForestWithEnvelope(ForestWithStoppingStrategy):
    envelope: Envelope

    def _get_log_prob_stop(self):
        prob_stop = np.empty_like(self._log_state_probabilities)

        for i_step in range(self.n_steps - 1):
            prob_stop[i_step, :] = 1 - self._get_mask_for_bounds(self._n_values, *self.envelope[i_step])

        prob_stop[self.n_steps - 1, :] = 1

        return np.log(prob_stop)

    @classmethod
    def create(cls, n_total, n_total_positive, envelope=None):
        if envelope is None:
            envelope = get_null_envelope(n_total)

        forest = Forest(n_total, n_total_positive)
        return cls(forest, envelope)

    @classmethod
    def create_greedy(cls, n_total, n_positive, allowable_error):
        log_allowable_error = np.log(allowable_error)

        envelope = get_null_envelope(n_total)
        forest_with_envelope = cls.create(n_total, math.ceil((n_total + 1) / 2), envelope)

        for i_step in range(1, n_total + 1):
            log_prob_state = forest_with_envelope.get_log_state_probability(i_step, envelope[i_step][0])
            if log_prob_state <= log_allowable_error:
                log_allowable_error = logsumexp([log_allowable_error, log_prob_state], b=[1, -1])
                forest_with_envelope.add_increment_to_envelope(i_step)

        return ForestWithEnvelope.create(n_total, n_positive, envelope)

    def _invalidate_state_probabilities(self, start_index=0):
        self._i_last_valid_state_probabilities = start_index

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

    def add_increment_to_envelope(self, increment_index: int):
        add_increment_to_envelope(self.envelope, increment_index)
        self._invalidate_state_probabilities(start_index=increment_index)


@dataclasses.dataclass(frozen=True)
class ForestAnalysis:
    prob_error: float
    expected_runtime: float
