from __future__ import annotations

import copy
import dataclasses
from typing import TypeAlias

import numpy as np

from .utils import shift_array

Envelope: TypeAlias = list[tuple[int, int]]


@dataclasses.dataclass(frozen=True)
class ForestAnalysis:
    prob_error: float
    expected_runtime: float


@dataclasses.dataclass
class ErrorBudgetMetric:
    allowable_error: float
    remaining_allowable_error: float = dataclasses.field(init=False)
    
    def __post_init__(self):
        self.remaining_allowable_error = self.allowable_error
    
    def __call__(self, forest_with_envelope: ForestWithEnvelope, n_seen: int, n_seen_good: int) -> bool:
        prob_state = forest_with_envelope.state_probabilities[n_seen][n_seen_good]
        if prob_state <= self.remaining_allowable_error:
            self.remaining_allowable_error -= prob_state
            return True
        return False


@dataclasses.dataclass(frozen=True)
class Forest:
    n_total: int
    n_total_positive: int

    @property
    def result(self) -> bool:
        return self.n_total_positive > self.n_total / 2

    @property
    def n_steps(self) -> int:
        return self.n_total + 1

    def get_null_envelope(self) -> Envelope:
        return self.fill_boundary_to_envelope([0], is_upper=False)

    def fill_lower_boundary(self, partial_boundary: list[int]) -> list[int]:
        return partial_boundary + [partial_boundary[-1]] * (self.n_steps - len(partial_boundary))

    def fill_upper_boundary(self, partial_boundary: list[int]) -> list[int]:
        return partial_boundary + [
            partial_boundary[-1] + i + 1
            for i in range(self.n_steps - len(partial_boundary))
        ]

    def fill_boundary(self, partial_boundary: list[int], is_upper: bool) -> list[int]:
        if is_upper:
            return self.fill_upper_boundary(partial_boundary)
        return self.fill_lower_boundary(partial_boundary)

    def fill_boundary_to_envelope(self, partial_boundary: list[int], is_upper: bool, symmetrical: bool = False) -> Envelope:
        boundary = self.fill_boundary(partial_boundary, is_upper)

        if symmetrical:
            other_boundary = self.get_mirror_boundary(boundary)
        else:
            other_boundary = self.fill_boundary([0], not is_upper)

        if is_upper:
            lower_boundary, upper_boundary = other_boundary, boundary
        else:
            lower_boundary, upper_boundary = boundary, other_boundary

        return list(zip(lower_boundary, upper_boundary))

    def get_greedy_lower_boundary(self, metric) -> list[int]:
        if not self.result:
            raise ValueError("get_greedy_lower_boundary() should only be called when the correct result is positive")

        metric = copy.copy(metric)

        boundary = [0]
        envelope = self.fill_boundary_to_envelope(boundary, is_upper=False)
        forest_with_envelope = ForestWithEnvelope(self, envelope)

        for step in range(1, self.n_steps):
            if metric(forest_with_envelope, step, boundary[-1]):
                boundary.append(boundary[-1] + 1)
                envelope = self.fill_boundary_to_envelope(boundary, is_upper=False)
                forest_with_envelope.update_envelope_suffix(envelope[step:])
            else:
                boundary.append(envelope[step][0])

        return boundary

    @staticmethod
    def get_mirror_boundary(boundary: list[int]) -> list[int]:
        return [
            i_step - bound
            for i_step, bound
            in enumerate(boundary)
        ]

    def get_greedy_upper_boundary(self, metric) -> list[int]:
        if self.result:
            raise ValueError("get_greedy_upper_boundary() should only be called when the correct result is negative")

        mirror_forest = Forest(self.n_total, self.n_total - self.n_total_positive)
        mirror_boundary = mirror_forest.get_greedy_lower_boundary(metric)
        return self.get_mirror_boundary(mirror_boundary)


@dataclasses.dataclass
class ForestWithEnvelope:
    forest: Forest
    envelope: Envelope

    state_probabilities: list[np.ndarray] = dataclasses.field(init=False, repr=False, compare=False, hash=False)

    n_total = property(lambda self: self.forest.n_total)
    n_total_positive = property(lambda self: self.forest.n_total_positive)
    result = property(lambda self: self.forest.result)
    n_steps = property(lambda self: self.forest.n_steps)

    def __post_init__(self):
        self._n_steps = self.n_total + 1
        self._n_values = self.n_total + 1

        self._n_bad = self.n_total - self.n_total_positive

        self._n_seen = np.row_stack([np.full(self._n_values, i_step) for i_step in range(self._n_steps)])
        self._n_seen_good = np.column_stack([np.full(self._n_steps, i_value) for i_value in range(self._n_values)])

        self._n_unseen = self.n_total - self._n_seen
        self._n_seen_bad = self._n_seen - self._n_seen_good
        self._n_unseen_good = self.n_total_positive - self._n_seen_good
        self._n_unseen_bad = self._n_bad - self._n_seen_bad

        self._prob_see_good = self._n_unseen_good / self._n_unseen
        self._prob_see_bad = self._n_unseen_bad / self._n_unseen

        self._is_nonterminal = []

        self.state_probabilities = [np.zeros(shape=self.n_total + 1)]
        self.state_probabilities[0][0] = 1

        self._recompute_state_probabilities()

    @classmethod
    def create(cls, n_total, n_total_positive, envelope=None):
        forest = Forest(n_total, n_total_positive)

        if envelope is None:
            envelope = forest.get_null_envelope()

        return cls(forest, envelope)

    def _recompute_state_probabilities(self, starting_index=0):
        self._is_nonterminal[starting_index:] = [
            np.concatenate((
                np.zeros(lower_bound),
                np.ones(upper_bound + 1 - lower_bound),
                np.zeros(self.n_total - upper_bound),
            ))
            for (lower_bound, upper_bound) in self.envelope[starting_index:]
        ]

        del self.state_probabilities[starting_index + 1:]

        for i_step in range(starting_index + 1, self._n_steps):
            nonterminal_prev_prob = self.state_probabilities[-1] * self._is_nonterminal[i_step - 1]
            prob_by_bad_observation = nonterminal_prev_prob * self._prob_see_bad[i_step - 1, :]
            prob_by_good_observation = shift_array(
                nonterminal_prev_prob * self._prob_see_good[i_step - 1, :],
                1,
                fill_value=0
            )
            self.state_probabilities.append(prob_by_bad_observation + prob_by_good_observation)

    @staticmethod
    def get_state_result(n_seen, n_seen_good):
        return n_seen_good > n_seen / 2

    def analyse(self) -> ForestAnalysis:
        prob_error = 0
        expected_runtime = 0

        for n_seen in range(self.n_steps - 1):
            lower_bound, upper_bound = self.envelope[n_seen]
            terminal_values = [lower_bound - 1, upper_bound + 1]

            for n_seen_positive in terminal_values:
                prob_state = self.state_probabilities[n_seen][n_seen_positive]
                expected_runtime += n_seen * prob_state

                if self.get_state_result(n_seen, n_seen_positive) != self.result:
                    prob_error += prob_state

        for n_seen_positive in range(self.n_total_positive + 1):
            expected_runtime += self.n_total * self.state_probabilities[self.n_total][n_seen_positive]

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
        index = len(self.envelope) - len(envelope_suffix)
        self.envelope[index:] = envelope_suffix
        self._recompute_state_probabilities(starting_index=index)
