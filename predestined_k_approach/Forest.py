from __future__ import annotations

import dataclasses
from itertools import zip_longest
from typing import TypeAlias

import numpy as np

from .ForestState import ForestState, ImpossibleForestState

Envelope: TypeAlias = list[tuple[int, int]]


@dataclasses.dataclass(frozen=True)
class ForestAnalysis:
    prob_error: float
    expected_runtime: float


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

    def get_null_envelope(self):
        return ((0, self.n_total_positive),) * self.n_steps

    def partial_lower_boundary_to_envelope(self, boundary):
        return list(zip_longest(boundary, [self.n_total_positive] * self.n_steps, fillvalue=boundary[-1]))

    def partial_upper_boundary_to_envelope(self, boundary):
        naively_extrapolated_boundary = boundary + [
            boundary[-1] + i + 1
            for i in range(self.n_steps - len(boundary))
        ]
        return list(zip([0] * self.n_steps, naively_extrapolated_boundary))

    def get_greedy_lower_boundary(self, allowable_error) -> list[int]:
        if not self.result:
            raise ValueError("get_greedy_lower_boundary() should only be called when the correct result is positive")

        remaining_allowable_error = allowable_error

        boundary = [0]
        envelope = self.partial_lower_boundary_to_envelope(boundary)
        forest_with_envelope = ForestWithEnvelope(self, envelope)

        for step in range(1, self.n_steps):
            state = forest_with_envelope[step, boundary[-1]]

            if state.get_prob() <= remaining_allowable_error:
                boundary.append(boundary[-1] + 1)
                envelope = self.partial_lower_boundary_to_envelope(boundary)
                forest_with_envelope.update_envelope_suffix(envelope[step:])

                remaining_allowable_error -= state.get_prob()

            else:
                boundary.append(envelope[step][0])

        return boundary

    def get_greedy_upper_boundary(self, allowable_error) -> list[int]:
        # TODO: Reduce code duplication between this function and get_greedy_lower_boundary
        if self.result:
            raise ValueError("get_greedy_upper_boundary() should only be called when the correct result is negative")

        remaining_allowable_error = allowable_error

        boundary = [0]
        envelope = self.partial_upper_boundary_to_envelope(boundary)
        forest_with_envelope = ForestWithEnvelope(self, envelope)

        for step in range(1, self.n_steps):
            state = forest_with_envelope[step, boundary[-1] + 1]

            if state.get_prob() <= remaining_allowable_error:
                boundary.append(boundary[-1])
                envelope = self.partial_upper_boundary_to_envelope(boundary)
                forest_with_envelope.update_envelope_suffix(envelope[step:])

                remaining_allowable_error -= state.get_prob()

            else:
                boundary.append(envelope[step][1])

        return boundary


@dataclasses.dataclass
class ForestWithEnvelope:
    forest: Forest
    envelope: Envelope

    _states: list[list[ForestState | None]] = dataclasses.field(init=False, repr=False, compare=False, hash=False)

    n_total = property(lambda self: self.forest.n_total)
    n_total_positive = property(lambda self: self.forest.n_total_positive)
    result = property(lambda self: self.forest.result)
    n_steps = property(lambda self: self.forest.n_steps)

    def _initialize_states(self, starting_index=0):
        self._states[starting_index:] = [[None] * self.n_steps for _ in range(starting_index, self.n_steps)]

    def __post_init__(self):
        self._states = []
        self._initialize_states()

    @classmethod
    def create(cls, n_total, n_total_positive, envelope=None):
        forest = Forest(n_total, n_total_positive)

        if envelope is None:
            envelope = forest.get_null_envelope()

        return cls(forest, envelope)

    def __getitem__(self, index):
        n_seen, n_seen_positive = index

        if 0 <= n_seen_positive <= n_seen <= self.n_total and n_seen_positive <= self.n_total_positive:
            state = self._states[n_seen][n_seen_positive]

            if state is None:
                state = ForestState(self, n_seen, n_seen_positive)
                self._states[n_seen][n_seen_positive] = state

            return state

        return ImpossibleForestState(self, n_seen, n_seen_positive)

    def analyse(self):
        prob_error = 0
        expected_runtime = 0

        for n_seen in range(self.n_steps):
            lower_bound, upper_bound = self.envelope[n_seen]
            terminal_states = [self[n_seen, lower_bound-1], self[n_seen, upper_bound+1]]

            for state in terminal_states:
                expected_runtime += n_seen * state.get_prob()

                if state.result != self.result:
                    prob_error += state.get_prob()

        return ForestAnalysis(
            prob_error=prob_error,
            expected_runtime=expected_runtime
        )

    def update_envelope_suffix(self, envelope_suffix: Envelope):
        index = len(self.envelope) - len(envelope_suffix)
        self.envelope[index:] = envelope_suffix
        self._initialize_states(index)

    def get_state_probs(self):
        return np.array([
            [self[n_seen, n_seen_positive].get_prob() for n_seen_positive in range(self.n_total_positive + 1)]
            for n_seen in range(self.n_steps)
        ])
