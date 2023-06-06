from __future__ import annotations

import dataclasses
from typing import ClassVar, Self, TypeAlias
from weakref import WeakKeyDictionary

import numpy as np


Envelope: TypeAlias = tuple[tuple[float, float]]


@dataclasses.dataclass(frozen=True)
class Forest:
    n_total: int
    n_total_positive: int
    envelope: Envelope

    _states: ClassVar[WeakKeyDictionary[Self, list[list[ForestState | None]]]] = WeakKeyDictionary()

    @property
    def n_steps(self) -> int:
        return self.n_total + 1

    @property
    def n_values(self) -> int:
        return self.n_total_positive + 1

    def __post_init__(self):
        self._states[self] = [[None] * self.n_values for _ in range(self.n_steps)]

    @classmethod
    def create(cls, n_total, n_total_positive, envelope=None):
        if envelope is None:
            envelope = tuple((-np.inf, np.inf) for _ in range(n_total + 1))

        return cls(n_total, n_total_positive, envelope)

    def __getitem__(self, index):
        n_observed, n_observed_positive = index
        state = self._states[self][n_observed][n_observed_positive]

        if state is None:
            state = ForestState(self, n_observed, n_observed_positive)
            self._states[self][n_observed][n_observed_positive] = state

        return state

    def get_state_probs(self):
        return [
            [self[step, value].get_prob() for value in range(self.n_values)]
            for step in range(self.n_steps)
        ]


@dataclasses.dataclass(frozen=True)
class ForestState:
    forest: Forest
    n_observed: int
    n_observed_positive: int

    _prob: ClassVar[WeakKeyDictionary[ForestState, float]] = WeakKeyDictionary()

    def __post_init__(self):
        if self.n_observed == 0:
            if self.n_observed_positive == 0:
                self._prob[self] = 1
            else:
                self._prob[self] = 0

    @property
    def n_total(self) -> int:
        return self.forest.n_total

    @property
    def n_total_positive(self) -> int:
        return self.forest.n_total_positive

    @property
    def n_remaining(self) -> int:
        return self.n_total - self.n_observed

    @property
    def n_remaining_positive(self) -> int:
        return self.n_total_positive - self.n_observed_positive

    @property
    def negative_observation_prob(self):
        return (self.n_remaining - self.n_remaining_positive) / self.n_remaining

    @property
    def positive_observation_prob(self):
        return self.n_remaining_positive / self.n_remaining

    @property
    def is_terminal(self) -> bool:
        lower_es_boundary, upper_es_boundary = self.forest.envelope[self.n_observed]
        return not (self.n_remaining > 0 and lower_es_boundary <= self.n_observed_positive <= upper_es_boundary)

    def get_prob(self):
        try:
            return self._prob[self]
        except KeyError:
            prob = self._compute_prob()
            self._prob[self] = prob
            return prob

    def _compute_prob(self):
        prob = 0
        
        upper_parent_state = self.forest[self.n_observed - 1, self.n_observed_positive]

        if not upper_parent_state.is_terminal:
            prob = upper_parent_state.get_prob() * upper_parent_state.negative_observation_prob

        if self.n_observed_positive > 0:
            lower_parent_state = self.forest[self.n_observed - 1, self.n_observed_positive - 1]

            if not lower_parent_state.is_terminal:
                prob += lower_parent_state.get_prob() * lower_parent_state.positive_observation_prob

        return prob


if __name__ == "__main__":
    print(Forest.create(10, 5).get_state_probs())
