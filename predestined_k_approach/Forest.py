from __future__ import annotations

import dataclasses
from typing import TypeAlias, ClassVar, Self
from weakref import WeakKeyDictionary

import numpy as np

from .ForestState import ForestState


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
        n_seen, n_seen_positive = index
        state = self._states[self][n_seen][n_seen_positive]

        if state is None:
            state = ForestState(self, n_seen, n_seen_positive)
            self._states[self][n_seen][n_seen_positive] = state

        return state

    def get_state_probs(self):
        return [
            [self[step, value].get_prob() for value in range(self.n_values)]
            for step in range(self.n_steps)
        ]
