from __future__ import annotations

import dataclasses
from typing import TypeAlias, ClassVar, Self
from weakref import WeakKeyDictionary

import numpy as np

from .ForestState import ForestState


Envelope: TypeAlias = tuple[tuple[float, float], ...]


@dataclasses.dataclass(frozen=True)
class ForestAnalysis:
    probs_states: tuple[tuple[float, ...], ...]
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

    @property
    def n_values(self) -> int:
        return self.n_total_positive + 1

    def get_null_envelope(self):
        return ((-np.inf, np.inf),) * self.n_steps


@dataclasses.dataclass(frozen=True)
class ForestWithEnvelope:
    forest: Forest
    envelope: Envelope

    _states: ClassVar[WeakKeyDictionary[Self, list[list[ForestState | None]]]] = WeakKeyDictionary()

    n_total = property(lambda self: self.forest.n_total)
    n_total_positive = property(lambda self: self.forest.n_total_positive)
    result = property(lambda self: self.forest.result)
    n_steps = property(lambda self: self.forest.n_steps)
    n_values = property(lambda self: self.forest.n_values)

    def __post_init__(self):
        self._states[self] = [[None] * self.n_values for _ in range(self.n_steps)]

    @classmethod
    def create(cls, n_total, n_total_positive, envelope=None):
        forest = Forest(n_total, n_total_positive)

        if envelope is None:
            envelope = forest.get_null_envelope()

        return cls(forest, envelope)

    def __getitem__(self, index):
        n_seen, n_seen_positive = index
        state = self._states[self][n_seen][n_seen_positive]

        if state is None:
            state = ForestState(self, n_seen, n_seen_positive)
            self._states[self][n_seen][n_seen_positive] = state

        return state

    def analyse(self):
        probs_states = []
        prob_error = 0
        expected_runtime = 0

        for n_seen in range(self.n_steps):
            probs_states.append([])

            for n_seen_positive in range(self.n_values):
                state = self[n_seen, n_seen_positive]

                if state.is_terminal:
                    expected_runtime += n_seen * state.get_prob()

                    if state.result != self.result:
                        prob_error += state.get_prob()

                probs_states[-1].append(state.get_prob())

        return ForestAnalysis(
            probs_states=tuple(tuple(probs) for probs in probs_states),
            prob_error=prob_error,
            expected_runtime=expected_runtime
        )
