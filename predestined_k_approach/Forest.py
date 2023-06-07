from __future__ import annotations

import dataclasses
from itertools import zip_longest
from typing import TypeAlias

from .ForestState import ForestState, ImpossibleForestState

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

    def get_null_envelope(self):
        return ((0, self.n_total_positive),) * self.n_steps

    def get_optimal_lower_boundary(self, allowable_error, verbose=False) -> list[int]:
        if not self.result:
            raise ValueError("get_optimal_lower_boundary() should only be called when the correct result is positive")

        remaining_allowable_error = allowable_error

        boundary = [0]

        for step in range(1, self.n_steps):
            envelope = tuple(zip_longest(boundary, [self.n_total_positive] * self.n_steps, fillvalue=boundary[-1]))
            forest_with_envelope = ForestWithEnvelope(self, envelope)

            state = forest_with_envelope[step, boundary[-1]]

            if state.get_prob() <= remaining_allowable_error:
                if verbose:
                    print(f"Stop if <={state.n_seen_positive} / {state.n_seen} are positive: p = {state.get_prob()}")

                boundary.append(boundary[-1] + 1)
                remaining_allowable_error -= state.get_prob()
            else:
                boundary.append(boundary[-1])

        return boundary

    def get_optimal_upper_boundary(self, allowable_error, verbose=False) -> list[int]:
        # TODO: Reduce code duplication between this function and get_optimal_lower_boundary
        if self.result:
            raise ValueError("get_optimal_upper_boundary() should only be called when the correct result is negative")

        remaining_allowable_error = allowable_error
        boundary = [0]

        for step in range(1, self.n_steps):
            naively_extrapolated_boundary = boundary + [
                boundary[-1] + i + 1
                for i in range(self.n_steps - len(boundary))
            ]
            envelope = tuple(zip([0] * self.n_steps, naively_extrapolated_boundary))
            forest_with_envelope = ForestWithEnvelope(self, envelope)

            state = forest_with_envelope[step, boundary[-1] + 1]

            if state.get_prob() <= remaining_allowable_error:
                if verbose:
                    print(f"Stop if >={state.n_seen_positive} / {state.n_seen} are positive: p = {state.get_prob()}")
                boundary.append(boundary[-1])
                remaining_allowable_error -= state.get_prob()
            else:
                boundary.append(boundary[-1] + 1)

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

    def __post_init__(self):
        self._states = [[None] * self.n_steps for _ in range(self.n_steps)]

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
        probs_states = []
        prob_error = 0
        expected_runtime = 0

        for n_seen in range(self.n_steps):
            probs_states.append([])

            for n_seen_positive in range(self.n_steps):
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
