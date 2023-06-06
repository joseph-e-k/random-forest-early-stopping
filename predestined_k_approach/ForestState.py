from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Forest import ForestWithEnvelope


@dataclasses.dataclass
class ForestState:
    forest: ForestWithEnvelope
    n_seen: int
    n_seen_positive: int

    _prob: float | None = dataclasses.field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self):
        if self.n_seen == 0:
            if self.n_seen_positive == 0:
                self._prob = 1
            else:
                self._prob = 0
        else:
            self._prob = None

    @property
    def n_total(self) -> int:
        return self.forest.n_total

    @property
    def n_total_positive(self) -> int:
        return self.forest.n_total_positive

    @property
    def n_remaining(self) -> int:
        return self.n_total - self.n_seen

    @property
    def n_remaining_positive(self) -> int:
        return self.n_total_positive - self.n_seen_positive

    @property
    def prob_see_negative(self):
        return (self.n_remaining - self.n_remaining_positive) / self.n_remaining

    @property
    def prob_see_positive(self):
        return self.n_remaining_positive / self.n_remaining

    @property
    def is_terminal(self) -> bool:
        lower_es_boundary, upper_es_boundary = self.forest.envelope[self.n_seen]
        return not (self.n_remaining > 0 and lower_es_boundary <= self.n_seen_positive <= upper_es_boundary)

    @property
    def result(self) -> bool:
        return self.n_seen_positive > self.n_seen / 2

    def get_prob(self):
        if self._prob is None:
            self._prob = self._compute_prob()
        return self._prob

    def _compute_prob(self):
        prob = 0

        upper_parent_state = self.forest[self.n_seen - 1, self.n_seen_positive]

        if not upper_parent_state.is_terminal:
            prob = upper_parent_state.get_prob() * upper_parent_state.prob_see_negative

        if self.n_seen_positive > 0:
            lower_parent_state = self.forest[self.n_seen - 1, self.n_seen_positive - 1]

            if not lower_parent_state.is_terminal:
                prob += lower_parent_state.get_prob() * lower_parent_state.prob_see_positive

        return prob
