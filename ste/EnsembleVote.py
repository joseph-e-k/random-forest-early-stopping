from __future__ import annotations

import numpy as np

import dataclasses

from scipy import stats


@dataclasses.dataclass(frozen=True)
class EnsembleVote:
    n_total: int
    n_yes: int

    @property
    def result(self) -> bool:
        return self.n_yes > self.n_total / 2

    @property
    def n_steps(self) -> int:
        return self.n_total + 1

    @property
    def n_values(self) -> int:
        return self.n_yes + 1

    def get_abstract_probabilities(self):
        m = np.empty((self.n_steps, self.n_values))

        for n_seen in range(self.n_steps):
            for n_seen_yes in range(self.n_values):
                m[n_seen, n_seen_yes] = stats.hypergeom(self.n_total, self.n_yes, n_seen).pmf(n_seen_yes)

        return m
