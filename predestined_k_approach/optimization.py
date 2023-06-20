from __future__ import annotations

import copy
import dataclasses

from .Forest import Forest
from .envelopes import fill_boundary_to_envelope, get_mirror_boundary
from .ForestWithEnvelope import ForestWithEnvelope


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


def get_greedy_lower_boundary(forest: Forest, metric) -> list[int]:
    if not forest.result:
        raise ValueError("get_greedy_lower_boundary() should only be called when the correct result is positive")

    metric = copy.copy(metric)

    boundary = [0]
    envelope = fill_boundary_to_envelope(forest.n_total, boundary, is_upper=False)
    forest_with_envelope = ForestWithEnvelope(forest, envelope)

    for step in range(1, forest.n_steps):
        if metric(forest_with_envelope, step, boundary[-1]):
            boundary.append(boundary[-1] + 1)
            envelope = fill_boundary_to_envelope(forest.n_total, boundary, is_upper=False)
            forest_with_envelope.update_envelope_suffix(envelope[step:])
        else:
            boundary.append(envelope[step][0])

    return boundary


def get_greedy_upper_boundary(forest, metric) -> list[int]:
    if forest.result:
        raise ValueError("get_greedy_upper_boundary() should only be called when the correct result is negative")

    mirror_forest = Forest(forest.n_total, forest.n_total - forest.n_total_positive)
    mirror_boundary = get_greedy_lower_boundary(mirror_forest, metric)
    return get_mirror_boundary(mirror_boundary)
