from __future__ import annotations

import math

from .Forest import Forest
from .envelopes import fill_boundary_to_envelope, fill_envelope, Envelope
from .ForestWithEnvelope import ForestWithEnvelope
from .utils import powerset


def get_envelope_greedy_eb(n_total, allowable_error) -> Envelope:
    increments = []

    n_good = math.ceil((n_total + 1) / 2)

    envelope = selected_indices_to_symmetric_envelope(n_total, increments)
    forest_with_envelope = ForestWithEnvelope.create(n_total, n_good, envelope)

    for step in range(1, n_total + 1):
        prob_state = forest_with_envelope.state_probabilities[step][envelope[step][0]]
        if prob_state <= allowable_error:
            allowable_error -= prob_state
            increments.append(step)
            envelope = selected_indices_to_symmetric_envelope(n_total, increments)
            forest_with_envelope.update_envelope_suffix(envelope[step:])

    return envelope


def get_score_envelope(n_total, allowable_error):
    forest_with_envelope = ForestWithEnvelope.create(n_total, math.ceil((n_total + 1) / 2))

    for step in range(1, forest_with_envelope.n_steps):
        old_envelope = list(forest_with_envelope.envelope)
        prev_lower_bound, prev_upper_bound = old_envelope[step - 1]
        new_envelope_prefix = old_envelope[:step] + [(prev_lower_bound + 1, prev_upper_bound)]
        new_envelope = fill_envelope(n_total, new_envelope_prefix)

        old_score = forest_with_envelope.get_score(allowable_error)
        forest_with_envelope.update_envelope_suffix(new_envelope[step:])
        new_score = forest_with_envelope.get_score(allowable_error)

        if new_score <= old_score:
            forest_with_envelope.update_envelope_suffix(old_envelope[step:])

    return forest_with_envelope.envelope


def envelope_to_lower_bound_selected_indices(envelope):
    increments = []

    for i in range(1, len(envelope)):
        if envelope[i][0] > envelope[i-1][0]:
            increments.append(i)

    return increments


def selected_indices_to_symmetric_envelope(n_total, indices):
    lower_boundary = [0]

    for index in indices:
        last_bound = lower_boundary[-1]
        lower_boundary += [last_bound] * (index - len(lower_boundary)) + [last_bound + 1]

    return fill_boundary_to_envelope(n_total, lower_boundary, is_upper=False, symmetrical=True)


def find_max_score_envelope(n_total, allowable_error):
    indices = list(range(1, n_total - 1))
    forests = [
        Forest(n_total, n_good)
        for n_good in [math.ceil((n_total + 1) / 2), math.ceil((n_total - 1) / 2)]
    ]

    envelopes = (
        selected_indices_to_symmetric_envelope(n_total, selected_indices)
        for selected_indices in powerset(indices, max_size=math.ceil(len(indices) / 2))
    )

    return max(
        envelopes,
        key=lambda envelope: min(
            ForestWithEnvelope(forest, envelope).get_score(allowable_error)
            for forest in forests
        )
    )
