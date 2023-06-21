from __future__ import annotations

import math

from .Forest import Forest
from .envelopes import fill_envelope, Envelope, increments_to_symmetric_envelope, get_null_envelope
from .ForestWithEnvelope import ForestWithEnvelope
from .utils import powerset


def get_envelope_by_eb_greedily(n_total, allowable_error) -> Envelope:
    increments = []

    n_good = math.ceil((n_total + 1) / 2)

    envelope = increments_to_symmetric_envelope(n_total, increments)
    forest_with_envelope = ForestWithEnvelope.create(n_total, n_good, envelope)

    for step in range(1, n_total + 1):
        prob_state = forest_with_envelope.get_state_probability(step, envelope[step][0])
        if prob_state <= allowable_error:
            allowable_error -= prob_state
            increments.append(step)
            envelope = increments_to_symmetric_envelope(n_total, increments)
            forest_with_envelope.update_envelope_suffix(envelope[step:])

    return envelope


def get_envelope_by_score_greedily(n_total, allowable_error):
    increments = []

    n_good = math.ceil((n_total + 1) / 2)

    envelope = increments_to_symmetric_envelope(n_total, increments)
    forest_with_envelope = ForestWithEnvelope.create(n_total, n_good, envelope)
    score = forest_with_envelope.get_score(allowable_error)

    for step in range(1, n_total + 1):
        increments.append(step)
        envelope = increments_to_symmetric_envelope(n_total, increments)
        forest_with_envelope.update_envelope_suffix(envelope[step:])

        if forest_with_envelope.get_score(allowable_error) <= score:
            increments.pop()
            envelope = increments_to_symmetric_envelope(n_total, increments)
            forest_with_envelope.update_envelope_suffix(envelope[step:])

    return envelope


def get_envelope_by_score_heuristically(n_total, allowable_error):
    n_good = math.ceil((n_total + 1) / 2)
    best_envelope = get_null_envelope(n_total)
    best_score = 1 / n_total
    increment_sequence_length = int(n_total / 2)

    for step in range(1, n_total + 1 - increment_sequence_length):
        increments = list(range(step, step + increment_sequence_length))
        envelope = increments_to_symmetric_envelope(n_total, increments)
        forest_with_envelope = ForestWithEnvelope.create(n_total, n_good, envelope)
        score = forest_with_envelope.get_score(allowable_error)

        if score > best_score:
            best_score = score
            best_envelope = envelope

    return best_envelope


def get_envelope_by_score_combinatorically(n_total, allowable_error):
    indices = list(range(1, n_total))
    forests = [
        Forest(n_total, n_good)
        for n_good in [math.ceil((n_total + 1) / 2), math.ceil((n_total - 1) / 2)]
    ]

    envelopes = (
        increments_to_symmetric_envelope(n_total, selected_indices)
        for selected_indices in powerset(indices, max_size=math.ceil(len(indices) / 2))
    )

    return max(
        envelopes,
        key=lambda envelope: min(
            ForestWithEnvelope(forest, envelope).get_score(allowable_error)
            for forest in forests
        )
    )
