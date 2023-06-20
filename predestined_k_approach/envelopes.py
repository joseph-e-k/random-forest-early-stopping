from __future__ import annotations

from typing import TypeAlias

Envelope: TypeAlias = list[tuple[int, int]]


def get_null_envelope(n_total) -> Envelope:
    return fill_boundary_to_envelope(n_total, [0], is_upper=False)


def fill_lower_boundary(n_total, partial_boundary: list[int]) -> list[int]:
    n_steps = n_total + 1
    return partial_boundary + [partial_boundary[-1]] * (n_steps - len(partial_boundary))


def fill_upper_boundary(n_total, partial_boundary: list[int]) -> list[int]:
    n_steps = n_total + 1
    return partial_boundary + [
        partial_boundary[-1] + i + 1
        for i in range(n_steps - len(partial_boundary))
    ]


def fill_boundary(n_total, partial_boundary: list[int], is_upper: bool) -> list[int]:
    if is_upper:
        return fill_upper_boundary(n_total, partial_boundary)
    return fill_lower_boundary(n_total, partial_boundary)


def fill_boundary_to_envelope(n_total, partial_boundary: list[int], is_upper: bool, symmetrical: bool = False) -> Envelope:
    boundary = fill_boundary(n_total, partial_boundary, is_upper)

    if symmetrical:
        other_boundary = get_mirror_boundary(boundary)
    else:
        other_boundary = fill_boundary(n_total, [0], not is_upper)

    if is_upper:
        lower_boundary, upper_boundary = other_boundary, boundary
    else:
        lower_boundary, upper_boundary = boundary, other_boundary

    return list(zip(lower_boundary, upper_boundary))


def fill_envelope(n_total, partial_envelope: Envelope) -> Envelope:
    partial_lower_boundary = [l for (l, u) in partial_envelope]
    partial_upper_boundary = [u for (l, u) in partial_envelope]

    lower_boundary = fill_lower_boundary(n_total, partial_lower_boundary)
    upper_boundary = fill_upper_boundary(n_total, partial_upper_boundary)

    return list(zip(lower_boundary, upper_boundary))


def get_mirror_boundary(boundary: list[int]) -> list[int]:
    return [
        i_step - bound
        for i_step, bound
        in enumerate(boundary)
    ]


def envelope_to_lower_bound_increments(envelope):
    increments = []

    for i in range(1, len(envelope)):
        if envelope[i][0] > envelope[i-1][0]:
            increments.append(i)

    return increments


def increments_to_symmetric_envelope(n_total, increments):
    lower_boundary = [0]

    for index in increments:
        last_bound = lower_boundary[-1]
        lower_boundary += [last_bound] * (index - len(lower_boundary)) + [last_bound + 1]

    return fill_boundary_to_envelope(n_total, lower_boundary, is_upper=False, symmetrical=True)
