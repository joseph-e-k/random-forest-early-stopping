from __future__ import annotations

import numpy as np

from typing import TypeAlias, Iterable

Envelope: TypeAlias = np.ndarray  # array of nonnegative ints with shape (n, 2) for some n
Boundary: TypeAlias = np.ndarray  # 1-d array of nonnegative ints


def get_null_envelope(n_total) -> Envelope:
    envelope = np.empty((n_total + 1, 2), dtype=int)
    envelope[0, 0] = 0
    fill_envelope_by_partial_lower_boundary(envelope, 1)
    return envelope


def fill_lower_boundary(partial_boundary: Boundary, n_valid_bounds: int) -> None:
    partial_boundary[n_valid_bounds:] = partial_boundary[n_valid_bounds - 1]


def fill_upper_boundary(partial_boundary: Boundary, n_valid_bounds: int) -> None:
    bump = partial_boundary[n_valid_bounds - 1] + 1
    partial_boundary[n_valid_bounds:] = np.arange(len(partial_boundary) - n_valid_bounds) + bump


def fill_boundary(partial_boundary: Boundary, n_valid_bounds: int, is_upper: bool) -> None:
    if is_upper:
        fill_upper_boundary(partial_boundary, n_valid_bounds)
    else:
        fill_lower_boundary(partial_boundary, n_valid_bounds)


def fill_envelope_by_partial_lower_boundary(envelope: Envelope, n_valid_bounds: int) -> None:
    fill_boundary(envelope[:, 0], n_valid_bounds, is_upper=False)
    fill_mirror_boundary(envelope[:, 1], envelope[:, 0])


def fill_mirror_boundary(dst: Boundary, src: Boundary) -> None:
    dst[:] = np.arange(len(src)) - src


def envelope_to_lower_bound_increments(envelope):
    increments = []

    for i in range(1, len(envelope)):
        if envelope[i][0] > envelope[i-1][0]:
            increments.append(i)

    return increments


def increments_to_symmetric_envelope(n_total: int, increments: Iterable[int]) -> Envelope:
    envelope = np.empty((n_total + 1, 2), dtype=int)
    lower_boundary = envelope[:, 0]
    lower_boundary[0] = 0
    i_bound = 1
    last_bound = 0

    for i_increment in increments:
        lower_boundary[i_bound:i_increment] = last_bound
        lower_boundary[i_increment] = last_bound + 1
        i_bound = i_increment + 1
        last_bound += 1

    fill_envelope_by_partial_lower_boundary(envelope, i_bound)
    return envelope


def add_increment_to_envelope(envelope: Envelope, increment_index: int) -> None:
    envelope[increment_index:, 0] += 1
    envelope[increment_index:, 1] -= 1


def describe_envelope(envelope):
    lower_boundary = [bounds[0] for bounds in envelope]

    shifts = []

    for i in range(1, len(lower_boundary)):
        if lower_boundary[i] > lower_boundary[i-1]:
            shifts.append((i, lower_boundary[i]))

    return ", ".join(f"< {value} / {index}" for (index, value) in shifts)
