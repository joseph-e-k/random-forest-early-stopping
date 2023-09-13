from __future__ import annotations

import functools
import math

import numpy as np
from scipy.special import logsumexp

from .Forest import Forest
from .envelopes import Envelope, increments_to_symmetric_envelope, get_null_envelope
from .ForestWithEnvelope import ForestWithEnvelope
from .utils import powerset


@functools.lru_cache()
def get_envelope_by_eb_greedily(n_total, allowable_error) -> Envelope:
    n_good = math.ceil((n_total + 1) / 2)
    log_allowable_error = np.log(allowable_error)

    envelope = get_null_envelope(n_total)
    forest_with_envelope = ForestWithEnvelope.create(n_total, n_good, envelope)

    for i_step in range(1, n_total + 1):
        log_prob_state = forest_with_envelope.get_log_state_probability(i_step, envelope[i_step][0])
        if log_prob_state <= log_allowable_error:
            log_allowable_error = logsumexp([log_allowable_error, log_prob_state], b=[1, -1])
            forest_with_envelope.add_increment_to_envelope(i_step)

    return envelope
