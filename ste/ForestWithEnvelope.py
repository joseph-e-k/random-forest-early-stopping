from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
from scipy.special import logsumexp

from .Forest import Forest
from .ForestWithStoppingStrategy import ForestWithStoppingStrategy
from .envelopes import Envelope, get_null_envelope, add_increment_to_envelope
from .utils import memoize


@dataclasses.dataclass
class ForestWithEnvelope(ForestWithStoppingStrategy):
    envelope: Envelope

    def get_prob_stop(self):
        prob_stop = np.empty_like(self._log_state_probabilities)

        for i_step in range(self.n_steps - 1):
            prob_stop[i_step, :] = 1 - self._get_mask_for_bounds(self._n_values, *self.envelope[i_step])

        prob_stop[self.n_steps - 1, :] = 1
        return prob_stop

    def _get_log_prob_stop(self):
        with warnings.catch_warnings(category=RuntimeWarning, action="ignore"):
            return np.log(self.get_prob_stop())

    @classmethod
    def create(cls, n_total, n_total_positive, envelope=None):
        if envelope is None:
            envelope = get_null_envelope(n_total)

        forest = Forest(n_total, n_total_positive)
        return cls(forest, envelope)

    @classmethod
    def create_greedy(cls, n_total, n_positive, allowable_error):
        log_allowable_error = np.log(allowable_error)

        envelope = get_null_envelope(n_total)
        forest_with_envelope = cls.create(n_total, math.ceil((n_total + 1) / 2), envelope)

        for i_step in range(1, n_total + 1):
            log_prob_state = forest_with_envelope.get_log_state_probability(i_step, envelope[i_step][0])
            if log_prob_state <= log_allowable_error:
                log_allowable_error = logsumexp([log_allowable_error, log_prob_state], b=[1, -1])
                forest_with_envelope.add_increment_to_envelope(i_step)

        return ForestWithEnvelope.create(n_total, n_positive, envelope)

    def _invalidate_state_probabilities(self, start_index=0):
        self._i_last_valid_state_probabilities = start_index

    @staticmethod
    def _get_mask_for_bounds(size, lower, upper):
        effective_lower = np.clip(lower, 0, size)
        lower_mask = np.concatenate([
            np.zeros(effective_lower),
            np.ones(size - effective_lower)
        ])

        effective_upper = np.clip(upper, -1, size - 1)
        upper_mask = np.concatenate([
            np.ones(effective_upper + 1),
            np.zeros(size - effective_upper - 1)
        ])

        return np.logical_and(lower_mask, upper_mask)

    def add_increment_to_envelope(self, increment_index: int):
        add_increment_to_envelope(self.envelope, increment_index)
        self._invalidate_state_probabilities(start_index=increment_index)


@memoize("get_greedy_stopping_strategy")
def get_greedy_stopping_strategy(n_total, allowable_error):
    fwe = ForestWithEnvelope.create_greedy(n_total=n_total, n_positive=n_total, allowable_error=allowable_error)
    return fwe.get_prob_stop()
