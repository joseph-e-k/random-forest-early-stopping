import dataclasses
import random
from functools import cached_property
from typing import Sequence

from numpy import cumsum


@dataclasses.dataclass(frozen=True)
class WeightedMultinomial:
    options: Sequence
    weights: Sequence | None = dataclasses.field(default=None)

    @cached_property
    def _cumulative_probabilities(self):
        weights = self.weights or [1] * len(self.options)
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        return cumsum(probabilities)

    def __call__(self):
        r = random.random()
        for option, cumulative_probability in zip(self.options, self._cumulative_probabilities):
            if cumulative_probability >= r:
                return option


@dataclasses.dataclass(frozen=True)
class Multinomial:
    options: Sequence

    def __call__(self):
        return random.choice(self.options)


@dataclasses.dataclass(frozen=True)
class Uniform:
    minimum: float = 0.0
    maximum: float = 1.0

    @cached_property
    def _length(self):
        return self.maximum - self.minimum

    def __call__(self):
        return self.minimum + random.random() * self._length
