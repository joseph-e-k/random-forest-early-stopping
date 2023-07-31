import dataclasses
import itertools
import time

import numpy as np
from sklearn.utils import Bunch


def shift_array(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


@dataclasses.dataclass
class TimerContext:
    tag: str = None
    verbose: bool = True
    start_time: float = None
    end_time: float = None

    @property
    def elapsed_time(self):
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

        if self.verbose:
            message = f"Time ({self.tag})" if self.tag is not None else "Time"
            message += f": {self.elapsed_time}s"
            print(message)


def powerset(iterable, max_size=None):
    items = list(iterable)

    if max_size is None:
        max_size = len(items)

    return itertools.chain.from_iterable(
        itertools.combinations(items, n) for n in range(max_size + 1)
    )


def iter_unique_combinations(iterable, length):
    items = list(iterable)

    if length == 0:
        yield ()
        return

    last_index = len(items) - length
    for index, item in enumerate(items[:last_index + 1]):
        for combination in iter_unique_combinations(items[index + 1:], length - 1):
            yield (item,) + combination


def covariates_response_split(dataset, response_column=-1):
    match dataset:
        case (_, _):
            return dataset
        case Bunch():
            return dataset.data, dataset.target
        case np.ndarray():
            if response_column < 0:
                response_column = dataset.shape[1] + response_column
            covariate_columns = tuple(j for j in range(dataset.shape[1]) if j != response_column)
            return dataset[:, covariate_columns], dataset[:, response_column]
        case _:
            raise TypeError(f"expected dataset to be a Bunch, ndarray, or pair; got {dataset!r} instead")
