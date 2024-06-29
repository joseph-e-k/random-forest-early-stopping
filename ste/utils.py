import dataclasses
from datetime import datetime, timezone
import functools
import inspect
import itertools
import os
import time
from typing import Callable

import numpy as np
import pandas as pd
from diskcache import Cache
from diskcache.core import full_name
from scipy import stats

from ste.logging_utils import get_module_logger


_logger = get_module_logger()


DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "../data")
RESULTS_DIRECTORY = os.path.join(os.path.dirname(__file__), "../results")
DATASETS = {
    "Banknotes": pd.read_csv(os.path.join(DATA_DIRECTORY, "data_banknote_authentication.txt")),
    "Heart Attacks": pd.read_csv(os.path.join(DATA_DIRECTORY, "heart_attack.csv")),
    "Salaries": pd.read_csv(os.path.join(DATA_DIRECTORY, "adult.data")),
    "Dry Beans": pd.read_excel(os.path.join(DATA_DIRECTORY, "dry_beans.xlsx"))
}


cache = Cache(os.path.join(os.path.dirname(__file__), ".cache"))


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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

        if self.verbose:
            message = f"Time ({self.tag})" if self.tag is not None else "Time"
            message += f": {self.elapsed_time}s"
            _logger.info(message)


def timed(function):
    @functools.wraps(function)
    def function_wrapper(*args, **kwargs):
        bound_sig = inspect.signature(function).bind(*args, **kwargs)
        with TimerContext(f"{function.__name__}({stringify_kwargs(bound_sig.arguments)})"):
            return function(*args, **kwargs)
    return function_wrapper


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


def covariates_response_split(dataset: pd.DataFrame, response_column=-1):
    if response_column < 0:
        response_column = dataset.shape[1] + response_column
    covariate_columns = [j for j in range(dataset.shape[1]) if j != response_column]
    return dataset.iloc[:, covariate_columns], dataset.iloc[:, response_column]


def stringify_kwargs(kwargs: dict) -> str:
    if not kwargs:
        return ""

    return ", ".join(f"{key}={value!r}" for key, value in kwargs.items())


def rolling_average(numbers, window_length):
    return np.convolve(numbers, np.ones(window_length), "valid") / window_length


def is_mean_surprising(observations, expected_mean, confidence_level=0.9):
    return stats.ttest_1samp(observations, expected_mean).pvalue < 1 - confidence_level


def is_proportion_surprising(observations, expected_proportion, confidence_level=0.9):
    return stats.binomtest(sum(observations), len(observations), expected_proportion).pvalue < 1 - confidence_level


def _robust_cache_key(function, name, args_to_ignore, *args, **kwargs):
    signature = inspect.signature(function)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return (name,) + tuple(
        (key, value) for (key, value) in bound_arguments.arguments.items() if key not in args_to_ignore
    )


def memoize(name=None, args_to_ignore=()):
    def decorator(function, name=name, args_to_ignore=args_to_ignore):
        if name is None:
            name = full_name(function)
        memoized = cache.memoize(name=name)(function)
        memoized.__cache_key__ = functools.partial(_robust_cache_key, memoized, name, args_to_ignore)
        return memoized
    return decorator


def forwards_to[RInner, ROuter, **P](inner_function: Callable[P, RInner]) -> Callable[[Callable[P, ROuter]], Callable[P, ROuter]]:
    def decorator(outer_function: Callable[P, ROuter], inner_function=inner_function) -> Callable[P, ROuter]:
        return outer_function
    return decorator


def get_output_path(partial_file_name: str):
    timestamp = datetime.now().astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    return os.path.join(RESULTS_DIRECTORY, f"{partial_file_name}_{timestamp}")


def enumerate_product(*iterables):
    indices = itertools.product(*(
        range(len(iterable))
        for iterable in iterables  
    ))
    return zip(
        indices,
        itertools.product(*iterables)
    )
