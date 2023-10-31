import dataclasses
import functools
import inspect
import itertools
import time

import numpy as np
import pandas as pd
from scipy import stats


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
            print(message)


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


def plot_function(ax, x_axis_arg_name, function, function_kwargs=None, plot_kwargs=None, results_transform=lambda y: y,
                  x_axis_values_transform=lambda x: x, verbose=False):
    function_kwargs = function_kwargs or {}
    plot_kwargs = plot_kwargs or {}

    x_axis_values = function_kwargs.pop(x_axis_arg_name)

    ax.set_xlabel(x_axis_arg_name)

    title = function.__name__
    if function_kwargs:
        title += f" ({stringify_kwargs(function_kwargs)})"
    ax.set_title(title)

    results = np.zeros(len(x_axis_values))

    for i, x_axis_value in enumerate(x_axis_values):
        if verbose:
            print(f"Computing {function.__name__} value at {x_axis_value!r}")
        results[i] = function(**(function_kwargs | {x_axis_arg_name: x_axis_value}))

    results = results_transform(results)
    x_axis_values = x_axis_values_transform(x_axis_values)

    ax.plot(x_axis_values, results, **plot_kwargs)


def plot_functions(ax, x_axis_arg_name, functions, function_kwargs=None, plot_kwargs=None,
                   results_transform=lambda y: y,
                   x_axis_values_transform=lambda x: x,
                   verbose=False):
    if plot_kwargs is None:
        plot_kwargs = {}

    for function in functions:
        if verbose:
            print(f"Plotting {function.__name__}")
        plot_function(ax, x_axis_arg_name, function, dict(function_kwargs), plot_kwargs | dict(label=function.__name__),
                      results_transform, x_axis_values_transform, verbose)

    title = ", ".join(function.__name__ for function in functions)
    if function_kwargs:
        function_kwargs.pop(x_axis_arg_name)
        title += f" ({stringify_kwargs(function_kwargs)})"
    ax.set_title(title)

    ax.legend()


def plot_function_many_curves(ax, x_axis_arg_name, distinct_curves_arg_name, function,
                              function_kwargs=None, plot_kwargs=None):
    function_kwargs = function_kwargs or {}
    plot_kwargs = plot_kwargs or {}

    distinct_curves_arg_values = function_kwargs.pop(distinct_curves_arg_name)

    for distinct_curves_arg_value in distinct_curves_arg_values:
        plot_function(
            ax,
            x_axis_arg_name,
            function,
            function_kwargs | {distinct_curves_arg_name: distinct_curves_arg_value},
            plot_kwargs | dict(label=f"{distinct_curves_arg_name}={distinct_curves_arg_value}")
        )

    title = function.__name__
    if function_kwargs:
        title_kwargs = dict(function_kwargs)
        title_kwargs.pop(x_axis_arg_name)
        title += f" ({stringify_kwargs(title_kwargs)})"
    ax.set_title(title)

    ax.legend()


def rolling_average(numbers, window_length):
    return np.convolve(numbers, np.ones(window_length), "valid") / window_length


def is_mean_surprising(observations, expected_mean, confidence_level=0.9):
    return stats.ttest_1samp(observations, expected_mean).pvalue < 1 - confidence_level


def is_proportion_surprising(observations, expected_proportion, confidence_level=0.9):
    return stats.binomtest(sum(observations), len(observations), expected_proportion).pvalue < 1 - confidence_level
