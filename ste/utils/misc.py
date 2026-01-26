import dataclasses
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import itertools
import os
import time
from typing import Callable

import numpy as np
from scipy import stats
from diskcache.core import full_name

from .logging import get_module_logger


_logger = get_module_logger()


RESULTS_DIRECTORY = os.path.join(os.path.dirname(__file__), "../../results")


def unzip(sequence_of_tuples):
    """
    Unzip a sequence of tuples into separate lists.

    Args:
        sequence_of_tuples (Iterable[tuple]): The sequence of tuples to unzip.

    Returns:
        list[list]: A list of lists, where each list corresponds to one index of the tuples.
    """
    return list(zip(*sequence_of_tuples))


def extend_array(old_array, new_shape, fill_value=0):
    """
    Extend an array to a new shape, filling new elements with a specified value.

    Args:
        old_array (np.ndarray): The original array.
        new_shape (tuple[int]): The desired shape of the new array.
        fill_value (Any, optional): The value to fill new elements with. Defaults to 0.

    Returns:
        np.ndarray: The new, extended array.

    Raises:
        ValueError: If the new shape has fewer dimensions or shrinks any dimension.
    """
    old_shape = old_array.shape
    if len(new_shape) != len(old_shape):
        raise ValueError(f"extend_array cannot change number of dimensions ({len(old_shape)} -> {len(new_shape)})")
    
    for i_dimension, (old_dimension, new_dimension) in enumerate(zip(old_shape, new_shape)):
        if new_dimension < old_dimension:
            raise ValueError(f"extend_array cannot shrink dimensions (dimension {i_dimension}: {old_dimension} -> {new_dimension})")

    new_array = np.full(new_shape, fill_value, dtype=old_array.dtype)
    slices = tuple(slice(0, old_dimension) for old_dimension in old_shape)
    new_array[slices] = old_array
    return new_array


def shift_array(arr, num, fill_value=np.nan):
    """
    Create a new array which contains the same elements as an existing one, shifted by a certain number of positions.

    Args:
        arr (np.ndarray): The original array of values.
        num (int): The number of positions to shift. Positive values shift right, negative values shift left.
        fill_value (Any, optional): The value to fill in the empty positions. Defaults to np.nan.

    Returns:
        np.ndarray: The new, shifted array.
    """
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
    """
    Context manager to measure the execution time of a code block.

    Attributes:
        tag (str, optional): A tag to include in the log message. Defaults to None.
        verbose (bool, optional): Whether to log the elapsed time. Defaults to True.
        start_time (float, optional): The start time of the context. Defaults to None.
        end_time (float, optional): The end time of the context. Defaults to None.
    """
    tag: str = None
    verbose: bool = True
    start_time: float = None
    end_time: float = None

    @property
    def elapsed_time(self):
        """
        Calculate the elapsed time.

        Returns:
            float | None: The elapsed time in seconds, or None if the context has not been exited.
        """
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
    """
    Decorator to measure and log the execution time of a function.

    Args:
        function (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.
    """
    @functools.wraps(function)
    def function_wrapper(*args, **kwargs):
        bound_sig = inspect.signature(function).bind(*args, **kwargs)
        with TimerContext(f"{function.__name__}({stringify_kwargs(bound_sig.arguments)})"):
            return function(*args, **kwargs)
    return function_wrapper


def powerset(iterable, max_size=None):
    """
    Generate all subsets of an iterable up to a specified size.

    Args:
        iterable (Iterable): The input iterable.
        max_size (int, optional): The maximum size of subsets to generate. Defaults to the size of the iterable.

    Returns:
        Iterator[tuple]: An iterator over subsets of the iterable.
    """
    items = list(iterable)

    if max_size is None:
        max_size = len(items)

    return itertools.chain.from_iterable(
        itertools.combinations(items, n) for n in range(max_size + 1)
    )


def iter_unique_combinations(iterable, length):
    """
    Generate unique combinations of a specified length from an iterable.

    Args:
        iterable (Iterable): The input iterable.
        length (int): The length of combinations to generate.

    Yields:
        tuple: A unique combination of elements.
    """
    items = list(iterable)

    if length == 0:
        yield ()
        return

    last_index = len(items) - length
    for index, item in enumerate(items[:last_index + 1]):
        for combination in iter_unique_combinations(items[index + 1:], length - 1):
            yield (item,) + combination


def stringify_kwargs(kwargs: dict) -> str:
    """
    Convert a dictionary of keyword arguments to a string representation.

    Args:
        kwargs (dict[str, Any]): The dictionary of keyword arguments.

    Returns:
        str: A string representation of the keyword arguments, as they would appear as part of a Python function call.
    """
    if not kwargs:
        return ""

    return ", ".join(f"{key}={value!r}" for key, value in kwargs.items())


def rolling_average(numbers, window_length):
    """
    Compute the rolling average of a sequence of numbers.

    Args:
        numbers (Iterable[float]): The input sequence of numbers.
        window_length (int): The length of the rolling window.

    Returns:
        np.ndarray: The rolling averages.
    """
    return np.convolve(numbers, np.ones(window_length), "valid") / window_length


def is_mean_surprising(observations, expected_mean, confidence_level=0.9):
    """
    Test whether the mean of observations is significantly different from an expected mean.

    Args:
        observations (Iterable[float]): The observed values.
        expected_mean (float): The expected mean.
        confidence_level (float, optional): The confidence level for the test. Defaults to 0.9.

    Returns:
        bool: True if the mean is surprising, False otherwise.
    """
    return stats.ttest_1samp(observations, expected_mean).pvalue < 1 - confidence_level


def is_proportion_surprising(observations, expected_proportion, confidence_level=0.9):
    """
    Test whether the proportion of successes in observations is significantly different from an expected proportion.

    Args:
        observations (Iterable[bool]): The observed successes (True/False).
        expected_proportion (float): The expected proportion of successes.
        confidence_level (float, optional): The confidence level for the test. Defaults to 0.9.

    Returns:
        bool: True if the proportion is surprising, False otherwise.
    """
    return stats.binomtest(sum(observations), len(observations), expected_proportion).pvalue < 1 - confidence_level


def forwards_to[RInner, ROuter, **P](inner_function: Callable[P, RInner]) -> Callable[[Callable[P, ROuter]], Callable[P, ROuter]]:
    """Decorator to mark a function as forwarding its arguments to another function.
    This is useful for type hinting and documentation purposes."""
    def decorator(outer_function: Callable[P, ROuter], inner_function=inner_function) -> Callable[P, ROuter]:
        return outer_function
    return decorator


def get_output_path(partial_file_name: str, file_name_suffix=".pdf"):
    """
    Generate a timestamped output file path.

    Args:
        partial_file_name (str): The base name of the file.
        file_name_suffix (str, optional): The file extension. Defaults to ".pdf".

    Returns:
        str: The full output file path.
    """
    timestamp = datetime.now().astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    return os.path.join(RESULTS_DIRECTORY, f"{partial_file_name}_{timestamp}{file_name_suffix}")


def enumerate_product(*iterables):
    """
    Enumerate the Cartesian product of multiple iterables.

    Args:
        *iterables (Iterable): The input iterables.

    Returns:
        Iterator[tuple[tuple[int, ...], tuple]]: An iterator over (index, t) pairs, where t is a tuple from the Cartesian product.
    """
    indices = itertools.product(*(
        range(len(iterable))
        for iterable in iterables  
    ))
    return zip(
        indices,
        itertools.product(*iterables)
    )


def repeat_enumerated(n_repetitions, enumerated_iterable):
    """
    Repeat an enumerated iterable a specified number of times.

    Args:
        n_repetitions (int): The number of repetitions.
        enumerated_iterable (Iterable): The enumerated iterable to repeat.

    Yields:
        tuple[tuple[int, ...], Any]: A pair of (index, item), where the index is itself a tuple (of length at least 2, potentially more if the original already had tuple-y indices).
    """
    for i_rep, (index, item) in itertools.product(range(n_repetitions), enumerated_iterable):
        if isinstance(index, tuple):
            yield (i_rep,) + index, item
        else:
            yield (i_rep, index), item


def get_name(callable):
    """
    Get the name of a callable object.

    Args:
        callable (Callable): The callable object.

    Returns:
        str: The name of the callable, or "<unnamed>" if it has no name.
    """
    if isinstance(callable, functools.partial):
        return callable.func.__name__
    return getattr(callable, "__name__", "<unnamed>")


def swap_indices_of_axis(array, i, j, axis):
    """
    Swap two indices along a specified axis of an array.

    Args:
        array (np.ndarray): The input array.
        i (int): The first index to swap.
        j (int): The second index to swap.
        axis (int): The axis along which to swap indices.

    Returns:
        np.ndarray: A view of the input array, with swapped indices.
    """
    array_swap = np.swapaxes(array, axis, 0)
    array_swap[[i, j], ...] = array_swap[[j, i], ...]
    array = np.swapaxes(array_swap, 0, axis)
    return array


def function_call_to_tuple(function, name, args_to_ignore, arg_transformations, /, *args, **kwargs):
    """
    Convert a function call into a tuple representation.

    Args:
        function (Callable): The function being called.
        name (str): The name of the function.
        args_to_ignore (Iterable[str]): Arguments to ignore in the tuple.
        arg_transformations (dict[str, Callable]): Transformations to apply to specific arguments.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        tuple: A tuple representing the function call: (name, (arg0_name, arg0_value), (arg1_name, arg1_value), ...).
    """
    if name is None:
        name = full_name(function)

    signature = inspect.signature(function)
    try:
        bound_arguments = signature.bind(*args, **kwargs)
    except TypeError as e:
        raise TypeError(f"{function} cannot be bound to arguments {args} and keyword arguments {kwargs}") from e
    bound_arguments.apply_defaults()

    key = [name]

    for arg_name, arg_value in bound_arguments.arguments.items():
        if arg_name in args_to_ignore:
            continue
        if arg_name in arg_transformations:
            transformation = arg_transformations[arg_name]
            arg_value = transformation(arg_value)
        key.append((arg_name, arg_value))

    return tuple(key)


class Dummy:
    """
    A dummy object that can pretend to have any attribute or method.

    Attributes:
        rep (str): The string representation of the dummy object.
    """
    def __init__(self, rep):
        self.__rep = rep

    def __call__(self, *args, **kwargs):
        """
        Simulate a callable object.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dummy: A new dummy object representing the return value of the call.
        """
        args_rep = ", ".join(repr(arg) for arg in args)
        kwargs_rep = ", ".join(f"{key!r}={value!r}" for key, value in kwargs.items())
        args_and_kwargs_rep = ", ".join([args_rep, kwargs_rep])
        return Dummy(f"{self!r}({args_and_kwargs_rep})")
    
    def __getattr__(self, attribute):
        """
        Simulate attribute access.

        Args:
            attribute (str): The attribute name.

        Returns:
            Dummy: A new dummy object representing the attribute.
        """
        return Dummy(f"{self!r}.{attribute}")

    def __repr__(self):
        """
        Get the string representation of the dummy object.

        Returns:
            str: The string representation.
        """
        return self.__rep


def no_change(arg):
    """
    Return the input argument unchanged.

    Args:
        arg (Any): The input argument.

    Returns:
        Any: The same argument.
    """
    return arg


def retain_central_nonzeros(arr):
    center = len(arr) // 2
    nonzero_indices = np.nonzero(arr)[0]

    below = nonzero_indices[nonzero_indices < center]
    above = nonzero_indices[nonzero_indices > center]

    lower_idx = below[-1] if below.size > 0 else None
    upper_idx = above[0] if above.size > 0 else None

    mask = np.zeros_like(arr, dtype=bool)
    if lower_idx is not None:
        mask[lower_idx] = True
    if upper_idx is not None:
        mask[upper_idx] = True

    return np.where(mask, arr, 0)
