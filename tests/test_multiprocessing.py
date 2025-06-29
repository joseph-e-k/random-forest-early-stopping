import functools
import time

import numpy as np
from ste.utils.misc import TimerContext
from ste.utils.multiprocessing import parallelize, parallelize_to_array


def _slow_unary_noop(arg):
    time.sleep(1)
    return arg


def _slowly_double(arg):
    time.sleep(1)
    return 2 * arg


def test_parallelize_sanity():
    with TimerContext(verbose=False) as timer:
        task_outcomes = parallelize(
            _slow_unary_noop,
            argses_to_iter=[(i,) for i in range(10)]
        )
        results = set(outcome.result for outcome in task_outcomes)

    assert results == set(range(10))
    assert timer.elapsed_time < 2


def _binary_operation(x, y):
    time.sleep(1)
    return y + 10 * x


def test_parallelize_to_array_sanity():
    with TimerContext(verbose=False) as timer:
        results = parallelize_to_array(
            _binary_operation,
            argses_to_combine=[range(3), range(5)]
        )
    
    assert timer.elapsed_time < 2

    assert results.shape == (3, 5)

    for i in range(3):
        for j in range(5):
            assert results[i, j] == j + 10 * i


def _other_binary_operation(x, y):
    time.sleep(1)
    return 2**x * 3**y

def test_parallelize_to_array_multiple_functions():
    with TimerContext(verbose=False) as timer:
        results = parallelize_to_array(
            [_binary_operation, _other_binary_operation],
            argses_to_combine=[range(3), range(5)]
        )
    
    assert timer.elapsed_time < 2

    assert results.shape == (2, 3, 5)

    for i in range(3):
        for j in range(5):
            assert results[0, i, j] == j + 10 * i
            assert results[1, i, j] == 2**i * 3**j


def test_parallelize_two_unary_functions():
    with TimerContext(verbose=False) as timer:
        tasks = list(parallelize(
            [_slow_unary_noop, _slowly_double],
            argses_to_combine=[range(3)]
        ))  
    
    assert timer.elapsed_time < 2

    for function in [_slow_unary_noop, _slowly_double]:
        for arg in range(3):
            assert sum(task.function == function and task.args_or_kwargs == (arg,) for task in tasks) == 1

    for task in tasks:
        arg, = task.args_or_kwargs
        if task.index[0] == 0:
            assert task.function == _slow_unary_noop
            assert task.result == arg
        if task.index[0] == 1:
            assert task.function == _slowly_double
            assert task.result == 2 * arg


def test_partial_function():
    results = parallelize_to_array(
        functools.partial(_binary_operation, y=0),
        argses_to_combine=[
            range(5)
        ]
    )
    assert results.shape == (5,)
    assert np.all(results == np.array([0, 10, 20, 30, 40]))
