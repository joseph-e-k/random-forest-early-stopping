import time
from ste.utils.misc import TimerContext
from ste.utils.multiprocessing import parallelize, parallelize_to_array


def _unary_operation(arg):
    time.sleep(1)
    return arg


def test_parallelize_sanity():
    with TimerContext(verbose=False) as timer:
        task_outcomes = parallelize(
            _unary_operation,
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
