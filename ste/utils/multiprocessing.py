import contextlib
import dataclasses
import functools
import multiprocessing
import operator
import os
import random
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np
import tblib

from .logging import get_breadcrumbs, get_module_logger, logged, breadcrumbs
from .misc import TimerContext, enumerate_product, get_name, repeat_enumerated


_logger = get_module_logger()


N_WORKER_PROCESSES = int(os.getenv("STE_N_WORKER_PROCESSES", 32))
SHOULD_DUMMY_MULTIPROCESSING = bool(os.getenv("STE_DUMMY_MULTIPROCESS", False))


@dataclasses.dataclass
class _RawTaskOutcome:
    """
    Represents the outcome of a task executed in parallel, before processing to parse exception traceback.

    Attributes:
        index (int | tuple[int, ...]): The index of the task.
        args_or_kwargs (tuple | dict): The arguments or keyword arguments used for the task.
        duration (timedelta): The duration of the task execution.
        result (Any, optional): The result of the task. Defaults to None.
        exception (Exception | None, optional): The exception raised during the task, if any. Defaults to None.
        traceback_string (str, optional): The traceback string of the exception, if any. Defaults to None.
    """
    index: int | tuple[int, ...]
    args_or_kwargs: tuple | dict
    duration: timedelta
    result: Any = None
    exception: Exception | None = None
    traceback_string: str = None


@dataclasses.dataclass(frozen=True)
class TaskOutcome:
    """
    Represents the processed outcome of a task executed in parallel.

    Attributes:
        index (int | tuple[int, ...]): The index of the task.
        args_or_kwargs (tuple | dict): The arguments or keyword arguments used for the task.
        duration (timedelta): The duration of the task execution.
        result (Any): The result of the task.
        exception (Exception): The exception raised during the task, if any, along with its traceback.
    """
    index: int | tuple[int, ...]
    args_or_kwargs: tuple | dict
    duration: timedelta
    result: Any = None
    exception: Exception = None


@dataclasses.dataclass
class _Job:
    """
    Represents a job consisting of multiple tasks to be executed in parallel, each being the same function with different arguments.

    Attributes:
        function (Callable): The function to execute for each task.
        name (str): The name of the job.
        breadcrumbs_at_creation (tuple[str, ...]): The breadcrumbs at the time of job creation.
        n_total_tasks (int): The total number of tasks. Defaults to None.
        start_time_ns (int): The start time of the job in nanoseconds.
        n_completed_tasks (int): The number of completed tasks.
        random_nonce (float): A random nonce for seeding random number generators.
        np_random_nonce (float): A random nonce for seeding NumPy random number generators.
    """
    function: Callable
    name: str
    breadcrumbs_at_creation: tuple[str, ...]
    n_total_tasks: int = None
    start_time_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    n_completed_tasks: int = 0
    random_nonce: float = dataclasses.field(default_factory=random.random)
    np_random_nonce: float = dataclasses.field(default_factory=np.random.random)

    def run_single_task(self, index_and_args_or_kwargs):
        """
        Execute a single task.

        Args:
            index_and_args_or_kwargs (tuple): A tuple containing the task index and arguments.

        Returns:
            _RawTaskOutcome: The raw outcome of the task.
        """
        index, args_or_kwargs = index_and_args_or_kwargs
        task_name = self.get_single_task_name(index)

        random.seed(hash((self.random_nonce, index)) % (2 ** 32 - 1))
        np.random.seed(hash((self.np_random_nonce, index)) % (2 ** 32 - 1))
        
        try:
            with breadcrumbs(self.breadcrumbs_at_creation + (task_name,)), TimerContext(verbose=False) as timer:
                if isinstance(args_or_kwargs, Mapping):
                    result = self.function(**args_or_kwargs)
                else:
                    result = self.function(*args_or_kwargs)
        except Exception as e:
            tb_string = traceback.format_exc()
            return _RawTaskOutcome(
                index,
                args_or_kwargs,
                timer.elapsed_time,
                exception=e,
                traceback_string=tb_string
            )
        else:
            return _RawTaskOutcome(
                index,
                args_or_kwargs,
                timer.elapsed_time,
                result=result
            )
        
    def get_single_task_name(self, index):
        """
        Get the name of a single task.

        Args:
            index (int | tuple[int, ...]): The index of the task. Can be a tuple if the job was started using `argses_to_combine`.

        Returns:
            str: The name of the task.
        """
        if isinstance(index, int):
            return f"{self.name}[{index}]"
        return f"{self.name}[{', '.join(str(i) for i in index)}]"
    
    def single_task_completed(self, task_index):
        """
        Mark a single task as completed and log its progress.

        Args:
            task_index (int | tuple[int, ...]): The index of the completed task.
        """
        self.n_completed_tasks += 1
        now = datetime.now().astimezone(timezone.utc)
        task_name = self.get_single_task_name(task_index)

        if self.n_total_tasks is None:
            log_message = f"{task_name} completed ({self.n_completed_tasks} total)"
        else:
            ns_so_far = time.monotonic_ns() - self.start_time_ns
            ns_per_task = ns_so_far / self.n_completed_tasks
            n_remaining_tasks = self.n_total_tasks - self.n_completed_tasks
            expected_time_remaining_ns = ns_per_task * n_remaining_tasks
            expected_time_remaining = timedelta(microseconds=expected_time_remaining_ns // 1e3)
            expected_end_time = now + expected_time_remaining
            timing_estimate = f"expect to be finished at {expected_end_time.strftime('%Y-%m-%d %H:%M:%S')}"
            log_message = f"{task_name} completed ({self.n_completed_tasks} / {self.n_total_tasks}; {timing_estimate})"

        _logger.info(log_message)


def _infer_n_tasks(reps, argses_to_iter, argses_to_combine):
    """
    Infer the total number of tasks to execute.

    Args:
        reps (int | None): Number of times each argument set is repeated.
        argses_to_iter (Iterable | None): Iterable of argument sets to iterate over. Mutually exclusive with `argses_to_combine`.
        argses_to_combine (Iterable | None): Iterable of iterables of individual arguments to combine into argument sets.

    Returns:
        int | None: The total number of tasks, or None if it cannot be determined.
    """
    if reps is None:
        reps = 1
    try:
        return reps * len(argses_to_iter)
    except TypeError:
        try:
            return reps * functools.reduce(
                operator.mul,
                (len(values) for values in argses_to_combine)
            )
        except TypeError:
            return None


def _process_raw_task_outcome(raw_outcome: _RawTaskOutcome, reraise_exceptions: bool) -> TaskOutcome:
    """
    Process the raw outcome of a task by attaching the traceback to the exception if it exists.
    If `reraise_exceptions` is True, re-raise the exception.

    Args:
        raw_outcome (_RawTaskOutcome): The raw outcome of the task.
        reraise_exceptions (bool): Whether to re-raise exceptions.

    Returns:
        TaskOutcome: The processed task outcome.

    Raises:
        Exception: If `reraise_exceptions` is True and an exception occurred during the task.
    """
    if raw_outcome.exception is None:
        return TaskOutcome(
            index=raw_outcome.index,
            args_or_kwargs=raw_outcome.args_or_kwargs,
            duration=raw_outcome.duration,
            result=raw_outcome.result
        )
    
    traceback = tblib.Traceback.from_string(raw_outcome.traceback_string).as_traceback()
    exception = raw_outcome.exception.with_traceback(traceback)
    if reraise_exceptions:
        raise exception
    return TaskOutcome(
        index=raw_outcome.index,
        args_or_kwargs=raw_outcome.args_or_kwargs,
        duration=raw_outcome.duration,
        exception=exception
    )


@logged()
def parallelize(function, reps=None, argses_to_iter=None, argses_to_combine=None, n_workers=N_WORKER_PROCESSES, reraise_exceptions=True, job_name=None, dummy=False):
    """
    Execute a function in parallel across multiple tasks.

    Args:
        function (Callable): The function to execute.
        reps (int, optional): Number of repetitions per argument set. Defaults to None.
        argses_to_iter (Iterable, optional): Iterable of argument sets to iterate over. Mutually exclusive with `argses_to_combine`.
        argses_to_combine (Iterable, optional): Iterable of iterables of individual arguments to combine into argument sets.
        n_workers (int, optional): Number of worker processes. Defaults to `N_WORKER_PROCESSES`.
        reraise_exceptions (bool, optional): Whether to re-raise exceptions that occur in tasks. Defaults to True.
        job_name (str, optional): The name of the job. Defaults to None.
        dummy (bool, optional): If True, no actual multiprocessing will be used. Defaults to False.

    Yields:
        TaskOutcome: The outcome of each task.
    """
    if not ((argses_to_iter is None) ^ (argses_to_combine is None)):
        raise TypeError("argses_to_iter or argses_to_combine must be specified (but not both)")
    
    if job_name is None:
        job_name = get_name(function)
    
    _logger.info(f"Preparing task pool for {job_name}")

    n_tasks = _infer_n_tasks(reps, argses_to_iter, argses_to_combine)
    
    if argses_to_iter is None:
        indices_and_argses = enumerate_product(*argses_to_combine)
    else:
        indices_and_argses = enumerate(argses_to_iter)

    if reps is not None:
        indices_and_argses = repeat_enumerated(reps, indices_and_argses)

    job = _Job(function, job_name, get_breadcrumbs(), n_tasks)

    if SHOULD_DUMMY_MULTIPROCESSING:
        _logger.info("Falling back to dummy multiprocessing, per environment variable flag")
        dummy = True

    elif multiprocessing.current_process().daemon:
        _logger.warning("Daemon process; falling back to dummy behaviour")
        dummy = True

    elif n_tasks == 1:
        _logger.info("Only 1 task; dummy mode activated")
        dummy = True

    if dummy:
        context = contextlib.nullcontext()
        mapper = map
    else:
        context = multiprocessing.Pool(n_workers)
        mapper = context.imap_unordered

    with context:
        for raw_outcome in mapper(job.run_single_task, indices_and_argses):
            outcome = _process_raw_task_outcome(raw_outcome, reraise_exceptions)
            job.single_task_completed(raw_outcome.index)
            yield outcome


def parallelize_to_array(function, reps=None, argses_to_iter=None, argses_to_combine=None, n_workers=N_WORKER_PROCESSES, reraise_exceptions=True, job_name=None, dummy=False):
    """
    Execute a function in parallel and store the results in an array.

    Args:
        function (Callable): The function to execute.
        reps (int, optional): Number of repetitions per argument set. Defaults to None (equivalent to 1, except it is given no dimension in the result array).
        argses_to_iter (Iterable, optional): Iterable of argument sets to iterate over. Mutually exclusive with `argses_to_combine`.
        argses_to_combine (Iterable, optional): Iterable of iterables of individual arguments to combine into argument sets.
        n_workers (int, optional): The number of worker processes. Defaults to `N_WORKER_PROCESSES`.
        reraise_exceptions (bool, optional): Whether to re-raise exceptions from tasks. Defaults to True.
        job_name (str, optional): The name of the job. Defaults to None.
        dummy (bool, optional): If True, no actual multiprocessing will be used. Defaults to False.

    Returns:
        np.ndarray: An array containing the results of the tasks. Its dimensions are inferred from the input arguments:
            - If `reps` is provided, the first dimension is `reps`.
            - If `argses_to_iter` is provided, the (rest of the) shape is `(len(argses_to_iter),)`.
            - If `argses_to_combine` is provided, the (rest of the) shape matches that of `argses_to_combine`.
    """
    results_array = None

    if argses_to_iter:
        results_shape = (len(argses_to_iter),)
    else:
        results_shape = tuple(len(args) for args in argses_to_combine)
    
    if reps is not None:
        results_shape = (reps,) + results_shape

    for task in parallelize(function, reps, argses_to_iter, argses_to_combine, n_workers, reraise_exceptions, job_name, dummy):
        result = task.result

        if results_array is None:
            if isinstance(result, np.ndarray):
                results_shape += result.shape
                result_type = result.dtype
            else:
                result_type = type(result)

            results_array = np.empty(shape=results_shape, dtype=result_type)
        
        results_array[task.index] = result

    return results_array
