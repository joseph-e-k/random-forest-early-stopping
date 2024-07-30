import contextlib
import dataclasses
import functools
import multiprocessing
import multiprocessing.dummy
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

from ste.utils.logging import get_breadcrumbs, get_module_logger, logged, breadcrumbs
from ste.utils.misc import TimerContext, enumerate_product, forwards_to, get_name, repeat_enumerated


_logger = get_module_logger()


N_WORKER_PROCESSES = int(os.getenv("STE_N_WORKER_PROCESSES", 32))


@dataclasses.dataclass
class _RawTaskOutcome:
    index: int | tuple[int, ...]
    args_or_kwargs: tuple | dict
    duration: timedelta
    result: Any = None
    exception: Exception | None = None
    traceback_string: str = None


@dataclasses.dataclass(frozen=True)
class TaskOutcome:
    index: int | tuple[int, ...]
    args_or_kwargs: tuple | dict
    duration: timedelta
    result: Any = None
    exception: Exception = None


@dataclasses.dataclass
class _Job:
    function: Callable
    name: str
    breadcrumbs_at_creation: tuple[str, ...]
    n_total_tasks: int = None
    start_time_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    n_completed_tasks: int = 0
    random_nonce: float = dataclasses.field(default_factory=random.random)
    np_random_nonce: float = dataclasses.field(default_factory=np.random.random)

    def run_single_task(self, index_and_args_or_kwargs):
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
        if isinstance(index, int):
            return f"{self.name}[{index}]"
        return f"{self.name}[{', '.join(str(i) for i in index)}]"
    
    def single_task_completed(self, task_index):
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
            log_message= log_message = f"{task_name} completed ({self.n_completed_tasks} / {self.n_total_tasks}; {timing_estimate})"

        _logger.info(log_message)


def _infer_n_tasks(reps, argses_to_iter, argses_to_combine):
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
def parallelize(function, reps=None, argses_to_iter=None, argses_to_combine=None, n_workers=N_WORKER_PROCESSES, reraise_exceptions=True, job_name=None, dummy=True):
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

    if multiprocessing.current_process().daemon:
        _logger.warning("Daemon process; falling back to dummy behaviour")
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
