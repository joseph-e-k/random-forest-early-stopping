import dataclasses
import functools
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Mapping
from ctypes import c_uint
from datetime import datetime, timedelta
from typing import Any, Callable

import tblib

from ste.utils import TimerContext, enumerate_product


_ENV_KEY_N_WORKER_PROCESSES = "STE_N_WORKER_PROCESSES"
N_WORKER_PROCESSES = int(os.getenv(_ENV_KEY_N_WORKER_PROCESSES, 32))


class SharedValue:
    def __init__(self, manager, ctype, initial_value):
        self.value = manager.Value(ctype, initial_value)
        self.lock = manager.RLock()

    def change(self, function):
        with self.lock:
            new_value = function(self.value.get())
            self.value.set(new_value)
            return new_value
        
    def get_value(self):
        return self.value.get()


class SharedCounter(SharedValue):
    def __init__(self, manager):
        super().__init__(manager, c_uint, 0)

    def increment(self):
        return self.change(lambda x: x + 1)

@dataclasses.dataclass
class _RawTaskOutcome:
    index: int | tuple[int, ...]
    args_or_kwargs: tuple | dict
    duration: timedelta
    result: Any = None
    exception: Exception = None
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
    n_total_tasks: int = None
    verbose: bool = False
    job_start_time_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    counter: SharedCounter = dataclasses.field(default_factory=lambda: SharedCounter(mp.Manager()))

    def __call__(self, index_and_args_or_kwargs):
        index, args_or_kwargs = index_and_args_or_kwargs
        try:
            with TimerContext(verbose=False) as timer:
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
    
    def finish_one_task(self):
        n_completed_tasks = self.counter.increment()

        if not self.verbose:
            return
        
        now = datetime.utcnow()
        timestamp = now.isoformat()
        message = f"Completed {n_completed_tasks} tasks"
        if self.n_total_tasks is not None:
            message += f" out of {self.n_total_tasks}"
            ns_so_far = time.monotonic_ns() - self.job_start_time_ns
            ns_per_task = ns_so_far / n_completed_tasks
            n_remaining_tasks = self.n_total_tasks - n_completed_tasks
            expected_time_remaining_ns = ns_per_task * n_remaining_tasks
            expected_time_remaining = timedelta(microseconds=expected_time_remaining_ns // 1e3)
            expected_end_time = now + expected_time_remaining
            message += f" (expect to be finished at {expected_end_time.isoformat()})"

        print(f"{timestamp}: {message}")


def _process_raw_task_outcome(task: _Job, raw_outcome: _RawTaskOutcome, reraise_exceptions: bool) -> TaskOutcome:
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


def parallelize(function, argses_to_iter=None, argses_to_combine=None, n_workers=N_WORKER_PROCESSES, verbose=False, n_tasks=None, reraise_exceptions=True):
    if not ((argses_to_iter is None) ^ (argses_to_combine is None)):
        raise TypeError("argses_to_iter or argses_to_multiply must be specified (but not both)")
    
    if verbose:
        if isinstance(function, functools.partial):
            name = function.func.__name__
        else:
            name = getattr(function, "__name__", "<function name unknown>")
        print(f"{datetime.utcnow().isoformat()}: preparing task pool for {name}")
    
    if argses_to_iter is None:
        indices_and_argses = enumerate_product(*argses_to_combine)
    else:
        indices_and_argses = enumerate(argses_to_iter)

    job = _Job(function, n_tasks, verbose)

    with mp.Pool(n_workers) as pool:
        for raw_outcome in pool.imap_unordered(job, indices_and_argses):
            outcome = _process_raw_task_outcome(job, raw_outcome, reraise_exceptions)
            job.finish_one_task()
            yield outcome
