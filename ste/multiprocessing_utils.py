import dataclasses
import functools
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import tblib

from ste.logging_utils import get_module_logger
from ste.utils import TimerContext, enumerate_product


_logger = get_module_logger()


_ENV_KEY_N_WORKER_PROCESSES = "STE_N_WORKER_PROCESSES"
N_WORKER_PROCESSES = int(os.getenv(_ENV_KEY_N_WORKER_PROCESSES, 32))


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
    start_time_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    n_completed_tasks: int = 0

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
        self.n_completed_tasks += 1
        
        now = datetime.now().astimezone(timezone.utc)
        log_message = f"Completed {self.n_completed_tasks} tasks"
        if self.n_total_tasks is not None:
            log_message += f" out of {self.n_total_tasks}"
            ns_so_far = time.monotonic_ns() - self.start_time_ns
            ns_per_task = ns_so_far / self.n_completed_tasks
            n_remaining_tasks = self.n_total_tasks - self.n_completed_tasks
            expected_time_remaining_ns = ns_per_task * n_remaining_tasks
            expected_time_remaining = timedelta(microseconds=expected_time_remaining_ns // 1e3)
            expected_end_time = now + expected_time_remaining
            log_message += f" (expect to be finished at {expected_end_time.strftime('%Y-%m-%dT%H:%M:%S')})"

        _logger.info(log_message)


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


def parallelize(function, argses_to_iter=None, argses_to_combine=None, n_workers=N_WORKER_PROCESSES, n_tasks=None, reraise_exceptions=True):
    if not ((argses_to_iter is None) ^ (argses_to_combine is None)):
        raise TypeError("argses_to_iter or argses_to_multiply must be specified (but not both)")
    
    if isinstance(function, functools.partial):
        name = function.func.__name__
    else:
        name = getattr(function, "__name__", "<function name unknown>")
    _logger.info(f"Preparing task pool for {name}")
    
    if argses_to_iter is None:
        indices_and_argses = enumerate_product(*argses_to_combine)
    else:
        indices_and_argses = enumerate(argses_to_iter)

    job = _Job(function, n_tasks)

    with mp.Pool(n_workers) as pool:
        for raw_outcome in pool.imap_unordered(job, indices_and_argses):
            outcome = _process_raw_task_outcome(job, raw_outcome, reraise_exceptions)
            job.finish_one_task()
            yield outcome
