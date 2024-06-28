import dataclasses
import functools
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Mapping
from ctypes import c_uint
from datetime import datetime, timedelta
from typing import Callable

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
class _CallableForWorkerProcesses:
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
            traceback.print_exc()
            print(f"{index=}, {len(args_or_kwargs)=}, {args_or_kwargs=}")
            return index, args_or_kwargs, False, e, timer.elapsed_time
        else:
            return index, args_or_kwargs, True, result, timer.elapsed_time
        finally:
            self.finish()
    
    def finish(self):
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


def parallelize(function, argses_to_iter=None, argses_to_combine=None, n_workers=N_WORKER_PROCESSES, verbose=False, n_tasks=None):
    if not ((argses_to_iter is None) ^ (argses_to_combine is None)):
        raise TypeError("argses_to_iter or argses_to_multiply must be specified (but not both)")
    
    if verbose:
        if isinstance(function, functools.partial):
            name = function.func.__name__
        else:
            name = getattr(function, "name", "<function name unknown>")
        print(f"{datetime.utcnow().isoformat()}: preparing task pool for {name}")
    
    if argses_to_iter is None:
        indices_and_argses = enumerate_product(*argses_to_combine)
    else:
        indices_and_argses = enumerate(argses_to_iter)

    worker = _CallableForWorkerProcesses(function, n_tasks, verbose)

    with mp.Pool(n_workers) as pool:
        yield from pool.imap(
            worker,
            indices_and_argses
        )
