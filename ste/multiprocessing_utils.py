import dataclasses
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Mapping
from ctypes import c_uint
from datetime import datetime, timedelta
from typing import Callable

from ste.utils import TimerContext


_ENV_KEY_N_WORKER_PROCESSES = "STE_N_WORKER_PROCESSES"
N_WORKER_PROCESSES = int(os.getenv(_ENV_KEY_N_WORKER_PROCESSES, 32))


class SharedValue:
    def __init__(self, manager, ctype):
        self.value = manager.Value(ctype, 0)
        self.lock = manager.RLock()

    def change(self, function):
        with self.lock:
            new_value = function(self.value.get())
            self.value.set(new_value)
            return new_value
        
    def get_value(self):
        return self.value.get_value()


class SharedCounter(SharedValue):
    def __init__(self, manager):
        super().__init__(manager, c_uint)

    def increment(self):
        return self.change(lambda x: x + 1)


@dataclasses.dataclass(frozen=True)
class _CallableForWorkerProcesses:
    function: Callable
    job_start_time_ns: int
    counter: SharedCounter = None
    n_total_tasks: int = None

    def __call__(self, args_or_kwargs):
        try:
            with TimerContext(verbose=False) as timer:
                if isinstance(args_or_kwargs, Mapping):
                    result = self.function(**args_or_kwargs)
                else:
                    result = self.function(*args_or_kwargs)
        except Exception as e:
            traceback.print_exc()
            return args_or_kwargs, False, e, timer.elapsed_time
        else:
            return args_or_kwargs, True, result, timer.elapsed_time
        finally:
            self.finish()
    
    def finish(self):
        if self.counter is not None:
            n_completed_tasks = self.counter.increment()
            timestamp = datetime.utcnow().isoformat()
            message = f"Completed {n_completed_tasks} tasks"
            if self.n_total_tasks is not None:
                message += f" out of {self.n_total_tasks}"
                ns_so_far = time.monotonic_ns() - self.job_start_time_ns
                ns_per_task = ns_so_far / n_completed_tasks
                n_remaining_tasks = self.n_total_tasks - n_completed_tasks
                expected_time_remaining_ns = ns_per_task * n_remaining_tasks
                expected_time_remaining = timedelta(microseconds=expected_time_remaining_ns // 1e3)
                expected_end_time = datetime.utcnow() + expected_time_remaining
                message += f" (expect to be finished at {expected_end_time.isoformat()})"

            print(f"{timestamp}: {message}")


def parallelize(function, iter_argses, fixed_args=(), n_workers=N_WORKER_PROCESSES, verbose=False, n_tasks=None):
    manager = mp.Manager()
    counter = SharedCounter(manager) if verbose else None

    with mp.Pool(n_workers) as pool:
        yield from pool.imap(
            _CallableForWorkerProcesses(function, time.monotonic_ns(), counter, n_tasks),
            (iter_args + fixed_args for iter_args in iter_argses)
        )
