import dataclasses
import datetime
import multiprocessing as mp
import os
import traceback
from collections.abc import Mapping
from ctypes import c_uint
from typing import Callable

from ste.utils import TimerContext

_ENV_KEY_N_WORKER_PROCESSES = "STE_N_WORKER_PROCESSES"
N_WORKER_PROCESSES = int(os.getenv(_ENV_KEY_N_WORKER_PROCESSES, 32))


class Counter:
    def __init__(self, manager):
        self.value = manager.Value(c_uint, 0)
        self.lock = manager.RLock()

    def increment(self):
        with self.lock:
            new_value = self.value.get() + 1
            self.value.set(new_value)
            return new_value


@dataclasses.dataclass(frozen=True)
class _CallableForWorkerProcesses:
    function: Callable
    counter: Counter = None

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
            n_completed = self.counter.increment()
            print(f"{datetime.datetime.utcnow().isoformat()}:  Completed {n_completed} tasks")



def parallelize(function, iter_argses, fixed_args=(), n_workers=N_WORKER_PROCESSES, verbose=False):
    manager = mp.Manager()
    counter = Counter(manager) if verbose else None

    with mp.Pool(n_workers) as pool:
        yield from pool.imap(
            _CallableForWorkerProcesses(function, counter),
            (iter_args + fixed_args for iter_args in iter_argses)
        )
