import dataclasses
import multiprocessing as mp
import os
import traceback
from collections.abc import Mapping
from typing import Callable

from ste.utils import TimerContext

_ENV_KEY_N_WORKER_PROCESSES = "STE_N_WORKER_PROCESSES"
N_WORKER_PROCESSES = int(os.getenv(_ENV_KEY_N_WORKER_PROCESSES, 32))


@dataclasses.dataclass(frozen=True)
class _CallableForWorkerProcesses:
    function: Callable

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
        return args_or_kwargs, True, result, timer.elapsed_time


def parallelize(function, argses):
    with mp.Pool(N_WORKER_PROCESSES) as pool:
        yield from pool.imap(_CallableForWorkerProcesses(function), argses)
