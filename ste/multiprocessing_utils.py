import dataclasses
import multiprocessing as mp
import os
import traceback
from typing import Callable

from ste.utils import TimerContext

_ENV_KEY_N_WORKER_PROCESSES = "STE_N_WORKER_PROCESSES"
N_WORKER_PROCESSES = int(os.getenv(_ENV_KEY_N_WORKER_PROCESSES, 32))


@dataclasses.dataclass(frozen=True)
class _CallableForWorkerProcesses:
    function: Callable

    def __call__(self, args):
        try:
            with TimerContext(verbose=False) as timer:
                result = self.function(*args)
        except Exception as e:
            traceback.print_exc()
            return args, False, e, timer.elapsed_time
        return args, True, result, timer.elapsed_time


def parallelize(function, argses):
    with mp.Pool(N_WORKER_PROCESSES) as pool:
        yield from pool.imap(_CallableForWorkerProcesses(function), argses)
