import dataclasses
import time

import numpy as np


def shift_array(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


@dataclasses.dataclass
class TimerContext:
    tag: str = None
    verbose: bool = True
    start_time: float = None
    end_time: float = None

    @property
    def elapsed_time(self):
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

        if self.verbose:
            message = f"Time ({self.tag})" if self.tag is not None else "Time"
            message += f": {self.elapsed_time}s"
            print(message)
