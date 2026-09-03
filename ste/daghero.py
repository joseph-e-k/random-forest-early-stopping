import numpy as np

from ste.utils.caching import memoize


@memoize()
def get_ss(N, threshold):
    ss = np.zeros(shape=(N+1, N+1), dtype=float)
    ss[-1, :] = 1  # Force last row to be ones because the ensemble has ended

    i, j = np.indices(ss.shape)
    ss[j > threshold] = 1
    ss[i - j > threshold] = 1

    return ss
