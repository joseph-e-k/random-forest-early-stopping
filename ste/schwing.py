import numpy as np
from scipy.special import hyp2f1
from scipy.special import factorial

from ste.utils.caching import memoize
from ste.utils.multiprocessing import parallelize_to_array


@memoize()
def compute_prob_under_null(i, j):
    K = i
    k1 = j

    return 1 - \
        (factorial(K + 1) * 0.5**(k1+1)) / (factorial(k1 + 1) * factorial(K - k1)) \
            * hyp2f1(k1 + 1, k1 - K, k1 + 2, 0.5)


def compute_probs_under_null(N):
    return parallelize_to_array(
        compute_prob_under_null,
        argses_to_combine=[list(range(N + 1)), list(range(N + 1))]
    )


@memoize()
def get_ss(N, alpha):
    probs = compute_probs_under_null(N)
    ss = np.logical_or(probs < alpha, probs > (1 - alpha))
    ss[-1, :] = True  # Force last row to be ones because the ensemble has ended
    return np.asarray(ss, dtype=float)


if __name__ == "__main__":
    print(get_ss(10, 0.05))
