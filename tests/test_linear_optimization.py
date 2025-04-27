import numpy as np

from ste.optimization import *

def test_minimax_sanity():
    computed_oss = get_optimal_stopping_strategy(5, 0.05)
    expected_oss = np.array([
        [0. , 1. , 1. , 1. , 1. , 1. ],
        [0. , 0. , 1. , 1. , 1. , 1. ],
        [0.5, 0. , 0.5, 1. , 1. , 1. ],
        [1. , 0. , 0. , 1. , 1. , 1. ],
        [1. , 1. , 0. , 1. , 1. , 1. ],
        [1. , 1. , 1. , 1. , 1. , 1. ]
    ])
    assert np.allclose(np.tril(computed_oss), np.tril(expected_oss))