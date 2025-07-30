import numpy as np


def get_envelope(ss):
    envelope = np.full(shape=ss.shape[0], fill_value=0, dtype=int)

    for i in range(ss.shape[0]):
        nonzero_entries = np.nonzero(ss[i])[0]
        try:
            envelope[i] = min(j for j in nonzero_entries if j > i / 2)
        except ValueError:
            envelope[i] = i + 1
    
    return envelope
