import os
import pickle
import re


from ste.utils.caching import default_cache
from ste.theoretical_performance import get_minimax_ss, get_minimean_flat_ss, get_minimixed_flat_ss


PRECOMPUTED_SS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "precomputed_stopping_strategies"
)


FLOAT_RE = r"\d+(?:\.\d+)?(?:e[+-]\d+)?"
INT_RE = r"\d+"
FILE_NAME_RE = r"(minimax|minimixed_flat|minimean_flat)_ss_N_(" + INT_RE + r")_alpha_(" + FLOAT_RE + r")_\d{14}\.pickle"


def main():
    for file_name in os.listdir(PRECOMPUTED_SS_DIR):
        file_path = os.path.join(PRECOMPUTED_SS_DIR, file_name)
        if not (m := re.match(FILE_NAME_RE, file_name)):
            continue

        print(file_name)

        n_trees = int(m.group(2))
        adr = float(m.group(3))

        method = {
            "minimax": get_minimax_ss,
            "minimixed_flat": get_minimixed_flat_ss,
            "minimean_flat": get_minimean_flat_ss
        }[m.group(1)]

        with open(file_path, "rb") as file:
            result = pickle.load(file)

        key = method.__cache_key__(
            n_trees=n_trees,
            adr=adr,
            estimated_smopdis=None
        )

        default_cache.set(key, result)


if __name__ == "__main__":
    main()
