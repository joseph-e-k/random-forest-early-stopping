import sys
from pprint import pprint

from predestined_k_approach.Forest import ForestWithEnvelope, Forest

if __name__ == "__main__":
    sys.setrecursionlimit(2000)
    forest = Forest(100, 51)
    # forest_with_envelope = ForestWithEnvelope(forest, forest.get_null_envelope())
    boundary = forest.get_optimal_lower_boundary(0.05, verbose=True)
