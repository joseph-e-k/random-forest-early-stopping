from pprint import pprint

from predestined_k_approach.Forest import Forest

if __name__ == "__main__":
    pprint(Forest.create(5, 2).analyse())
