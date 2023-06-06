from pprint import pprint

from predestined_k_approach.Forest import ForestWithEnvelope

if __name__ == "__main__":
    pprint(ForestWithEnvelope.create(5, 2).analyse())
