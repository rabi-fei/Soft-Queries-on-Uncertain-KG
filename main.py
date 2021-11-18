from model.kg import KG
from model.predicate import NeuralBinaryPredicate
from efl import EFL

TEST_KG_FILE = "../datasets-knowledge-embedding/WN18/edges_as_id_train.tsv"


if __name__ == "__main__":
    # test

    # create the KG
    finite_model = KG.create(TEST_KG_FILE)