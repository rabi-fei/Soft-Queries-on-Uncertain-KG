from model import KG, TransE, NeuralBinaryPredicate

from efl import EFL
import torch


TEST_KG_FILE = "data/family-full.tsv"


if __name__ == "__main__":
    # create the KG
    finite_model = KG.create(TEST_KG_FILE)

    # create the neural model
    neural_model = TransE.create(finite_model)

    # create the optimizer
    optimizer = torch.optim.Adam(neural_model.parameters())

    # create the EFL
    efl = EFL(finite_model, neural_model)
    for _ in range(10):
        efl.learning_step(1, optimizer)
