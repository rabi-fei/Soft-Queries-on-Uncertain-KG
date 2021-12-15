import sys

from torch.functional import Tensor
sys.path.append('/home/zwanggc/project/EFG-KG-FOS-Verification')

import torch

from model.abstract_models import KG
from model.transe import TransE
from learner.efl import TensorizedEFG



if __name__ == '__main__':
    device='cuda:1'
    kg = KG.create(triple_file='data/family-loss-0.1/train.tsv', device=device)

    batch_size = 3
    num_entities = 5
    entities = torch.randint(low=0, high=1000, size=(batch_size, num_entities), device=device)

    kg.get_non_neightbor_new_tail(entities)
    nbp = TransE.create(kg, device=device)
    tefg = TensorizedEFG(kg, nbp)
    tefg.play(torch.randint(0, 1000, (128,)))