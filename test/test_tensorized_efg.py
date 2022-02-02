import sys

from torch.functional import Tensor
sys.path.append('/home/zwanggc/project/EFG-KG-FOS-Verification')

import torch

from src.model.abstract_models import KG
from src.model.transe import TransE
from src.learner.elementary import BatchedEFG



if __name__ == '__main__':
    device='cuda:1'
    kg = KG.create(triple_file='data/family-loss-0.1/train.tsv', device=device)

    batch_size = 3
    num_entities = 5
    entities = torch.randint(low=0, high=1000, size=(batch_size, num_entities), device=device)

    nbp = TransE.create(kg, device=device)
    tefg = BatchedEFG(kg, nbp)
    tefg.play(torch.randint(0, 1000, (128,)))