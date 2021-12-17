from typing import List
from torch import nn
import torch
from model.abstract_models import Triple, NeuralBinaryPredicate, KG

from model.model_utils import triples_to_tensors


class TransE(nn.Module, NeuralBinaryPredicate):

    criteria = nn.CrossEntropyLoss()

    def __init__(self, num_entities, num_relations, embedding_dim, device):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

    @classmethod
    def create(cls, kg: KG, embedding_dim=300, device='cpu'):
        model = cls(num_entities=kg.num_entities,
                    num_relations=kg.num_relations,
                    embedding_dim=embedding_dim,
                    device=device)
        model.to(device)
        print(model)
        return model

    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        board castable for the last dimension
        """
        return - torch.norm(torch.abs(head_emb + rel_emb - tail_emb), dim=-1)
