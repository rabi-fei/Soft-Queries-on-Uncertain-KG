from abc import abstractmethod

import torch
from torch import nn

from .abstract_models import NeuralBinaryPredicate


class TransE(nn.Module, NeuralBinaryPredicate):
    def __init__(self, num_entities, num_relations, embedding_dim, p, device):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.p = p
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim, max_norm=1)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim, max_norm=1)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        board castable for the last dimension
        """
        return - torch.norm(head_emb + rel_emb - tail_emb, p=self.p, dim=-1)

    def score2prob(self, score, margin):
        return torch.sigmoid(margin + score)
