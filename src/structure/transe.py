from torch import nn
import torch
from src.structure.abstract_models import NeuralBinaryPredicate, KnowledgeGraph


class TransE(nn.Module, NeuralBinaryPredicate):
    def __init__(self, num_entities, num_relations, embedding_dim, device):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

    @classmethod
    def create(cls, kg: KnowledgeGraph, embedding_dim=300, device='cpu'):
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

    def score2prob(score, margin):
        return torch.sigmoid(margin + score)
