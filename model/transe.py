from typing import List
from torch import nn
import torch
from model.kg import Triple

from .predicate import NeuralBinaryPredicate
from .kg import KG
from .model_utils import triples_to_tensors


class TransE(NeuralBinaryPredicate):

    criteria = nn.CrossEntropyLoss()

    def __init__(self, num_entities, num_relations, embedding_dim, device='cpu'):
        super(TransE, self).__init__(
            entity_embedding=nn.Embedding(
                num_embeddings=num_entities, embedding_dim=embedding_dim),
            relation_embedding=nn.Embedding(
                num_embeddings=num_relations, embedding_dim=embedding_dim),
            device=device)

    @classmethod
    def create(cls, kg: KG, embedding_dim=300):
        return cls(num_entities=kg.num_entities,
                   num_relations=kg.num_relations,
                   embedding_dim=embedding_dim)

    def embedding_score(self, head_emb, rel_emb, tail_emb):
        return - torch.norm(torch.abs(head_emb + rel_emb - tail_emb), dim=-1)

    def compute_triple_loss(self, triples: List[Triple], labels: List[int]):
        """
        compute the loss to learn the neural model
        """
        head, rel, tail = triples_to_tensors(triples)
        tv_tensor = torch.tensor(labels, device=self.device)
        scores = torch.sigmoid(self.embedding_score(head, rel, tail))
        return self.criteria(scores, tv_tensor)
