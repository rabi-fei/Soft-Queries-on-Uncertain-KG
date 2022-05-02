from abc import abstractmethod

import torch
from torch import nn


class NeuralBinaryPredicate:
    num_entities: int
    num_relations: int
    device: torch.device

    @abstractmethod
    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        This method computes the score for the triple given the head, tail and
        relation embedding. The higher score means more likely to be a predicate.
        Inputs:
            Three embeddings are in the shape [..., embed_dim]
        Returns:
            The tensor of scores in the shape [...]
        """
        pass

    @abstractmethod
    def score2prob(self, score, margin):
        pass

    @abstractmethod
    def estimate_tail_emb(self, head_emb, rel_emb):
        pass

    @abstractmethod
    def estimate_head_emb(self, tail_emb, rel_emb):
        pass

    @abstractmethod
    def estiamte_rel_emb(self, head_emb, tail_emb):
        pass

    def get_relation_emb(self, relation_id_or_tensor):
        rel_id = torch.tensor(relation_id_or_tensor, device=self.device)
        return self.relation_embedding(rel_id)

    def get_entity_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self.entity_embedding(ent_id)


    @classmethod
    def create(cls, device, **kwargs):
        obj = cls(device=device, **kwargs)
        obj = obj.to(device)
        return obj

    def batch_predicate_score(self,
                              triple_tensor: torch.Tensor) -> torch.Tensor:
        """
        This method computes the scores for the triple. triple tensors the
        shape of [..., 3]
        It returns the same size of predicate scores.
        """
        if isinstance(triple_tensor, list):
            assert len(triple_tensor) == 3
            head_id_ten, rel_id_ten, tail_id_ten = triple_tensor
        else:
            head_id_ten, rel_id_ten, tail_id_ten = torch.split(
                triple_tensor, 1, dim=-1)
        head_emb = self.entity_embedding(head_id_ten)
        rel_emb = self.relation_embedding(rel_id_ten)
        tail_emb = self.entity_embedding(tail_id_ten)
        return self.embedding_score(head_emb, rel_emb, tail_emb)


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
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        board castable for the last dimension
        """
        return - torch.norm(head_emb + rel_emb - tail_emb, p=self.p, dim=-1)

    def score2prob(self, score, margin, scale=1):
        return torch.sigmoid(margin + score * scale)

    def estimate_tail_emb(self, head_emb, rel_emb):
        return head_emb + rel_emb

    def estimate_head_emb(self, tail_emb, rel_emb):
        return tail_emb - rel_emb

    def estiamte_rel_emb(self, head_emb, tail_emb):
        return tail_emb - head_emb
