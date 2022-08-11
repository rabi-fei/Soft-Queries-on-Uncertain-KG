from abc import abstractmethod

import torch
from torch import embedding, nn


class NeuralBinaryPredicate:
    num_entities: int
    num_relations: int
    device: torch.device
    margin: float

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
    def score2truth_value(self, score):
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

    @abstractmethod
    def get_relation_emb(self, relation_id_or_tensor):
        rel_id = torch.tensor(relation_id_or_tensor, device=self.device)
        return self._relation_embedding(rel_id)

    @abstractmethod
    def get_entity_emb(self, entity_id_or_tensor):
        pass

    @abstractmethod
    def get_random_entity_embed(self, batch_size):
        pass

    @property
    def entity_embedding(self):
        pass

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
        head_emb = self._entity_embedding(head_id_ten)
        rel_emb = self._relation_embedding(rel_id_ten)
        tail_emb = self._entity_embedding(tail_id_ten)
        return self.embedding_score(head_emb, rel_emb, tail_emb)

    @abstractmethod
    def get_all_entity_rankings(self, batch_embedding_input):
        pass
