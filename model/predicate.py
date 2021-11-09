from abc import abstractmethod
from typing import Optional

import torch
from torch import nn

class NeuralBinaryPredicate:
    def __init__(self, **kwargs):
        self.entity_embedding:   Optional[nn.Embedding] = None
        self.relation_embedding: Optional[nn.Embedding] = None
        self.kwargs = kwargs

    @abstractmethod
    @staticmethod
    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        This method computes the score for the triple given the head, tail and
        relation embedding.
        Three embeddings are in the shape [batch_size|1, embed_dim]
        The higher score means more likely to be a predicate.
        """
        pass

    def batch_pred_score(self,
                         head_id_ten: torch.Tensor,
                         rel_id_ten: torch.Tensor,
                         tail_id_ten: torch.Tensor) -> torch.Tensor:
        """
        This method computes the scores for the triple. Three tensors are in the
        same shape, i.e. [batch_size]
        It returns the same size of predicate scores.
        """
        head_emb = self.entity_embedding(head_id_ten)
        rel_emb  = self.relation_embedding(rel_id_ten)
        tail_emb = self.entity_embedding(tail_id_ten)
        return self.embedding_score(head_emb, rel_emb, tail_emb)

    def batch_find_least_score_id(self, head_id_ten, rel_id_ten, tail_id_ten):
        """
        One and only one tensor of the three input tensors is None.
        This function returns the id with the least score.
        It can be interpreted as a sentence.
        """
        # TODO
        pass


    @abstractmethod
    def compute_loss(self, head_id_ten, rel_id_ten, tail_id_ten):
        pass
