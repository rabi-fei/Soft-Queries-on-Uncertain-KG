from abc import abstractmethod
from typing import Optional

import torch
from torch import nn



 def triple_to_tensors(triples):
    H, R, T = [], [], []
    for h, r, t in triples:
        H.append(h)
        R.append(r)
        T.appned(t)
    return torch.tensor(H), torch.tensor(R), torch.tensor(T)


class NeuralBinaryPredicate:
    def __init__(self, **kwargs):
        self.entity_embedding:   Optional[nn.Embedding] = None
        self.relation_embedding: Optional[nn.Embedding] = None
        self.device = None
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

    def find_low_score_true_triples(self, neighbering_graph_triples, k=1):
        """
        One and only one tensor of the three input tensors is None.
        This function returns the id with the least score.
        It can be interpreted as a sentence.
        """
        h, r, t = triple_to_tensors(neighbering_graph_triples, self.device)
        triple_scores = self.batch_pred_score(h, r, t)


    @abstractmethod
    def compute_loss(self, head_id_ten, rel_id_ten, tail_id_ten):
        pass
