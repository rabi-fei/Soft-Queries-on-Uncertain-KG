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


class NeuralBinaryPredicate(nn.Module):
    def __init__(self, **kwargs):
        super(NeuralBinaryPredicate, self).__init__()
        self.entity_embedding:   Optional[nn.Embedding] = None
        self.relation_embedding: Optional[nn.Embedding] = None
        self.device = None
        self.kwargs = kwargs

    @abstractmethod
    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        This method computes the score for the triple given the head, tail and
        relation embedding. The higher score means more likely to be a predicate.
        Inputs:
            Three embeddings are in the shape [batch_size|1, embed_dim]
        Returns:
            The tensor of scores in the shape [batch_size|1]
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
        rel_emb = self.relation_embedding(rel_id_ten)
        tail_emb = self.entity_embedding(tail_id_ten)
        return self.embedding_score(head_emb, rel_emb, tail_emb)

    def sort_triples_by_scores(self, triples, assending=True, k=1):
        """
        One and only one tensor of the three input tensors is None.
        This function returns the id with the least score.
        It can be interpreted as a sentence.
        """
        h, r, t = triple_to_tensors(triples, self.device)
        triple_score_tensor = self.batch_pred_score(h, r, t)
        triple_scores = triple_score_tensor.cpu().numpy().tolist()
        triple_with_score = sorted(zip(triples, triple_scores),
                                   reverse=not assending,
                                   key=lambda x: x[1])
        ret = []
        for triple, score in triple_with_score:
            ret.append(triple)
        return ret

    @abstractmethod
    def compute_loss(self, head_id_ten, rel_id_ten, tail_id_ten):
        pass
