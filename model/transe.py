from typing import List
from torch import nn
import torch
from model.kg import Triple

from model.predicate import NeuralBinaryPredicate
from model.kg import KG
from model.model_utils import triples_to_tensors


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

    def compute_triple_loss(self,
                            pos_triples: List[Triple],
                            neg_triples: List[Triple],
                            pairwise_loss=True):
        """
        compute the loss to learn the neural model
        """
        phead, prel, ptail = triples_to_tensors(pos_triples, self.device)
        nhead, nrel, ntail = triples_to_tensors(neg_triples, self.device)

        head = self.entity_embedding(torch.cat([phead, nhead]))
        rel = self.relation_embedding(torch.cat([prel, nrel]))
        tail = self.entity_embedding(torch.cat([ptail, ntail]))

        scores = self.embedding_score(head, rel, tail)

        if pairwise_loss:
            pos_scores = scores[:len(pos_triples)]
            neg_scores = scores[len(pos_triples):]
            loss = torch.relu(neg_scores - pos_scores + 10).mean()
            return loss

        labels = torch.tensor([1] * len(pos_triples) + [0] * len(neg_triples))

        tv_tensor = torch.tensor(labels, device=self.device)
        return self.criteria(scores, tv_tensor)
