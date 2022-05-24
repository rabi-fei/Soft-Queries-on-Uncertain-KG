from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn


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
    def score2truth_value(self, score, margin):
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
    def get_head_emb(self, entity_id_or_tensor):
        pass

    @abstractmethod
    def get_tail_emb(self, entity_id_or_tensor):
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
        head_emb = self.entity_embedding(head_id_ten)
        rel_emb = self.relation_embedding(rel_id_ten)
        tail_emb = self.entity_embedding(tail_id_ten)
        return self.embedding_score(head_emb, rel_emb, tail_emb)

    @abstractmethod
    def get_all_entity_rankings(self, batch_embedding_input):
        pass

class TransE(nn.Module, NeuralBinaryPredicate):
    def __init__(self, num_entities, num_relations, embedding_dim, p, margin, device, **kwargs):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.margin = margin
        self.p = p
        self._entity_embedding = nn.Embedding(num_entities, embedding_dim, max_norm=1)
        nn.init.xavier_uniform_(self._entity_embedding.weight)
        self._relation_embedding = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self._relation_embedding.weight)

    @property
    def entity_embedding(self):
        return self._entity_embedding.weight

    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        board castable for the last dimension
        """
        return - torch.norm(head_emb + rel_emb - tail_emb, p=self.p, dim=-1)

    def score2truth_value(self, score, margin, scale=1):
        return torch.sigmoid(margin + score * scale)

    def estimate_tail_emb(self, head_emb, rel_emb):
        return head_emb + rel_emb

    def estimate_head_emb(self, tail_emb, rel_emb):
        return tail_emb - rel_emb

    def estiamte_rel_emb(self, head_emb, tail_emb):
        return tail_emb - head_emb

    def get_relation_emb(self, relation_id_or_tensor):
        rel_id = torch.tensor(relation_id_or_tensor, device=self.device)
        return self._relation_embedding(rel_id)

    def get_head_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)

    def get_tail_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)

    def get_all_entity_rankings(self, batch_embedding_input):
        batch_embedding_input = batch_embedding_input.unsqueeze(-2)
        # batch_size, all_candidates
        # ranking score should be the higher the better
        # ranking_score[entity_id] = the score of {entity_id}
        ranking_score = - torch.norm(batch_embedding_input - self.entity_embedding, p=self.p, dim=-1)
        # ranked_entity_ids[ranking] = {entity_id} at the {rankings}-th place
        ranked_entity_ids = torch.argsort(ranking_score, dim=-1, descending=True)
        # entity_rankings[entity_id] = {rankings} of the entity
        entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
        return entity_rankings


# from facebook KBC
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# from abc import ABC, abstractmethod
# from typing import Tuple, List, Dict
# import torch
# from torch import nn


# class KBCModel(nn.Module, ABC):
#     @abstractmethod
#     def get_rhs(self, chunk_begin: int, chunk_size: int):
#         pass

#     @abstractmethod
#     def get_queries(self, queries: torch.Tensor):
#         pass

#     @abstractmethod
#     def score(self, x: torch.Tensor):
#         pass

#     def get_ranking(
#             self, queries: torch.Tensor,
#             filters: Dict[Tuple[int, int], List[int]],
#             batch_size: int = 1000, chunk_size: int = -1
#     ):
#         """
#         Returns filtered ranking for each queries.
#         :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
#         :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
#         :param batch_size: maximum number of queries processed at once
#         :param chunk_size: maximum number of candidates processed at once
#         :return:
#         """
#         if chunk_size < 0:
#             chunk_size = self.sizes[2]
#         ranks = torch.ones(len(queries))
#         with torch.no_grad():
#             c_begin = 0
#             while c_begin < self.sizes[2]:
#                 b_begin = 0
#                 rhs = self.get_rhs(c_begin, chunk_size)
#                 while b_begin < len(queries):
#                     these_queries = queries[b_begin:b_begin + batch_size]
#                     q = self.get_queries(these_queries)

#                     scores = q @ rhs
#                     targets = self.score(these_queries)

#                     # set filtered and true scores to -1e6 to be ignored
#                     # take care that scores are chunked
#                     for i, query in enumerate(these_queries):
#                         filter_out = filters[(query[0].item(), query[1].item())]
#                         filter_out += [queries[b_begin + i, 2].item()]
#                         if chunk_size < self.sizes[2]:
#                             filter_in_chunk = [
#                                 int(x - c_begin) for x in filter_out
#                                 if c_begin <= x < c_begin + chunk_size
#                             ]
#                             scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
#                         else:
#                             scores[i, torch.LongTensor(filter_out)] = -1e6
#                     ranks[b_begin:b_begin + batch_size] += torch.sum(
#                         (scores >= targets).float(), dim=1
#                     ).cpu()

#                     b_begin += batch_size

#                 c_begin += chunk_size
#         return ranks


# class CP(KBCModel):
#     def __init__(
#             self, sizes: Tuple[int, int, int], rank: int,
#             init_size: float = 1e-3
#     ):
#         super(CP, self).__init__()
#         self.sizes = sizes
#         self.rank = rank

#         self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
#         self.rel = nn.Embedding(sizes[1], rank, sparse=True)
#         self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

#         self.lhs.weight.data *= init_size
#         self.rel.weight.data *= init_size
#         self.rhs.weight.data *= init_size

#     def score(self, x):
#         lhs = self.lhs(x[:, 0])
#         rel = self.rel(x[:, 1])
#         rhs = self.rhs(x[:, 2])

#         return torch.sum(lhs * rel * rhs, 1, keepdim=True)

#     def forward(self, x):
#         lhs = self.lhs(x[:, 0])
#         rel = self.rel(x[:, 1])
#         rhs = self.rhs(x[:, 2])
#         return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

#     def get_rhs(self, chunk_begin: int, chunk_size: int):
#         return self.rhs.weight.data[
#             chunk_begin:chunk_begin + chunk_size
#         ].transpose(0, 1)

#     def get_queries(self, queries: torch.Tensor):
#         return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data


class ComplEx(NeuralBinaryPredicate, nn.Module):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 margin: float = 0,
                 init_size: float = 1e-3,
                 device = 'cpu', **kwargs):
        super(ComplEx, self).__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.rank = embedding_dim
        self.device = device
        self.margin = margin

        self._entity_embedding = nn.Embedding(num_entities, 2*embedding_dim)
        self._entity_embedding.weight.data *= init_size

        self._relation_embedding = nn.Embedding(num_relations, 2*embedding_dim)
        self._relation_embedding.weight.data *= init_size

    @property
    def entity_embedding(self):
        return self._entity_embedding.weight


    def embedding_score(self, head_emb, rel_emb, tail_emb):
        lhs = head_emb[..., :self.rank], head_emb[..., self.rank:]
        rel = rel_emb[..., :self.rank],  rel_emb[..., self.rank:]
        rhs = tail_emb[..., :self.rank], tail_emb[..., self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            dim=-1)

    def score2truth_value(self, score, margin):
        return torch.sigmoid(score)

    def estimate_tail_emb(self, head_emb, rel_emb):
        lhs = head_emb[:, :self.rank], head_emb[:, self.rank:]
        rel = rel_emb[:, :self.rank], rel_emb[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def estimate_head_emb(self, tail_emb, rel_emb):
        rhs = tail_emb[:, :self.rank], tail_emb[:, self.rank:]
        rel = rel_emb[:, :self.rank], rel_emb[:, self.rank:]

        return torch.cat([
            rhs[0] * rel[0] + rhs[1] * rel[1],
            rhs[0] * rel[1] - rhs[1] * rel[0]
        ], 1)

    def estiamte_rel_emb(self, head_emb, tail_emb):
        lhs = head_emb[:, :self.rank], head_emb[:, self.rank:]
        rhs = tail_emb[:, :self.rank], tail_emb[:, self.rank:]

        return torch.cat([
            lhs[0] * rhs[0] + lhs[1] * rhs[1],
            lhs[0] * rhs[1] - lhs[1] * rhs[0]
        ], 1)

    def get_relation_emb(self, relation_id_or_tensor):
        rel_id = torch.tensor(relation_id_or_tensor, device=self.device)
        return self._relation_embedding(rel_id)

    def get_head_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)

    def get_tail_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)

    def get_all_entity_rankings(self, batch_embedding_input):
        batch_embedding_input = batch_embedding_input.unsqueeze(-2)
        # batch_size, all_candidates
        # ranking score should be the higher the better
        # ranking_score[entity_id] = the score of {entity_id}
        ranking_score = - torch.norm(batch_embedding_input - self.entity_embedding, dim=-1)
        # ranked_entity_ids[ranking] = {entity_id} at the {rankings}-th place
        ranked_entity_ids = torch.argsort(ranking_score, dim=-1, descending=True)
        # entity_rankings[entity_id] = {rankings} of the entity
        entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
        return entity_rankings

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def get_random_entity_embed(self, batch_size):
        return torch.normal(0, 1e-3, (batch_size, self.rank * 2), device=self.device, requires_grad=True)