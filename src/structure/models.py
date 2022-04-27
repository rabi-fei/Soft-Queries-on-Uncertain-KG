"""
This file models the basic data types including
- KG index
- Knowledge graph
- Neural binary predictor
"""
import os
import time
from abc import abstractmethod
from collections import defaultdict
from typing import List, Tuple, Union

import torch
from torch.utils.data import DataLoader

from ..utils.data import RaggedBatch, iter_triple_from_tsv, tensorize_batch_entities
from ..utils.config import KnowledgeGraphConfig, NeuralBinaryPredicateConfig

Triple = Tuple[int, int, int]


def register(index, object):
    if object in index:
        return index[object]
    else:
        idx = len(index)
        index[object] = idx
        return idx


class KnowledgeGraph:
    """
    Fully tensorized
    """

    def __init__(self, triple_files, num_entities=None, num_relations=None, device='cpu', tensorize=False, **kwargs):
        self.device = device
        self.triples = []

        if num_entities is None:
            self.entity_index = dict()
        else:
            self.num_entities = num_entities

        if num_relations is None:
            self.relation_index = dict()
        else:
            self.num_relations = num_relations

        self.hr2t = defaultdict(list)
        self.tr2h = defaultdict(list)
        self.r2ht = defaultdict(list)
        self.ht2r = defaultdict(list)

        for h, r, t in iter_triple_from_tsv(triple_files):
            if num_entities is None:
                h = register(h, self.entity_index)
                t = register(t, self.entity_index)
            if num_relations is None:
                r = register(r, self.relation_index)

            self.triples.append((h, r, t))
            self.hr2t[(h, r)].append(t)
            self.tr2h[(t, r)].append(h)
            self.r2ht[r].append((h, t))
            self.ht2r[(h, t)].append(r)

        if num_entities is None:
            self.num_entities = len(self.entity_index)

        if num_relations is None:
            self.num_relations = len(self.relation_index)

        self.entities = list(range(self.num_entities))

        if tensorize:
            self._build_triple_tensor()

    def _build_triple_tensor(self):
        """
        Build a triple tensor of size [num_triples, 3]
            for each row, it indices head, rel, tail ids
        """
        # a tensor of shape [num_triples, 3] records the triple information
        # this tensor is used to select observed triples
        print("building the triple tensor")
        t0 = time.time()
        self.triple_tensor = torch.tensor(
            self.triples,
            dtype=torch.long,
            device=self.device)
        print("use time", time.time() - t0)

        # a sparse tensor of shape [num_entities, num_triples, num_relations]
        # also records the triple information
        # this sparse tensor is used to filter the observed triples
        print("building the triple index")
        t0 = time.time()
        self.triple_index = torch.sparse_coo_tensor(
            indices=self.triple_tensor.T,
            values=torch.ones(size=(self.triple_tensor.size(0),)),
            size=(self.num_entities, self.num_relations, self.num_entities),
            dtype=torch.long,
            device=self.device)
        print("use time", time.time() - t0)

        print("building the directed connection tensor")
        t0 = time.time()
        _dconnect_index = torch.sparse.sum(self.triple_index, dim=1).coalesce()
        self.dconnect_tensor = _dconnect_index.indices().T
        print("use time", time.time() - t0)

        print("building the directed connection index")
        t0 = time.time()
        self.dconnect_index = torch.sparse_coo_tensor(
            indices=self.dconnect_tensor.T,
            values=torch.ones(size=(self.dconnect_tensor.size(0),)),
            size=(self.num_entities, self.num_entities),
            dtype=torch.long,
            device=self.device)
        print("use time", time.time() - t0)

    @classmethod
    def create(cls, triple_files, **kwargs):
        """
        Create the class
        TO be modified when certain parameters controls the triple_file
        triple files can be a list, but they should be in the same folder
        """
        assert 'device' in kwargs
        base_dir = os.path.dirname(triple_files[0])
        if 'num_entities' not in kwargs:
            with open(os.path.join(base_dir, 'map_entity_id_to_text.tsv')) as f:
                kwargs['num_entities'] = len(f.readlines())
        if 'num_relations' not in kwargs:
            with open(os.path.join(base_dir, 'map_relation_id_to_text.tsv')) as f:
                kwargs['num_relations'] = len(f.readlines())
        return cls(triple_files, **kwargs)

    @classmethod
    def from_config(cls, config: KnowledgeGraphConfig):
        return cls.create(triple_files=config.filelist, device=config.device)

    def get_triple_dataloader(self, **kwargs):
        dataloader = DataLoader(self.triples, **kwargs)
        return dataloader

    def get_entity_mask(self, entity_tensor):
        """
        this function returns the batched multi-hot vectors
        [batch, total_entity_number]
        """
        batch_size, num_entities = entity_tensor.shape
        first_indices = torch.tile(
            torch.arange(batch_size).view(batch_size, 1),
            dims=(1, num_entities))
        entity_mask = torch.zeros(
            size=(batch_size, self.num_entities),
            dtype=torch.bool,
            device=self.device)
        # since now, the input should be tensor [batch_size, num_entities]
        entity_mask[first_indices, entity_tensor] = 1
        return entity_mask

    def get_subgraph(self,
                     entities: Union[List[int], torch.Tensor],
                     num_hops: int = 0):
        """
        Get the k-hop subgraph triples for each entity set in the batch.
            Input;
                entities: input batch of entities [batch_size, num_entities]
                num_hops: int,
                    = 0, just get the subgraph of the given batches
                    > 0, k-hop subgraphs
            Return:
                RaggedBatch of triples
        """
        entity_tensor = tensorize_batch_entities(entities)
        entity_mask = self.get_entity_mask(entity_tensor)  # [batch_size, num_entities]

        for hop in num_hops:
            neighbor_entity_mask = torch.sparse.mm(
                entity_mask.type(torch.float), self.dconnect_index)
            neighbor_entity_mask.greater_(0)
            entity_mask = torch.logical_or(entity_mask, neighbor_entity_mask)

        # so far you have a mask of shape [batch_size, total_num_entities]
        batch_triple_mask = torch.logical_and(
            entity_mask[:, self.triple_tensor[:, 0]],
            entity_mask[:, self.triple_tensor[:, 2]])
        subgraph_batch_triple_count = torch.sum(batch_triple_mask, dim=-1)
        subgraph_flat_triple_ids = batch_triple_mask.nonzero()[:, 1]
        subgraph_flat_triples = self.triple_tensor[subgraph_flat_triple_ids]

        subgraph_triples = RaggedBatch(flatten=subgraph_flat_triples,
                                       sizes=subgraph_batch_triple_count)

        return subgraph_triples

    def _get_neighbor_triples(self,
                              entities: Union[List[int], torch.Tensor],
                              reverse=False,
                              filtered=True) -> RaggedBatch:
        """
        This function finds the triples in the KG but not in the sub graph
            Input args:
                - entities: tensor [batch_size, num_entities]
                - reverse:
                    - if true, search the entities with reversed edges
                    - if false, search the entities with directed edges
                - filter:
                    - if true, exclude the entities with
            Return args:
                - RaggedBatch Triples, each batch element is a list of triples
        """
        entity_tensor = tensorize_batch_entities(entities)
        entity_mask = self.get_entity_mask(entity_tensor)
        # so far you have a mask of shape [batch_size, total_num_entities]
        if reverse:
            # find the triples given the tail entities
            batch_triple_mask = entity_mask[:, self.triple_tensor[:, 2]]
            # head entity not in the triple
            if filtered:
                batch_triple_mask = batch_triple_mask.logical_and(
                    entity_mask[:, self.triple_tensor[:, 0]].logical_not())
        else:
            # find the triples given the head entities
            batch_triple_mask = entity_mask[:, self.triple_tensor[:, 0]]
            # tail entity not in the triple
            if filtered:
                batch_triple_mask = batch_triple_mask.logical_and(
                    entity_mask[:, self.triple_tensor[:, 2]].logical_not())

        batch_triple_count = torch.sum(batch_triple_mask, dim=-1)
        flatten_triple_ids = batch_triple_mask.nonzero()[:, 1]
        flatten_triples = self.triple_tensor[flatten_triple_ids]

        return RaggedBatch(flatten_triples, batch_triple_count)

    def get_neighbor_triples_by_head(self, entities, filtered=True) -> RaggedBatch:
        return self._get_neighbor_triples(entities,
                                          reverse=False,
                                          filtered=filtered)

    def get_neighbor_triples_by_tail(self, entities, filtered=True) -> RaggedBatch:
        return self._get_neighbor_triples(entities,
                                          reverse=True,
                                          filtered=filtered)

    def _get_non_neightbor_triples(self,
                                   entities: Union[List[int], torch.Tensor],
                                   k=10,
                                   reverse=False) -> RaggedBatch:
        """
        This function constructs negative triples not in the KG with
            - head (tail) in the given entites
            - tail (head) is not connected to the head (tail) entities of each case
        Input args:
            - entities: tensor [batch_size, num_entities]
                batch entity
            - k: int
                num_negative triples constructed for each batch
            - reverse: bool
                if True, then find the head is non neighbor of the tail
                if False, then find the tail is non neighbor of the head
        Return args:
            - neg_triples: [batch_size, num_entities, k]
        """
        entity_tensor = tensorize_batch_entities(entities)
        batch_size, num_entities = entity_tensor.shape

        # [batch_size * num_entities]
        flat_entity_tensor = entity_tensor.ravel()

        if reverse:  # if the reverse is true, it considers the reversed edges
            flat_possible_targets = torch.index_select(
                self.dconnect_index.t(),
                dim=0,
                index=flat_entity_tensor).to_dense()
        else:
            flat_possible_targets = torch.index_select(
                self.dconnect_index,
                dim=0,
                index=flat_entity_tensor).to_dense()

        flat_impossible_targets = 1 - flat_possible_targets
        flat_impossible_target_dist = flat_impossible_targets / \
            flat_impossible_targets.sum(-1, keepdim=True)

        flat_neg_target = torch.multinomial(input=flat_impossible_target_dist,
                                            num_samples=k).reshape(-1, 1)
        flat_neg_source = torch.tile(flat_entity_tensor.unsqueeze(-1),
                                     dims=(1, k)).reshape(-1, 1)

        if reverse:
            flat_neg_heads, flat_neg_tails = flat_neg_target, flat_neg_source
        else:
            flat_neg_heads, flat_neg_tails = flat_neg_source, flat_neg_target

        flat_neg_rels = torch.randint(low=0, high=self.num_relations,
                                      size=flat_neg_source.shape,
                                      device=self.device)

        flat_triples = torch.concat([flat_neg_heads, flat_neg_rels, flat_neg_tails],
                                     dim=-1)
        sizes = torch.ones(batch_size, device=self.device) * num_entities * k

        return RaggedBatch(flatten=flat_triples, sizes=sizes)

    def get_non_neightbor_triples_by_head(self, entities, k) -> RaggedBatch:
        return self._get_non_neightbor_triples(entities, k=k, reverse=False)

    def get_non_neightbor_triples_by_tail(self, entities, k) -> RaggedBatch:
        return self._get_non_neightbor_triples(entities, k=k, reverse=True)


class NeuralBinaryPredicate:
    def __init__(self):
        pass

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

    def score2prob(self, score, margin):
        pass

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

    @classmethod
    def create(cls, device, **kwargs):
        obj = cls(device=device, **kwargs)
        obj = obj.to(device)
        return obj
