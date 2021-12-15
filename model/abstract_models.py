import os
import time
import random
from abc import abstractmethod
from collections import defaultdict
from typing import List, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model.model_utils import triples_to_tensors

Triple = Tuple[int, int, int]


def iter_triple_from_tsv(triple_file):
    with open(triple_file, 'rt') as f:
        for line in f.readlines():
            tp = line.strip().split()
            assert len(tp) == 3
            triple = [int(t) for t in tp]
            yield triple


class KG:
    """
    Fully tensorized
    """

    def __init__(self, triple_file, num_entities=None, num_relations=None, device='cpu', **kwargs):
        self.device = device
        self.entity_set = set()
        self.ht2r = defaultdict(list)
        self.r2ht = defaultdict(list)

        self.h2t = defaultdict(list)
        self.t2h = defaultdict(list)

        self.h2r2t = defaultdict(dict)
        self.t2r2h = defaultdict(dict)
        self.triples = []

        self.tensor = None

        for h, r, t in iter_triple_from_tsv(triple_file):
            self.triples.append((h, r, t))

        self._build_index_by_triples()

        self.num_entities = len(
            self.entity_set) if num_entities is None else num_entities
        self.num_relations = len(
            self.r2ht) if num_relations is None else num_relations

        self._build_triple_tensor()

    # FIXME: this part might not be necessary
    def _build_index_by_triples(self):
        for h, r, t in self.triples:
            if h not in self.entity_set:
                self.entity_set.add(h)
            if t not in self.entity_set:
                self.entity_set.add(t)

            self.ht2r[(h, t)].append(r)
            self.r2ht[r].append((h, t))

            self.h2t[h].append(t)
            self.t2h[t].append(h)

            self.h2r2t[h][r] = t
            self.t2r2h[t][r] = h

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
    def create(cls, triple_file, auto_index=True, **kwargs):
        """
        Create the class
        TO be modified when certain parameters controls the triple_file
        """
        if auto_index:
            return cls(triple_file, **kwargs)
        else:
            base_dir = os.path.dirname(triple_file)
            with open(os.path.join(base_dir, 'map_entity_id_to_text.tsv')) as f:
                num_entities = len(f.readlines())
            with open(os.path.join(base_dir, 'map_relation_id_to_text.tsv')) as f:
                num_relations = len(f.readlines())
            print(
                f"load indexed #entities={num_entities} and #relation={num_relations}")
            return cls(triple_file, num_entities, num_relations, **kwargs)

    def get_eval_triple_iterator(self, **kwargs):
        dataloader = DataLoader(self.triples, **kwargs)
        return dataloader

    # def lcwa_negative_sampling(self,
    #                            positive_triples=None,
    #                            entity_list=None,
    #                            negative_sample_scope='subgraph'):

    #     assert positive_triples is not None

    #     if entity_list is None:
    #         entity_set = set()
    #         for h, r, t in positive_triples:
    #             entity_set.add(h)
    #             entity_set.add(t)
    #         entity_list = list(entity_set)

    #     assert negative_sample_scope in ['graph', 'subgraph']
    #     if negative_sample_scope == 'graph':
    #         def entity_sampler():
    #             return self.get_random_entity()
    #     else:
    #         def entity_sampler():
    #             return random.sample(entity_list, 1)[0]

    #     negative_triples = []
    #     positive_triples_sets = set(positive_triples)
    #     for triple in positive_triples:
    #         h, r, t = triple
    #         # make another background noise version
    #         which = np.random.choice([0, 1])
    #         if which == 0:
    #             neg_triple = (entity_sampler(), r, t)
    #         else:
    #             neg_triple = (h, r, entity_sampler())
    #         negative_triples.append(neg_triple)

    #         # while True:
    #         #     which = np.random.choice([0, 1])
    #         #     if which == 0:
    #         #         neg_triple = (entity_sampler(), r, t)
    #         #     elif which == 1:
    #         #         neg_triple = (h, r, entity_sampler())

    #         #     if neg_triple not in positive_triples_sets:
    #         #         negative_triples.append(neg_triple)
    #         #         break

    #     return negative_triples

    def __preproc_entities(self, entities: Union[List[int], torch.Tensor]):
        if isinstance(entities, list):
            if isinstance(entities[0], int):
                # in this case, batch size = 1
                entity_tensor = torch.tensor(
                    entities, device=self.device).reshape(1, -1)
            elif isinstance(entities[0], list):
                # in this case, batch size = len(entities)
                assert isinstance(entities[0][0], int)
                entity_tensor = torch.tensor(
                    entities, device=self.device).reshape(len(entities), -1)
            else:
                raise NotImplementedError(
                    "higher order nested list is not supported")
        elif isinstance(entities, torch.Tensor):
            assert entities.dim() == 2
            entity_tensor = entities.to(self.device)
        else:
            raise NotImplementedError("unsupported input entities type")
        return entity_tensor

    def __get_entity_mask(self, entity_tensor):
        batch_size, num_entities = entity_tensor.shape
        first_indices = torch.tile(
            torch.arange(batch_size).view(batch_size, 1),
            dims=(1, num_entities))
        node_mask = torch.zeros(
            size=(batch_size, self.num_entities),
            dtype=torch.bool,
            device=self.device)
        # since now, the input should be tensor [batch_size, num_entities]
        node_mask[first_indices, entity_tensor] = 1
        return node_mask

    def get_sub_graph(self,
                      entities: Union[List[int], torch.Tensor],
                      negative_sampling: bool = True,
                      k=10,
                      **kwargs):
        entity_tensor = self.__preproc_entities(entities)
        batch_size, num_entities = entity_tensor.shape

        entity_mask = self.__get_entity_mask(entity_tensor)

        # so far you have a mask of shape [batch_size, total_num_entities]
        batch_triple_mask = torch.logical_and(
            entity_mask[:, self.triple_tensor[:, 0]],
            entity_mask[:, self.triple_tensor[:, 2]])
        subgraph_batch_triple_count = torch.sum(batch_triple_mask, dim=-1)
        subgraph_flat_triple_ids = batch_triple_mask.nonzero()[:, 1]
        subgraph_flat_triples = self.triple_tensor[subgraph_flat_triple_ids]

        output = {
            "subgraph:flat_triples": subgraph_flat_triples,
            "subgraph:batch_triple_count": subgraph_batch_triple_count
        }

        # now do the negative sampling for noisy triples
        # we generate finite samples and returns the nomalized weights

        batch_entity_dist = entity_mask / entity_mask.sum(dim=-1, keepdim=True)
        noisy_head = torch.multinomial(batch_entity_dist,
                                       num_samples=k,
                                       replacement=True).unsqueeze(-1)
        noisy_tail = torch.multinomial(batch_entity_dist,
                                       num_samples=k,
                                       replacement=True).unsqueeze(-1)
        noisy_rel = torch.randint(
            low=0, high=self.num_relations, size=noisy_head.shape)

        noisy_triples = torch.cat([noisy_head, noisy_rel, noisy_tail], dim=-1)
        noisy_weights = torch.ones(size=(batch_size, k), device=self.device) / k

        #TODO: uniform weights now, may use weights now

        output['noisy:batch_triples'] = noisy_triples
        output['noisy:batch_weights'] = noisy_weights

        return output

    def get_neighbor_triples(self,
                             entities: Union[List[int], torch.Tensor],
                             reverse=False):
        """
        This function finds the triples in the KG but not in the sub graph
        Input args:
            - entities: tensor [batch_size, num_entities]
        Return args:
            - entities:
        """
        entity_tensor = self.__preproc_entities(entities)
        entity_mask = self.__get_entity_mask(entity_tensor)
        # so far you have a mask of shape [batch_size, total_num_entities]
        if reverse:
            batch_triple_mask = torch.logical_and(
                entity_mask[:, self.triple_tensor[:, 0]].logical_not(),
                entity_mask[:, self.triple_tensor[:, 2]])
        else:
            batch_triple_mask = torch.logical_and(
                entity_mask[:, self.triple_tensor[:, 0]],
                entity_mask[:, self.triple_tensor[:, 2]].logical_not())

        batch_selected_triple_count = torch.sum(batch_triple_mask, dim=-1)
        selected_triple_ids = batch_triple_mask.nonzero()[:, 1]
        neighbor_triples = self.triple_tensor[selected_triple_ids]

        return neighbor_triples, batch_selected_triple_count

    def get_non_neightbor_triple(self,
                                 entities: Union[List[int], torch.Tensor],
                                 k=10,
                                 reverse=False):
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
        entity_tensor = self.__preproc_entities(entities)
        batch_size, num_entities = entity_tensor.shape

        # [batch_size * num_entities, ]
        flat_entity_tensor = entity_tensor.ravel()
        if reverse:  # if the reverse is true, it considers the reversed edges
            possible_tails = torch.index_select(
                self.dconnect_index.T,
                dim=0,
                index=entity_tensor.ravel()).to_dense()
        else:
            possible_tails = torch.index_select(
                self.dconnect_index,
                dim=0,
                index=entity_tensor.ravel()).to_dense()
            # .reshape(shape=entity_tensor.shape + (-1,))
        impossible_tails = 1 - possible_tails
        impossible_tail_dist = impossible_tails / \
            impossible_tails.sum(-1, keepdim=True)
        flat_neg_tails = torch.multinomial(input=impossible_tail_dist,
                                           num_samples=k).unsqueeze(-1)
        flat_neg_heads = torch.tile(flat_entity_tensor.unsqueeze(-1),
                                    dims=(1, k)).unsqueeze(-1)
        flat_neg_rels = torch.randint(low=0, high=self.num_relations,
                                      size=flat_neg_tails.shape,
                                      device=self.device)
        if reverse:
            flat_neg_triples = torch.cat(
                [flat_neg_tails, flat_neg_rels, flat_neg_heads],
                dim=-1)
        else:
            flat_neg_triples = torch.cat(
                [flat_neg_heads, flat_neg_rels, flat_neg_tails],
                dim=-1)

        neg_triples = flat_neg_triples.view(batch_size, num_entities, k, 3)

        return neg_triples


class NeuralBinaryPredicate:
    def __init__(self):
        pass

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

    def sort_triples_by_scores(self, triples, assending=True):
        """
        One and only one tensor of the three input tensors is None.
        This function returns the id with the least score.
        It can be interpreted as a sentence.
        """
        h, r, t = triples_to_tensors(triples, self.device)
        triple_score_tensor = self.batch_pred_score(h, r, t)
        triple_scores = triple_score_tensor.detach().cpu().numpy().tolist()
        triple_with_score = sorted(zip(triples, triple_scores),
                                   reverse=not assending,
                                   key=lambda x: x[1])
        ret = []
        for triple, score in triple_with_score:
            ret.append(triple)
        return ret

    @abstractmethod
    def compute_triple_loss(self, pos_triples, neg_triples, **kwargs):
        """
        the pos and neg triples are in the form of tensors
        """
        pass

    def evaluate_kg(self, kg, init_batch_size=1024, bound_numel=100000):
        init_batch_size = min(init_batch_size,
                              bound_numel // self.entity_embedding.weight.shape[0])
        return self._evaluate_kg(kg, init_batch_size)
        # try:
        #     return self._evaluate_kg(kg, init_batch_size)
        # except:
        #     return self.evaluate_kg(kg, init_batch_size//2)

    def _evaluate_kg(self, kg: KG, batch_size):
        record = defaultdict(list)
        with tqdm(kg.get_eval_triple_iterator(batch_size=batch_size)) as t:
            for _head_id_ten, _rel_id_ten, _tail_id_ten in t:
                _head_id_ten = _head_id_ten.to(self.device)
                _rel_id_ten = _rel_id_ten.to(self.device)
                _tail_id_ten = _tail_id_ten.to(self.device)

                _cand_id_ten = torch.arange(
                    0,
                    end=self.entity_embedding.weight.shape[0],
                    step=1,
                    device=_head_id_ten.device)

                num_cases = len(_rel_id_ten)
                num_candidates = len(_cand_id_ten)

                cand_id_ten = torch.reshape(_cand_id_ten, (1, num_candidates))

                head_id_ten = torch.reshape(_head_id_ten, (num_cases, 1))
                rel_id_ten = torch.reshape(_rel_id_ten, (num_cases, 1))
                tail_id_ten = torch.reshape(_tail_id_ten, (num_cases, 1))

                # predict head
                head_cand_score_tensor = self.batch_pred_score(
                    cand_id_ten, rel_id_ten, tail_id_ten)  # [num_cases, num_candidates]

                head_score = torch.take_along_dim(input=head_cand_score_tensor,
                                                  indices=head_id_ten,
                                                  dim=1)

                head_rank = torch.sum(head_cand_score_tensor >
                                      head_score, -1).cpu().numpy()

                record['head_hit1'].extend((head_rank < 1).tolist())
                record['head_hit3'].extend((head_rank < 3).tolist())
                record['head_hit10'].extend((head_rank < 10).tolist())
                record['head_mrr'].extend((1/(1+head_rank)).tolist())

                # [num_cases, num_candidates]
                # head_cand_sorted = torch.argsort(head_cand_score_tensor,
                #  dim=-1, descending=True)

                # assert (head_cand_sorted[torch.arange(
                #     len(head_rank)), head_rank] == _head_id_ten).all()
                # predict tail
                tail_cand_score_tensor = self.batch_pred_score(
                    head_id_ten, rel_id_ten, cand_id_ten)  # [num_cases, num_candidates]

                tail_score = torch.take_along_dim(input=tail_cand_score_tensor,
                                                  indices=tail_id_ten,
                                                  dim=1)

                tail_rank = torch.sum(tail_cand_score_tensor >
                                      tail_score, -1).cpu().numpy()

                record['tail_hit1'].extend((tail_rank < 1).tolist())
                record['tail_hit3'].extend((tail_rank < 3).tolist())
                record['tail_hit10'].extend((tail_rank < 10).tolist())
                record['tail_mrr'].extend((1/(1+tail_rank)).tolist())
                metric = {}
                for k in record:
                    metric[k] = np.mean(record[k])

                t.set_postfix(metric)

        return metric
