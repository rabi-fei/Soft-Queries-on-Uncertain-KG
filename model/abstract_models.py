import os
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
        self.triple_tensor = torch.tensor(
            self.triples,
            dtype=int,
            device=self.device)

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

    def get_random_entity(self):
        return random.randint(0, self.num_entities-1)

    def get_random_relation(self):
        return random.randint(0, self.num_relations-1)

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

    def get_sub_graph(self,
                      entities: Union[List[int], torch.Tensor],
                      **kwargs):

        if isinstance(entities, list):
            # in this case, batch size = 1
            entity_tensor = torch.tensor(
                entities, device=self.device).reshape(1, -1)
        elif isinstance(entities, torch.Tensor):
            assert entities.dim() == 2
            entity_tensor = entities
        else:
            raise NotImplementedError("unsupported input entities type")

        # since now, the input should be tensor [batch_size, num_entities]
        batch_size, num_entities = entity_tensor.shape
        first_indices = torch.tile(
            torch.arange(batch_size).view(batch_size, 1),
            dims=(1, num_entities))
        node_mask = torch.zeros(
            size=(batch_size, self.num_entities),
            dtype=torch.bool,
            device=self.device)

        node_mask[first_indices, entity_tensor] = 1
        # so far you have a mask of shape [batch_size, total_num_entities]
        batch_triple_mask = node_mask[:, self.triple_tensor[:, 0]]
        batch_triple_mask = batch_triple_mask.logical_and(
            node_mask[:, self.triple_tensor[:, 2]])
        batch_selected_triple_count = torch.sum(batch_triple_mask, dim=-1)
        selected_triple_ids = batch_triple_mask.nonzero()[:, 1]
        subgraph_triples = self.triple_tensor[selected_triple_ids]



        return subgraph_triples, batch_selected_triple_count

    def get_neighbor_new_tail(self, entities: Union[List[int], torch.Tensor]):
        if isinstance(entities, list):
            # in this case, batch size = 1
            entity_tensor = torch.tensor(
                entities, device=self.device).reshape(1, -1)
        elif isinstance(entities, torch.Tensor):
            assert entities.dim() == 2
            entity_tensor = entities
        else:
            raise NotImplementedError("unsupported input entities type")

        # since now, the input should be tensor [batch_size, num_entities]
        batch_size, num_entities = entity_tensor.shape
        first_indices = torch.tile(
            torch.arange(batch_size).view(batch_size, 1),
            dims=(1, num_entities))
        node_mask = torch.zeros(
            size=(batch_size, self.num_entities), dtype=torch.bool, device=self.device)
        node_mask[first_indices, entity_tensor] = 1
        # so far you have a mask of shape [batch_size, total_num_entities]
        batch_triple_mask = node_mask[:, self.triple_tensor[:, 0]]
        batch_triple_mask = batch_triple_mask.logical_and(
            node_mask[:, self.triple_tensor[:, 2]].logical_not())
        batch_selected_triple_count = torch.sum(batch_triple_mask, dim=-1)
        selected_triple_ids = batch_triple_mask.nonzero()[:, 1]
        neighbor_triples = self.triple_tensor[selected_triple_ids]

        return neighbor_triples, batch_selected_triple_count

    def get_neighbor_new_head(self, entities: Union[List[int], torch.Tensor]):
        if isinstance(entities, list):
            # in this case, batch size = 1
            entity_tensor = torch.tensor(
                entities, device=self.device).reshape(1, -1)
        elif isinstance(entities, torch.Tensor):
            assert entities.dim() == 2
            entity_tensor = entities
        else:
            raise NotImplementedError("unsupported input entities type")

        # since now, the input should be tensor [batch_size, num_entities]
        batch_size, num_entities = entity_tensor.shape
        first_indices = torch.tile(
            torch.arange(batch_size).view(batch_size, 1),
            dims=(1, num_entities))
        node_mask = torch.zeros(
            size=(batch_size, self.num_entities), dtype=torch.bool, device=self.device)
        node_mask[first_indices, entity_tensor] = 1
        # so far you have a mask of shape [batch_size, total_num_entities]
        batch_triple_mask = node_mask[:, self.triple_tensor[:, 2]]
        batch_triple_mask = batch_triple_mask.logical_and(
            node_mask[:, self.triple_tensor[:, 0]].logical_not())
        batch_selected_triple_count = torch.sum(batch_triple_mask, dim=-1)
        selected_triple_ids = batch_triple_mask.nonzero()[:, 1]
        neighbor_triples = self.triple_tensor[selected_triple_ids]

        return neighbor_triples, batch_selected_triple_count


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
