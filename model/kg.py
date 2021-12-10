import os
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader

from model.predicate import NeuralBinaryPredicate

Triple = Tuple[int, int, int]


def iter_triple_from_tsv(triple_file):
    with open(triple_file, 'rt') as f:
        for line in f.readlines():
            tp = line.strip().split()
            assert len(tp) == 3
            triple = [int(t) for t in tp]
            yield triple


class KG:
    def __init__(self, triple_file, num_entities=None, num_relations=None):
        self.entity_set = set()
        self.ht2r = defaultdict(list)
        self.r2ht = defaultdict(list)

        self.h2t = defaultdict(list)
        self.t2h = defaultdict(list)

        self.h2r2t = defaultdict(dict)
        self.t2r2h = defaultdict(dict)
        self.triples = []

        for h, r, t in iter_triple_from_tsv(triple_file):
            self.triples.append((h, r, t))

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

        self.num_entities = len(
            self.entity_set) if num_entities is None else num_entities
        self.num_relations = len(
            self.r2ht) if num_relations is None else num_relations

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

    @classmethod
    def create(cls, triple_file, auto_index=True):
        """
        Create the class
        TO be modified when certain parameters controls the triple_file
        """
        if auto_index:
            return cls(triple_file)
        else:
            base_dir = os.path.dirname(triple_file)
            with open(os.path.join(base_dir, 'map_entity_id_to_text.tsv')) as f:
                num_entities = len(f.readlines())
            with open(os.path.join(base_dir, 'map_relation_id_to_text.tsv')) as f:
                num_relations = len(f.readlines())
            print(
                f"load indexed #entities={num_entities} and #relation={num_relations}")
            return cls(triple_file, num_entities, num_relations)

    def get_random_entity(self):
        return random.randint(0, self.num_entities-1)

    def get_random_relation(self):
        return random.randint(0, self.num_relations-1)

    def get_triple_train_dataloader(self, neural_model: NeuralBinaryPredicate, **kwargs):
        def collate_function(triples):
            neg_triples = self.lcwa_negative_sampling(
                triples, negative_sample_scope='graph')
            return triples, neg_triples
        return DataLoader(self.triples, collate_fn=collate_function, **kwargs)

    def get_node_train_dataloader(self, neural_model: NeuralBinaryPredicate, **kwargs):
        def collate_function(triples):
            pass
        return DataLoader(list(self.entity_set), collate_fn=lambda x: x, **kwargs)

    def get_triple_eval_dataloader(self, neural_model: NeuralBinaryPredicate, **kwargs):
        def collate_function(triples):
            pass
        return DataLoader(list(self.entity_set), collate_fn=lambda x: x, **kwargs)

    def lcwa_negative_sampling(self,
                               positive_triples=None,
                               entity_list=None,
                               negative_sample_scope='subgraph'):

        assert positive_triples is not None

        if entity_list is None:
            entity_set = set()
            for h, r, t in positive_triples:
                entity_set.add(h)
                entity_set.add(t)
            entity_list = list(entity_set)

        assert negative_sample_scope in ['graph', 'subgraph']
        if negative_sample_scope == 'graph':
            def entity_sampler():
                return self.get_random_entity()
        else:
            def entity_sampler():
                return random.sample(entity_list, 1)[0]

        negative_triples = []
        positive_triples_sets = set(positive_triples)
        for triple in positive_triples:
            h, r, t = triple
            while True:
                which = np.random.choice([0, 1])
                if which == 0:
                    neg_triple = (entity_sampler(), r, t)
                elif which == 1:
                    neg_triple = (h, r, entity_sampler())

                if neg_triple not in positive_triples_sets:
                    negative_triples.append(neg_triple)
                    break
        return negative_triples

    def get_sub_graph(self,
                      entity_list: List[int],
                      negative_sampling=True,
                      negative_sample_scope='graph',
                      **kwargs) -> List[Tuple[Triple, int]]:
        positive_triples = []
        for h in entity_list:
            for t in entity_list:
                if (h, t) in self.ht2r:
                    for r in self.ht2r[(h, t)]:
                        positive_triples.append((h, r, t))

        negative_triples = []
        if negative_sampling:
            negative_triples = self.lcwa_negative_sampling(
                positive_triples=positive_triples,
                entity_list=entity_list,
                negative_sample_scope=negative_sample_scope)

            # pair wise negative triple sampling
        return positive_triples, negative_triples

    def get_neighbor_graph(self, entity_list: List[int]) -> List[Triple]:
        triples = set()
        for h in entity_list:
            for t in self.h2t[h]:
                for r in self.ht2r[(h, t)]:
                    triples.add((h, r, t))
        for t in entity_list:
            for h in self.t2h[t]:
                for r in self.ht2r[(h, t)]:
                    triples.add((h, r, t))
        return list(triples)
