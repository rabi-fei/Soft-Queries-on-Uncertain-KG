import random
from collections import defaultdict

from typing import List, Tuple

import numpy as np

Triple = Tuple[int, int, int]


def iter_triple_from_tsv(triple_file):
    with open(triple_file, 'rt') as f:
        for line in f.readlines():
            tp = line.strip().split()
            assert len(tp) == 3
            triple = [int(t) for t in tp]
            yield triple


class KG:
    def __init__(self, triple_file):
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

    @classmethod
    def create(cls, triple_file):
        """
        Create the class
        TO be modified when certain parameters controls the triple_file
        """
        return cls(triple_file)

    @property
    def num_entities(self):
        return len(self.entity_set)

    @property
    def num_relations(self):
        return len(self.r2ht)

    def get_random_entity(self):
        return random.randint(0, self.num_entities-1)

    def get_random_relation(self):
        return random.randint(0, self.num_relations-1)

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
                which = np.random.choice([0, 1, 2])
                if which == 0:
                    neg_triple = (entity_sampler(), r, t)
                elif which == 1:
                    neg_triple = (h, self.get_random_relation(), t)
                elif which == 2:
                    neg_triple = (h, r, entity_sampler())

                if neg_triple not in positive_triples_sets:
                    negative_triples.append(neg_triple)
                    break
        return negative_triples

    def get_sub_graph(self,
                      entity_list: List[int],
                      negative_sampling=True,
                      negative_sample_scope='subgraph',
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
