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
        self.ht2r = defaultdict(list)
        self.r2ht = defaultdict(list)

        self.h2t = defaultdict(list)
        self.t2h = defaultdict(list)

        self.h2r2t = defaultdict(dict)
        self.t2r2h = defaultdict(dict)

        for h, r, t in iter_triple_from_tsv(triple_file):
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
        return len(self.h2t)

    @property
    def num_relations(self):
        return len(self.r2ht)

    def get_random_entity(self):
        return random.randint(0, self.num_entities-1)

    def get_random_relation(self):
        return random.randint(0, self.num_relations-1)

    def get_sub_graph(self,
                      entity_list: List[int],
                      negative_triples=True,
                      negative_sample_scope='subgraph',
                      pairwise_negative=True,
                      perturbation_vals=[1, 1, 1],
                      **kwargs) -> List[Tuple(Triple, 0 | 1)]:
        positive_triples = []
        for h in entity_list:
            for t in entity_list:
                if (h, t) in self.ht2r:
                    for r in self.ht2r[(h, t)]:
                        positive_triples.append((h, r, t))

        negative_triples = []
        if negative_triples:
            assert negative_sample_scope in ['graph', 'subgraph']
            if negative_sample_scope == 'graph':
                def entity_sampler():
                    return self.get_random_entity()
            else:
                def entity_sampler():
                    return random.sample(entity_list, 1)[0]

            if pairwise_negative:
                for triple in positive_triples:
                    neg_triple = triple[:]
                    while True:
                        which = np.random.choice([0, 1, 2])
                        if which == 1:
                            neg_triple[which] = self.get_random_relation()
                        else:
                            neg_triple[which] = entity_sampler()
                        if neg_triple not in positive_triples:
                            negative_triples.append(positive_triples)
                            break
            else:
                while len(negative_triples) < len(positive_triples):
                    neg_triple = [
                        entity_sampler(), self.get_random_relation(), entity_sampler()]
                    if neg_triple not in positive_triples:
                        negative_triples.append(neg_triple)

                # pair wise negative triple sampling
            for (h, r, t), _ in positive_triples:
                negative_triples.append()
        return positive_triples + negative_triples

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
