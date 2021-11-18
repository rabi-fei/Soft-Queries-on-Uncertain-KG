import os
from collections import defaultdict

from typing import List, Tuple

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

    def get_sub_graph(self, entity_list: List[int]) -> List[Triple]:
        triples = []
        for h in entity_list:
            for t in entity_list:
                if (h, t) in self.ht2r:
                    triples.append((h, self.ht2r[(h, t)], t))
        return list(triples)

    def get_neighbor_graph(self, entity_list: List[int]) -> List[Triple]:
        triples = set()
        for h in entity_list:
            for t in self.h2t[h]:
                triples.add((h, self.ht2r[(h, t)], t))
        for t in entity_list:
            for h in self.t2h[t]:
                triples.add((h, self.ht2r[(h, t)], t))
        return list(triples)

    @classmethod
    def create(cls, triple_file):
        """
        Create the class
        TO be modified when certain parameters controls the triple_file
        """
        return cls(triple_file)
