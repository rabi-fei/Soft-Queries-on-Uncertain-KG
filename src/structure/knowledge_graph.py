import time
from collections import defaultdict
from typing import List, Tuple, Union, Any
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import KnowledgeGraphConfig
from .knowledge_graph_index import KGIndex
from ..utils.data_util import iter_triple_from_tsv, tensorize_batch_entities, RaggedBatch
from ..utils.class_util import fixed_depth_nested_dict

Triple = Tuple[Any, Any, Any]


class KnowledgeGraph:
    """
    Fully tensorized
    """

    def __init__(self, triples: List[Triple], kgindex: KGIndex, device='cpu', tensorize=False, **kwargs):
        self.triples = triples
        self.kgindex = kgindex
        self.num_entities = kgindex.num_entities
        self.num_relations = kgindex.num_relations
        self.device = device

        self.hr2t = defaultdict(set)
        self.tr2h = defaultdict(set)
        self.r2ht = defaultdict(set)
        self.ht2r = defaultdict(set)
        self.r2h = defaultdict(set)
        self.r2t = defaultdict(set)
        self.h2t = defaultdict(set)
        self.t2h = defaultdict(set)
        self.node2or = fixed_depth_nested_dict(int, 2)
        self.node2ir = fixed_depth_nested_dict(int, 2)

        for h, r, t in self.triples:
            self.hr2t[(h, r)].add(t)
            self.tr2h[(t, r)].add(h)
            self.r2ht[r].add((h, t))
            self.ht2r[(h, t)].add(r)
            self.h2t[h].add(t)
            self.t2h[t].add(h)
            self.r2h[r].add(h)
            self.r2t[r].add(t)
            self.node2or[h][r] += 1
            self.node2ir[t][r] += 1

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
    def create(cls, triple_files, kgindex: KGIndex, **kwargs):
        """
        Create the class
        TO be modified when certain parameters controls the triple_file
        triple files can be a list
        """
        triples = []
        for h, r, t in iter_triple_from_tsv(triple_files):
            assert h in kgindex.inverse_entity_id_to_name
            assert r in kgindex.inverse_relation_id_to_name
            assert t in kgindex.inverse_entity_id_to_name
            triples.append((h, r, t))

        return cls(triples,
                   kgindex=kgindex,
                   **kwargs)

    def dump(self, filename):
        with open(filename, 'wt') as f:
            for h, r, t in self.triples:
                f.write(f"{h}\t{r}\t{t}\n")

    @classmethod
    def from_config(cls, config: KnowledgeGraphConfig):
        return cls.create(triple_files=config.filelist,
                          kgindex=KGIndex.load(config.kgindex_file),
                          device=config.device)

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


def subgraph_matching(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate_set: defaultdict,
                      data_graph: KnowledgeGraph):
    if len(now_candidate_set) == 1:
        final_node = list(now_candidate_set)[0]
        exist_answer = bool(now_candidate_set[final_node])
        return now_candidate_set, exist_answer
    now_leaf_node, adjacency_node = find_leaf_node(sub_graph, neg_sub_graph, now_candidate_set)
    if now_leaf_node:  # If there exists leaf node in the query graph, always possible to shrink into a sub_problem.
        adjacency_node_set = {adjacency_node}
        answer, exist_answer = cut_node_sub_problem(now_leaf_node, adjacency_node_set, sub_graph, neg_sub_graph,
                                                    now_candidate_set, data_graph)
        return answer, exist_answer
    else:
        before_topology_set = node_filter(sub_graph, now_candidate_set, data_graph)
        topology_filtered_set = topology_filter(sub_graph, neg_sub_graph, before_topology_set, data_graph)
        while before_topology_set != topology_filtered_set:
            before_topology_set = topology_filtered_set
            topology_filtered_set = topology_filter(sub_graph, neg_sub_graph, before_topology_set, data_graph)
        fixed_node, exist_answer = check_candidate_set(topology_filtered_set)
        if not exist_answer:
            return None, False
        if fixed_node:
            adjacency_node_set = set.union(*[sub_graph.h2t[fixed_node], sub_graph.t2h[fixed_node],
                                             neg_sub_graph.h2t[fixed_node], neg_sub_graph.t2h[fixed_node]])
            answer, exist_answer = cut_node_sub_problem(fixed_node, adjacency_node_set, sub_graph, neg_sub_graph,
                                                        now_candidate_set, data_graph)
            return answer, exist_answer
        else:  # Has to take a guess here.
            guess_node = min(now_candidate_set.items(), key=lambda x: len(x[1]))[0]
            collect_guess_ans = defaultdict(set)
            for candidate in now_candidate_set[guess_node]:
                new_candidate_set = deepcopy(now_candidate_set)
                new_candidate_set[guess_node] = candidate
                adjacency_node_set = set.union(*[sub_graph.h2t[guess_node], sub_graph.t2h[guess_node],
                                                 neg_sub_graph.h2t[guess_node], neg_sub_graph.t2h[guess_node]])
                answer, exist_answer = cut_node_sub_problem(guess_node, adjacency_node_set, sub_graph, neg_sub_graph,
                                                            new_candidate_set, data_graph)
                if exist_answer:
                    collect_guess_ans[guess_node].add(candidate)
                    for sub_node in answer:
                        collect_guess_ans[sub_node].update(answer[sub_node])
            exist_final_answer = bool(collect_guess_ans[guess_node])
            return collect_guess_ans, exist_final_answer


def node_filter(sub_graph, now_candidate_set, data_graph):  # negation is useless here.
    for node in sub_graph.node2or:
        for out_edge in sub_graph.node2or[node]:
            now_candidate_set[node] = now_candidate_set[node].intersection(data_graph.r2h[out_edge])
            if sub_graph.node2or[node][out_edge] > 1:
                for data_node in data_graph.r2h[out_edge]:
                    if data_node in now_candidate_set[node] and data_graph.node2or[data_node][out_edge] < \
                            sub_graph.node2or[node][out_edge]:
                        now_candidate_set[node].remove(data_node)
    for node in sub_graph.node2ir:
        for in_edge in sub_graph.node2ir[node]:
            now_candidate_set[node] = now_candidate_set[node].intersection(data_graph.r2t[in_edge])
            if sub_graph.node2ir[node][in_edge] > 1:
                for data_node in data_graph.r2h[in_edge]:
                    if data_node in now_candidate_set[node] and data_graph.node2ir[data_node][in_edge] < \
                            sub_graph.node2ir[node][in_edge]:
                        now_candidate_set[node].remove(data_node)
    return now_candidate_set


def topology_filter(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate_set: defaultdict,
                    data_graph: KnowledgeGraph):
    for node in now_candidate_set:
        adjacency_node_set = set.union(
            *[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
              neg_sub_graph.t2h[node]])
        for adjacency_node in adjacency_node_set:
            now_candidate_set, exist_answer = node_pair_filtering(node, adjacency_node, sub_graph, neg_sub_graph,
                                                                  now_candidate_set, data_graph)
            now_candidate_set, exist_answer = node_pair_filtering(adjacency_node, node, sub_graph, neg_sub_graph,
                                                                  now_candidate_set, data_graph)
        """
        for tail_node in sub_graph.h2t[node]:
            inner_relation_list = sub_graph.ht2r[(node, tail_node)]
            topology_filter_set = set.union(*[set.intersection(*[data_graph.tr2h[(data_tail, inner_r)] for inner_r in inner_relation_list]) for data_tail in now_candidate_set[tail_node]])
            now_candidate_set[node] = now_candidate_set[node].intersection(topology_filter_set)
        for head_node in sub_graph.t2h[node]:
            inner_relation_list = sub_graph.ht2r[(head_node, node)]
            topology_filter_set = set.union(*[set.intersection(*[data_graph.hr2t[(data_head, inner_r)] for inner_r in inner_relation_list]) for data_head in now_candidate_set[head_node]])
            now_candidate_set[node] = now_candidate_set[node].intersection(topology_filter_set)
        """
    return now_candidate_set


def node_pair_filtering(leaf_node, adjacency_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                        now_candidate_set, data_graph: KnowledgeGraph) -> Tuple[defaultdict, bool]:
    node_pair, reverse_node_pair = (leaf_node, adjacency_node), (adjacency_node, leaf_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]
    all_successor = set()
    for candidate_leaf in now_candidate_set[leaf_node]:
        single_node_successor = set(range(data_graph.num_entities))
        if h2t_relation:
            h2t_constraint = set.intersection(*[data_graph.hr2t[(candidate_leaf, rel)] for rel in h2t_relation])
            single_node_successor = h2t_constraint
        if t2h_relation:
            t2h_constraint = set.intersection(*[data_graph.tr2h[(candidate_leaf, rel)] for rel in t2h_relation])
            single_node_successor = single_node_successor.intersection(t2h_constraint)
        if h2t_negation:
            h2t_negation_exclude = set.union(*[data_graph.hr2t[(candidate_leaf, rel)] for rel in h2t_negation])
            single_node_successor = single_node_successor.difference(h2t_negation_exclude)
        if t2h_negation:
            t2h_negation_exclude = set.union(*[data_graph.tr2h[(candidate_leaf, rel)] for rel in t2h_negation])
            single_node_successor = single_node_successor.difference(t2h_negation_exclude)
        all_successor.update(single_node_successor)
    now_candidate_set[adjacency_node] = now_candidate_set[adjacency_node].intersection(all_successor)
    exist_answer = (len(now_candidate_set[adjacency_node]) != 0)
    return now_candidate_set, exist_answer


def find_leaf_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate):
    return_candidate = [None, None, 0]
    for node in now_candidate:
        adjacency_node_set = set.union(
            *[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
              neg_sub_graph.t2h[node]])
        if len(adjacency_node_set) == 1:
            if not return_candidate[0] or len(now_candidate[node]) < return_candidate[2]:
                return_candidate = [node, list(adjacency_node_set)[0], len(now_candidate[node])]
    return return_candidate[0], return_candidate[1]


def kg_remove_node(kg: KnowledgeGraph, node: int):
    remove_kg_triple = [triple for triple in kg.triples if (node != triple[0] and node != triple[2])]
    new_kg = KnowledgeGraph(remove_kg_triple, kg.kgindex)
    return new_kg


def check_candidate_set(candidate_set):
    fixed_node = None
    for node in candidate_set:
        if len(candidate_set[node]) == 0:
            return None, False
        elif len(candidate_set[node]) == 1:
            fixed_node = node
    return fixed_node, True


def cut_node_sub_problem(to_cut_node, adjacency_node_set, sub_graph: KnowledgeGraph,
                         neg_sub_graph: KnowledgeGraph, now_candidate_set, data_graph: KnowledgeGraph):
    all_adj_exist_ans = True
    for adjacency_node in adjacency_node_set:
        now_candidate_set, adj_exist_ans = node_pair_filtering(to_cut_node, adjacency_node, sub_graph, neg_sub_graph,
                                                               now_candidate_set, data_graph)
        all_adj_exist_ans = adj_exist_ans and all_adj_exist_ans
    if not all_adj_exist_ans:
        return None, False
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    cut_node_candidate_set = now_candidate_set.pop(to_cut_node)
    sub_answer, sub_exist_answer = subgraph_matching(new_sub_graph, new_sub_neg_graph, now_candidate_set,
                                                     data_graph)
    if sub_exist_answer:
        sub_answer[to_cut_node] = cut_node_candidate_set
        if len(cut_node_candidate_set) != 1:  # In this case, the reason to cut is leaf node, we double check the ans.
            assert len(adjacency_node_set) == 1
            adjacency_node = list(adjacency_node_set)[0]
            extended_answer, exist_answer = node_pair_filtering(adjacency_node, to_cut_node, sub_graph, neg_sub_graph,
                                                                sub_answer, data_graph)
            return extended_answer, exist_answer
        else:
            return sub_answer, True
    else:
        return None, False


def subgraph_sample(sample_matrix, data_matrix):
    """
    This function does a extremely easy task: the graph contains multi edge but no edge type, M_ij = k means that there
    is k edge form i to j.
    """
    def get_leaf_node(matrix):
        row_sum, col_sum = np.sum(matrix, dim=0), np.sum(matrix, dim=1)
        row_max, _, col_max, _ = np.max(matrix, dim=0), np.max(matrix, dim=1)
        one_hop_row = (row_sum == row_max)
        one_hop_column = (col_sum == col_max)
        return None

    leaf_node = get_leaf_node(sample_matrix)
    if leaf_node:
        sub_and = get_leaf_node()



def kg2matrix(kg: KnowledgeGraph):
    all_node_set = kg.node2or.keys().union(kg.node2ir.keys())
    node_num = len(all_node_set)
    kg_matrix = np.zeros((node_num, node_num), dtype=int)
    for triple in kg.triples:
        head, relation, tail = triple
        kg_matrix[head][tail] += 1
    return kg_matrix
