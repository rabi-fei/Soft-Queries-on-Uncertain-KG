import copy
import random
import time
import json
from collections import defaultdict
from scipy.sparse import coo_array, csr_array
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

    def __init__(self, facts: List[Triple], kgindex: KGIndex, device='cpu', tensorize=False, **kwargs):
        self.facts = facts
        self.kgindex = kgindex
        self.num_entities: int = kgindex.num_entities
        self.num_relations: int = kgindex.num_relations
        self.device = device
        self.hrt2p = defaultdict(list)
        self.hr2tp = defaultdict(list)
        self.tr2hp = defaultdict(list)
        self.ht2rab = defaultdict(set)

        self.hr2t = defaultdict(set)
        self.tr2h = defaultdict(set)
        self.r2ht = defaultdict(set)
        self.ht2r = defaultdict(set)
        self.r2h = defaultdict(set)
        self.r2t = defaultdict(set)
        self.h2t = defaultdict(set)
        self.t2h = defaultdict(set)
        self.r2percentile = defaultdict(list)
        self.node2or = fixed_depth_nested_dict(int, 2)
        self.node2ir = fixed_depth_nested_dict(int, 2)


        for fact in self.facts:
            if len(fact) == 3:
                h, r, t = fact
            elif len(fact) == 4:
                h, r, t, p = fact
                self.hrt2p[(h, r, t)].append(p)
                self.hr2tp[(h, r)].append((t, p))
                self.tr2hp[(t, r)].append((h, p))
            elif len(fact) == 5:
                h, r, t, alpha, beta = fact
                self.ht2rab[h, t].add((r, alpha, beta))
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
    def create(cls, quadruple_files, kgindex: KGIndex, **kwargs):
        """
        Create the class
        TO be modified when certain parameters controls the triple_file
        triple files can be a list
        """
        quadruples = []
        for h, r, t, p in iter_triple_from_tsv(quadruple_files):
            assert h in kgindex.inverse_entity_id_to_name
            assert r in kgindex.inverse_relation_id_to_name
            assert t in kgindex.inverse_entity_id_to_name
            quadruples.append((h, r, t, p))

        return cls(quadruples,
                   kgindex=kgindex,
                   **kwargs)
    def load_percentile(self, file_nmae):
        with open(file_nmae) as json_file:
            self.r2percentile = json.load(json_file)
        

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


def csp_efo1(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate_set: defaultdict,
             data_graph: KnowledgeGraph):
    if not sub_graph.triples and not neg_sub_graph.triples:
        return now_candidate_set, True
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
                new_candidate_set[guess_node] = {candidate}
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


def csp_efo1_soft(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate_set,
             data_graph: KnowledgeGraph):
    # solve soft queries.
    if not sub_graph.facts and not neg_sub_graph.facts:
        return now_candidate_set, True
    if len(now_candidate_set) == 1:
        final_node = list(now_candidate_set)[0]
        exist_answer = bool(now_candidate_set[final_node])
        return now_candidate_set, exist_answer
    now_node, adjacency_node = find_easy_node(sub_graph, neg_sub_graph, now_candidate_set)
    if now_node:  # If there exists leaf node in the query graph, always possible to shrink into a sub_problem.
        if "f" in now_node:
            pass #TODO.
        else:
            adjacency_node_set = adjacency_node if isinstance(adjacency_node, set) else {adjacency_node}
            answer, exist_answer = cut_node_sub_problem_soft(now_node, adjacency_node_set, sub_graph, neg_sub_graph,
                                                        now_candidate_set, data_graph)
            return answer, exist_answer
    else:
        if int(sub_graph.facts[0][3][:-1]):
            now_candidate_set_ = {}
            for node, candidate in now_candidate_set.items():
                if now_candidate_set[node].getnnz():
                    now_candidate_set_[node] = set(candidate.col)
                else:
                    now_candidate_set_[node] = set(range(candidate.shape[1]))
            topology_filtered_set = node_filter(sub_graph, now_candidate_set_, data_graph)
        else:
            null_candidate_set = {key: set() for key in now_candidate_set}
            topology_filtered_set = node_filter_v2(sub_graph, null_candidate_set, data_graph)
        fixed_node, exist_answer = check_candidate_set(topology_filtered_set)
        if not exist_answer:
            return None, False
        if fixed_node:
            adjacency_node_set = set.union(*[sub_graph.h2t[fixed_node], sub_graph.t2h[fixed_node],
                                             neg_sub_graph.h2t[fixed_node], neg_sub_graph.t2h[fixed_node]])
            new_candidate_set = deepcopy(now_candidate_set)
            fixed_grounded = list(topology_filtered_set[fixed_node])[0]
            fixed_node_candidate_value = now_candidate_set[fixed_node].toarray().squeeze()
            new_candidate_set.pop(fixed_node)
            new_candidate_set[fixed_node] = {fixed_grounded}
            answer, exist_answer = cut_node_sub_problem_soft(fixed_node, adjacency_node_set, sub_graph, neg_sub_graph,
                                                        new_candidate_set, data_graph)
            if exist_answer:
                if int(sub_graph.facts[0][3][:-1]):
                    final_answer = answer["f1"].toarray().squeeze() * fixed_node_candidate_value[fixed_grounded]
                else:
                        answer["f1"].toarray().squeeze() + fixed_node_candidate_value[candidate] #TODO:fix it
                        
        else:  # Has to take a guess here.
            guess_node = min(topology_filtered_set.items(), key=lambda x: len(x[1]))[0]
            collect_guess_ans = []
            guess_node_candidate_value = now_candidate_set[guess_node].toarray().squeeze()
            for candidate in topology_filtered_set[guess_node]:
                new_candidate_set = deepcopy(now_candidate_set)
                new_candidate_set.pop(guess_node)
                new_candidate_set[guess_node] = {candidate}
                adjacency_node_set = set.union(*[sub_graph.h2t[guess_node], sub_graph.t2h[guess_node],
                                                 neg_sub_graph.h2t[guess_node], neg_sub_graph.t2h[guess_node]])
                answer, exist_answer = cut_node_sub_problem_soft(guess_node, adjacency_node_set, sub_graph, neg_sub_graph,
                                                            new_candidate_set, data_graph)
                if exist_answer:
                    if int(sub_graph.facts[0][3][:-1]):
                        collect_guess_ans.append(answer["f1"].toarray().squeeze() * guess_node_candidate_value[candidate])
                    else:
                        collect_guess_ans.append(answer["f1"].toarray().squeeze() + guess_node_candidate_value[candidate])
            final_answer = np.array(collect_guess_ans).max(axis=0)
        cols = np.nonzero(final_answer)[0]
        final_answer = coo_array((final_answer[cols], (np.zeros(cols.shape[0]), cols)), shape=(1, final_answer.shape[0]))
        exist_final_answer = final_answer.max() > 0
        return {"f1": final_answer}, exist_final_answer



def csp_efox(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate_set: defaultdict,
             data_graph: KnowledgeGraph, free_variable_list: List):
    """
    Returns a list of dict, example:
    [{'f1': 13536, 'f2': 11440}, {'f1': 11441, 'f2': 11440}, {'f1': 7000, 'f2': 11440}]
    """
    if not sub_graph.triples and not neg_sub_graph.triples:
        copy_candidate_set = deepcopy(now_candidate_set)
        for variable_name in now_candidate_set:
            if variable_name not in free_variable_list:
                copy_candidate_set.pop(variable_name)
        if len(copy_candidate_set.keys()) == 0:
            exist_existential_ans = True
            for variable_name in now_candidate_set:
                if not now_candidate_set[variable_name]:
                    exist_existential_ans = False
                    break
            if exist_existential_ans:
                return [{}], True
            else:
                return None, False
        answer_list = candidate_set_to_ans(copy_candidate_set)
        return answer_list, bool(answer_list)
    if len(now_candidate_set) == 1:
        answer_list = candidate_set_to_ans(now_candidate_set)
        return answer_list, bool(answer_list)
    now_leaf_node, adjacency_node = find_leaf_node(sub_graph, neg_sub_graph, now_candidate_set)
    if now_leaf_node:  # If there exists leaf node in the query graph, always possible to shrink into a sub_problem.
        adjacency_node_set = {adjacency_node}
        answer, exist_answer = cut_node_sub_problem_x(now_leaf_node, adjacency_node_set, sub_graph, neg_sub_graph,
                                                      now_candidate_set, data_graph, free_variable_list)
        return answer, exist_answer
    else:
        before_topology_set = node_filter(sub_graph, now_candidate_set, data_graph)
        topology_filtered_set = topology_filter(sub_graph, neg_sub_graph, before_topology_set, data_graph)
        while before_topology_set != topology_filtered_set:
            before_topology_set = topology_filtered_set
            topology_filtered_set = topology_filter(sub_graph, neg_sub_graph, before_topology_set, data_graph)
        guess_node = min(now_candidate_set.items(), key=lambda x: len(x[1]))[0]   # Has to take a guess here.
        collect_guess_ans = []
        for candidate in now_candidate_set[guess_node]:
            new_candidate_set = deepcopy(now_candidate_set)
            new_candidate_set[guess_node] = {candidate}
            adjacency_node_set = set.union(*[sub_graph.h2t[guess_node], sub_graph.t2h[guess_node],
                                             neg_sub_graph.h2t[guess_node], neg_sub_graph.t2h[guess_node]])
            if guess_node in free_variable_list:
                new_free_variable_list = deepcopy(free_variable_list)
                new_free_variable_list.remove(guess_node)
                answer, exist_answer = cut_node_sub_problem_x(guess_node, adjacency_node_set, sub_graph,
                                                              neg_sub_graph,
                                                              new_candidate_set, data_graph, new_free_variable_list)
            else:
                answer, exist_answer = cut_node_sub_problem_x(guess_node, adjacency_node_set, sub_graph,
                                                              neg_sub_graph,
                                                              new_candidate_set, data_graph, free_variable_list)
            if exist_answer:
                for answer_instance in answer:
                    if guess_node in free_variable_list:
                        copy_instance = deepcopy(answer_instance)
                        copy_instance[guess_node] = candidate
                        collect_guess_ans.append(copy_instance)
                    else:
                        collect_guess_ans.append(answer_instance)
        exist_final_answer = bool(collect_guess_ans)
        return collect_guess_ans, exist_final_answer


def candidate_set_to_ans(now_candidate_set):
    if len(now_candidate_set) == 1:
        only_var = list(now_candidate_set.keys())[0]
        ans_list = []
        for candidate in now_candidate_set[only_var]:
            ans_list.append({only_var: candidate})
        return ans_list
    random_key = random.choice(list(now_candidate_set.keys()))
    this_node_candidate = now_candidate_set.pop(random_key)
    sub_ans_list = candidate_set_to_ans(now_candidate_set)
    new_ans_list = []
    for sub_ans in sub_ans_list:
        for candidate in this_node_candidate:
            new_ans = copy.deepcopy(sub_ans)
            new_ans[random_key] = candidate
            new_ans_list.append(new_ans)
    return new_ans_list


'''
def get_final_answer(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate_set: defaultdict,
             data_graph: KnowledgeGraph, free_variable_list: List):
    """
    In this function, we get the final answer since EFOX requires some manual backtracking.
    """
    assert set(free_variable_list) == set(now_candidate_set.keys())
    now_leaf_node, adjacency_node = find_leaf_node(sub_graph, neg_sub_graph, now_candidate_set)
    if now_leaf_node:  # Again, leaf node can be neglected for now.
        adjacency_node_set = {adjacency_node}
        answer, exist_answer = cut_node_final_answer(now_leaf_node, adjacency_node_set, sub_graph, neg_sub_graph,
                                                    now_candidate_set, data_graph, free_variable_list)
        return answer, exist_answer
    else:
        

    
def cut_node_final_answer(to_cut_node, adjacency_node_set, sub_graph: KnowledgeGraph,
                         neg_sub_graph: KnowledgeGraph, now_candidate_set, data_graph: KnowledgeGraph,
                          free_variable_list: List):
    new_candidate_set = copy.deepcopy(now_candidate_set)
    all_adj_exist_ans = True
    for adjacency_node in adjacency_node_set:
        new_candidate_set, adj_exist_ans = node_pair_filtering(to_cut_node, adjacency_node, sub_graph, neg_sub_graph,
                                                               new_candidate_set, data_graph)
        all_adj_exist_ans = adj_exist_ans and all_adj_exist_ans
    if not all_adj_exist_ans:
        return None, False
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
        kg_remove_node(neg_sub_graph, to_cut_node)
    cut_node_candidate_set = new_candidate_set.pop(to_cut_node)
    sub_free_variable_list = free_variable_list.remove(to_cut_node)
    sub_answer, sub_exist_answer = get_final_answer(new_sub_graph, new_sub_neg_graph, new_candidate_set,
                                            data_graph, sub_free_variable_list)
    if sub_exist_answer:
        new_answer_list = []
        for answer_instance in sub_answer:
            copy_answer_instance = copy.deepcopy(answer_instance)
            cut_node_instance_set = copy.deepcopy(cut_node_candidate_set)
            for adjacency_node in adjacency_node_set:
                cut_node_instance_candidate, adj_exist_ans = node_pair_filtering(
                    to_cut_node, adjacency_node, sub_graph, neg_sub_graph, copy.deepcopy(answer_instance), data_graph)
                cut_node_instance_set = cut_node_instance_set.intersection(cut_node_instance_candidate)
            for instance_candidate in cut_node_instance_set:
                copy_answer_instance[to_cut_node] = instance_candidate
                new_answer_list.append(copy_answer_instance)
        return new_answer_list, bool(len(new_answer_list))
    else:
        return None, False
'''


def node_filter(sub_graph, now_candidate_set, data_graph):  # negation is useless here.
    for node in sub_graph.node2or:
        for out_edge in sub_graph.node2or[node]:
            now_candidate_set[node] = now_candidate_set[node].intersection(data_graph.r2h[out_edge])
            '''
            if sub_graph.node2or[node][out_edge] > 1:
                for data_node in data_graph.r2h[out_edge]:
                    if data_node in now_candidate_set[node] and data_graph.node2or[data_node][out_edge] < \
                            sub_graph.node2or[node][out_edge]:
                        now_candidate_set[node].remove(data_node)
            '''
    for node in sub_graph.node2ir:
        for in_edge in sub_graph.node2ir[node]:
            now_candidate_set[node] = now_candidate_set[node].intersection(data_graph.r2t[in_edge])
            '''
            if sub_graph.node2ir[node][in_edge] > 1:
                for data_node in data_graph.r2h[in_edge]:
                    if data_node in now_candidate_set[node] and data_graph.node2ir[data_node][in_edge] < \
                            sub_graph.node2ir[node][in_edge]:
                        now_candidate_set[node].remove(data_node)
            '''
    return now_candidate_set


def node_filter_v2(sub_graph, now_candidate_set, data_graph):  # negation is useless here.
    #For zero requirements
    for node in sub_graph.node2or:
        for out_edge in sub_graph.node2or[node]:
            now_candidate_set[node] = now_candidate_set[node].union(data_graph.r2h[out_edge])
            '''
            if sub_graph.node2or[node][out_edge] > 1:
                for data_node in data_graph.r2h[out_edge]:
                    if data_node in now_candidate_set[node] and data_graph.node2or[data_node][out_edge] < \
                            sub_graph.node2or[node][out_edge]:
                        now_candidate_set[node].remove(data_node)
            '''
    for node in sub_graph.node2ir:
        for in_edge in sub_graph.node2ir[node]:
            now_candidate_set[node] = now_candidate_set[node].union(data_graph.r2t[in_edge])
            '''
            if sub_graph.node2ir[node][in_edge] > 1:
                for data_node in data_graph.r2h[in_edge]:
                    if data_node in now_candidate_set[node] and data_graph.node2ir[data_node][in_edge] < \
                            sub_graph.node2ir[node][in_edge]:
                        now_candidate_set[node].remove(data_node)
            '''
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


def node_pair_filtering(now_node, to_change_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                        now_candidate_set, data_graph: KnowledgeGraph) -> Tuple[defaultdict, bool]:
    """
    Use now node to change to_change node
    """
    node_pair, reverse_node_pair = (now_node, to_change_node), (to_change_node, now_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]
    all_successor = set()
    if len(now_candidate_set[now_node]) == data_graph.num_entities:  # Special speed up for whole set.
        if len(h2t_relation) + len(t2h_relation) + len(h2t_negation) + len(t2h_negation) == 1:
            if len(h2t_relation) == 1:
                now_candidate_set[to_change_node] = now_candidate_set[to_change_node].intersection(
                    data_graph.r2t[list(h2t_relation)[0]])
            elif len(t2h_relation) == 1:
                now_candidate_set[to_change_node] = now_candidate_set[to_change_node].intersection(
                    data_graph.r2h[list(t2h_relation)[0]])
            else:
                pass  # Do nothing because it is negation.
            exist_answer = (len(now_candidate_set[to_change_node]) != 0)
            return now_candidate_set, exist_answer
    for candidate_leaf in now_candidate_set[now_node]:
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
    now_candidate_set[to_change_node] = now_candidate_set[to_change_node].intersection(all_successor)
    exist_answer = (len(now_candidate_set[to_change_node]) != 0)
    return now_candidate_set, exist_answer


def merge_indice_value(tp):#or hp
    indice2value = defaultdict(list)
    for indice, value in tp:
        indice2value[indice].append(value)
    merge_indice2map = {indice:np.mean(indice2value[indice]) for indice in indice2value}
    array_list = [np.array(list(merge_indice2map.keys())), np.array(list(merge_indice2map.values()))]
    tp = np.vstack((array_list[0], array_list[1]))
    return tp

def node_pair_cutting_soft(now_node, to_change_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                        now_candidate_set, data_graph: KnowledgeGraph) -> Tuple[defaultdict, bool]:
    """
    Use now node to change to_change node
    """
    necess_index = int(sub_graph.facts[0][3][:-1]) // 25 -1
    if necess_index >= 0:
        necess_flag = True 
    else:
         necess_flag = False
    kg_num = data_graph.num_entities
    node_pair, reverse_node_pair = (now_node, to_change_node), (to_change_node, now_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]
    if isinstance(now_candidate_set[now_node], set): # Show that this is a constant!
        now_node_candidate = now_candidate_set[now_node]
    else:
        Leaf_values = now_candidate_set[now_node].toarray().squeeze()
        now_node_candidate_list = [data_graph.r2h[r] for r in h2t_relation] + [data_graph.r2t[r] for r in t2h_relation]
        if not necess_flag:
            now_node_candidate = set.union(*now_node_candidate_list)
        else:
            if len(now_candidate_set[now_node].col) > 0: # change the intitial values
                now_node_candidate_list += [set(now_candidate_set[now_node].col)]
                Leaf_values = now_candidate_set[now_node].toarray().squeeze()
            else:
                Leaf_values = np.exp(Leaf_values)
            now_node_candidate = set.intersection(*now_node_candidate_list)
    Candidate_values = csr_array((len(now_node_candidate), data_graph.num_entities))
    for index, candidate_leaf in enumerate(now_node_candidate):
        if necess_flag:
            candidate_values =  csr_array(
                (np.ones(kg_num), (np.zeros(kg_num), np.arange(kg_num))), 
                shape = (1, kg_num)) #All one.
            leaf_value = 1 if isinstance(now_candidate_set[now_node], set) else Leaf_values[candidate_leaf]
        else:
            candidate_values = csr_array((1, kg_num)) #All zero.
            leaf_value = 0 if isinstance(now_candidate_set[now_node], set) else Leaf_values[candidate_leaf]
        if h2t_relation:
            for rel in h2t_relation:
                node_pair_ = sub_graph.ht2rab[(now_node, to_change_node)]
                _, beta = [triple_[1:]  for triple_ in node_pair_ if triple_[0]==rel][0]
                alpha = data_graph.r2percentile[f"{rel}"][necess_index]
                if (candidate_leaf, rel) not in data_graph.hr2tp:
                    candidate_values *= 0 # for positive necess, don't need do something for zero necess
                    continue
                tp = merge_indice_value(data_graph.hr2tp[(candidate_leaf, rel)])
                if necess_flag:
                    tp = tp[:, tp[1]>=alpha]
                    indices, values = tp[0], tp[1]
                    candidate_values *= csr_array(
                                    (np.exp(beta * values) , (np.zeros(len(indices)), indices)), 
                                    shape=(1,kg_num))
                else:
                    indices, values = tp[0], tp[1]
                    candidate_values += csr_array(
                                    (beta * values , (np.zeros(len(indices)), indices)), 
                                    shape=(1,kg_num))
        if t2h_relation:
            for rel in t2h_relation:
                node_pair_ = sub_graph.ht2rab[(to_change_node, now_node)]
                _, beta = [triple_[1:]  for triple_ in node_pair_ if triple_[0]==rel][0]
                alpha = data_graph.r2percentile[f"{rel}"][necess_index]
                if (candidate_leaf, rel) not in data_graph.tr2hp:
                    candidate_values *= 0 # for positive necess, don't need do something for zero necess
                    continue
                hp = merge_indice_value(data_graph.tr2hp[(candidate_leaf, rel)])
                if necess_flag:
                    hp = hp[:, hp[1]>=alpha]
                    indices, values = hp[0], hp[1]
                    candidate_values *= csr_array(
                                    (np.exp(beta * values), (np.zeros(len(indices)), indices)), 
                                    shape=(1, kg_num))
                else:
                    indices, values = tp[0], tp[1]
                    candidate_values += csr_array(
                                    (beta * values , (np.zeros(len(indices)), indices)), 
                                    shape=(1, kg_num))
        if h2t_negation:
            for rel in h2t_negation:
                node_pair_ = neg_sub_graph.ht2rab[(now_node, to_change_node)]
                _, beta = [triple_[1:]  for triple_ in node_pair_ if triple_[0]==rel][0]
                alpha = data_graph.r2percentile[f"{rel}"][necess_index]
                all_values = np.zeros(kg_num)
                if (candidate_leaf, rel) in data_graph.hr2tp:
                    tp = merge_indice_value(data_graph.hr2tp[(candidate_leaf, rel)])
                    indices, values = tp[0], tp[1]
                    all_values[indices.astype(int)] = values
                
                if necess_flag:
                    filted_tp = np.exp(beta * (1 - all_values)) * ((1-all_values) >= alpha)
                    candidate_values *= csr_array(
                                    (filted_tp, (np.zeros(kg_num), np.arange(kg_num))), 
                                    shape=(1,kg_num))
                else:
                    candidate_values += csr_array(
                                    (beta * (1-all_values) + leaf_value, (np.zeros(kg_num), np.arange(kg_num))), 
                                    shape=(1,kg_num))
        if t2h_negation:
            t2h_negation_exclude = set.union(*[data_graph.tr2h[(candidate_leaf, rel)] for rel in t2h_negation])
            single_node_successor = single_node_successor.difference(t2h_negation_exclude)
        if necess_flag:
            Candidate_values[[index],:] = candidate_values * leaf_value
        else:
            candidate_values.data += leaf_value
            Candidate_values[[index],:] = candidate_values
    if necess_flag:
        if Candidate_values.getnnz() == 0:
            return None, False
        if now_candidate_set[to_change_node].getnnz() == 0: #Why?
            now_candidate_set[to_change_node] = coo_array(Candidate_values.max(axis=0))
        else:
            now_candidate_set[to_change_node] = coo_array(Candidate_values.max(axis=0) * now_candidate_set[to_change_node])
    else:
        if isinstance(now_candidate_set[now_node], set):
            now_candidate_set[to_change_node] = coo_array(Candidate_values + now_candidate_set[to_change_node])
        else:#TODO:Wong compute!
            All_Candidate_values = csr_array((Candidate_values.shape[0]+1, Candidate_values.shape[1]))
            All_Candidate_values[np.arange(Candidate_values.shape[0]), :] = Candidate_values
            All_Candidate_values[[Candidate_values.shape[0]], :] = csr_array(
                        (np.ones(kg_num)*now_candidate_set[now_node].max(), (np.zeros(kg_num), np.arange(kg_num))), 
                        shape = (1, kg_num)) 
            now_candidate_set[to_change_node] = coo_array(All_Candidate_values.max(axis=0) + now_candidate_set[to_change_node])
    exist_answer = (now_candidate_set[to_change_node].getnnz() != 0)
    return now_candidate_set, exist_answer


def node_pair_correspondence(now_node, to_change_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                             now_candidate_set, data_graph: KnowledgeGraph):
    """
    We want to find a one-many correspondence from adjacency node to leaf node.
    """
    node_pair, reverse_node_pair = (now_node, to_change_node), (to_change_node, now_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]
    correspondece_dict = {}
    for candidate_leaf in now_candidate_set[now_node]:
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
        correspondece_dict[candidate_leaf] = single_node_successor
    return correspondece_dict


def find_leaf_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate):
    """
    Find a leaf node with least possible candidate.
    """
    return_candidate = [None, None, 0]
    for node in now_candidate:
        adjacency_node_set = set.union(
            *[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
              neg_sub_graph.t2h[node]])
        if len(adjacency_node_set) == 1:
            if not return_candidate[0] or len(now_candidate[node]) < return_candidate[2]:
                return_candidate = [node, list(adjacency_node_set)[0], len(now_candidate[node])]
    return return_candidate[0], return_candidate[1]


def find_easy_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate):
    """
    Find a node to cut.
    Constant is first, then find existential leaf node, last return free variable leaf node.
    """
    return_candidate = [None, None, 0]
    for node in now_candidate:
        adjacency_node_set = set.union(
            *[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
              neg_sub_graph.t2h[node]])
        if isinstance(now_candidate[node], set):
            return node, adjacency_node_set

        if len(adjacency_node_set) == 1:
            if now_candidate[node].getnnz() == 0:
                now_candidate_num = now_candidate[node].shape[1]
            else:
                now_candidate_num = now_candidate[node].getnnz()

            if "f" in node and not return_candidate[0]:
                return_candidate = [node, list(adjacency_node_set)[0], now_candidate_num] 
            if "e" in node:
                if not return_candidate[0] or "f" in return_candidate[0] or now_candidate_num < return_candidate[2]:
                    return_candidate = [node, list(adjacency_node_set)[0], now_candidate_num] #FIX: lenghth of 
    return return_candidate[0], return_candidate[1]


def kg_remove_node(kg: KnowledgeGraph, node: int):
    remove_kg_triple = [triple for triple in kg.facts if (node != triple[0] and node != triple[2])]
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
    new_candidate_set = copy.deepcopy(now_candidate_set)
    all_adj_exist_ans = True
    for adjacency_node in adjacency_node_set:
        new_candidate_set, adj_exist_ans = node_pair_filtering(to_cut_node, adjacency_node, sub_graph, neg_sub_graph,
                                                               new_candidate_set, data_graph)
        all_adj_exist_ans = adj_exist_ans and all_adj_exist_ans
    if not all_adj_exist_ans:
        return None, False
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    cut_node_candidate_set = new_candidate_set.pop(to_cut_node)
    sub_answer, sub_exist_answer = csp_efo1(new_sub_graph, new_sub_neg_graph, new_candidate_set,
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

def cut_node_sub_problem_soft(to_cut_node, adjacency_node_set, sub_graph: KnowledgeGraph,
                         neg_sub_graph: KnowledgeGraph, now_candidate_set, data_graph: KnowledgeGraph):
    new_candidate_set = copy.deepcopy(now_candidate_set)
    all_adj_exist_ans = True
    for adjacency_node in adjacency_node_set:
        new_candidate_set, adj_exist_ans = node_pair_cutting_soft(to_cut_node, adjacency_node, sub_graph, neg_sub_graph,
                                                               new_candidate_set, data_graph)
        all_adj_exist_ans = adj_exist_ans and all_adj_exist_ans
    if not all_adj_exist_ans:
        return None, False
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    cut_node_candidate_set = new_candidate_set.pop(to_cut_node)
    sub_answer, sub_exist_answer = csp_efo1_soft(new_sub_graph, new_sub_neg_graph, new_candidate_set,
                                            data_graph)
    if sub_exist_answer:
        now_candidate_set.update(sub_answer)
        return now_candidate_set, sub_exist_answer
    else:
        return None, False


def cut_node_sub_problem_x(to_cut_node, adjacency_node_set, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                           now_candidate_set, data_graph: KnowledgeGraph, free_variable_list):
    """
    We need to pay attention to whether to_cut_node is in the free_variable_list.
    """
    new_candidate_set = copy.deepcopy(now_candidate_set)
    all_adj_exist_ans = True
    for adjacency_node in adjacency_node_set:
        new_candidate_set, adj_exist_ans = node_pair_filtering(to_cut_node, adjacency_node, sub_graph, neg_sub_graph,
                                                               new_candidate_set, data_graph)
        all_adj_exist_ans = adj_exist_ans and all_adj_exist_ans
    if not all_adj_exist_ans:
        return None, False
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    cut_node_candidate_set = new_candidate_set.pop(to_cut_node)
    new_free_variable_list = copy.deepcopy(free_variable_list)
    if to_cut_node in free_variable_list:
        new_free_variable_list.remove(to_cut_node)
        if len(adjacency_node_set) == 1 and list(adjacency_node_set)[0] not in free_variable_list:
            new_free_variable_list.append(list(adjacency_node_set)[0])
            cut_node_type = 'cut_free_e'  # A free node is cut, the connected existential node become free
        else:
            cut_node_type = 'cut_free_f'
    else:
        cut_node_type = 'cut_existential_or_constant'
    another_candidate_set = deepcopy(new_candidate_set)
    sub_answer, sub_exist_answer = csp_efox(new_sub_graph, new_sub_neg_graph, new_candidate_set,
                                            data_graph, new_free_variable_list)
    if sub_exist_answer:
        if cut_node_type == 'cut_existential_or_constant':
            return sub_answer, sub_exist_answer
        else:
            assert len(adjacency_node_set) == 1  # This may not right because of fixed point (only one candidate).
            adjacency_node = list(adjacency_node_set)[0]
            correspond_dict = node_pair_correspondence(
                adjacency_node, to_cut_node, sub_graph, neg_sub_graph, another_candidate_set, data_graph)
            new_answer_list = []
            for answer_instance in sub_answer:
                adj_ans = answer_instance[adjacency_node]
                cut_node_instance_candidate = correspond_dict[adj_ans]
                cut_node_instance_set = cut_node_candidate_set.intersection(cut_node_instance_candidate)
                for cut_node_candidate in cut_node_instance_set:
                    new_answer_instance = copy.deepcopy(answer_instance)
                    if cut_node_type == 'cut_free_f':  # Extend the answer, list appending.
                        new_answer_instance[to_cut_node] = cut_node_candidate
                    else:  # Change the answer accordingly.
                        new_answer_instance[to_cut_node] = cut_node_candidate
                        new_answer_instance.pop(adjacency_node)
                    new_answer_list.append(new_answer_instance)
            return new_answer_list, bool(len(new_answer_list))
    else:
        return None, False


def ground_variable(sample_matrix, data_matrix):
    """
    This function does a extremely easy task: the graph contains multi edge but no edge type, M_ij = k means that there
    is k edge form i to j.
    """
    if np.sum(sample_matrix) == 0:
        left_node_num = sample_matrix.shape[0]
        random_candidate = random.sample(set(range(data_matrix.shape[0])), left_node_num)
        return random_candidate, True
    leaf_node = get_matrix_leaf_node(sample_matrix)

    if leaf_node is not None:
        sub_query = remove_matrix_node(sample_matrix, leaf_node)
        sub_answer, sub_exist = ground_variable(sub_query, data_matrix)
        if not sub_exist:
            return None, False
        else:
            if np.sum(sample_matrix[leaf_node]) == 0:  # leaf node don't have out edge
                sub_answer.insert(leaf_node, None)
                adjacency_node = np.where(sample_matrix[:, leaf_node] != 0)[0][0]
                adjacency_ans = sub_answer[adjacency_node]
                leaf_in_edge_num = sample_matrix[adjacency_node][leaf_node]
                if np.max(data_matrix[adjacency_ans]) <= leaf_in_edge_num:
                    return None, False
                leaf_node_ans_list = np.where(data_matrix[adjacency_ans] >= leaf_in_edge_num)[0]
                leaf_node_ans = random.choice(leaf_node_ans_list)
                sub_answer[leaf_node] = leaf_node_ans
                return sub_answer, True
            else:
                sub_answer.insert(leaf_node, None)
                adjacency_node = np.where(sample_matrix[leaf_node] != 0)[0][0]
                adjacency_ans = sub_answer[adjacency_node]
                leaf_out_edge_num = sample_matrix[leaf_node][adjacency_node]
                if np.max(data_matrix[:, adjacency_ans]) < leaf_out_edge_num:
                    return None, False
                leaf_node_ans_list = np.where(data_matrix[:, adjacency_ans] >= leaf_out_edge_num)[0]
                leaf_node_ans = random.choice(leaf_node_ans_list)
                sub_answer[leaf_node] = leaf_node_ans
                return sub_answer, True
    else:
        node_num = sample_matrix.shape[0]
        if node_num == 1:
            random_ans = random.randint(0, data_matrix.shape[0] - 1)
            now_ans = [random_ans]
            return now_ans, True
        elif node_num == 3:
            try_time = 0
            while try_time < 30:
                try_time += 1
                guess_zero = random.randint(0, data_matrix.shape[0] - 1)
                candidate1_set = matrix_pair_filter(0, 1, [guess_zero], sample_matrix, data_matrix)
                candidate2_set = matrix_pair_filter(0, 2, [guess_zero], sample_matrix, data_matrix)
                for candidate1 in candidate1_set:
                    candidate2_by1_set = matrix_pair_filter(1, 2, [candidate1], sample_matrix, data_matrix)
                    candidate2_refined = candidate2_set.intersection(candidate2_by1_set)
                    if candidate2_refined:
                        candidate2 = random.sample(candidate2_refined, 1)[0]
                        triangle_answer = [guess_zero, candidate1, candidate2]
                        return triangle_answer, True
            return None, False
        else:
            raise NotImplementedError


def ground_variable_v2(sample_query, sample_matrix, path, data_matrix):
    """
    For zero reqquirements, there is a path for reaoning chain. Other nodes are randomly select.
    """
    node_num = sample_matrix.shape[0]
    grounded_entities_list = [None for i in range(node_num)]
    random_free_candidate = random.sample(set(range(data_matrix.shape[0])), 1)[0]
    grounded_entities_list[path[-1]] = random_free_candidate
    grounded_ans = random_free_candidate
    for index in np.arange(len(path)-1,0,-1):
        grounded_node, adjacency_node = path[index], path[index-1]
        leaf_in_edge_num = sample_matrix[adjacency_node][grounded_node]
        adj_node_ans_list = np.where(data_matrix[:, grounded_ans] >= leaf_in_edge_num)[0]
        if len(adj_node_ans_list) == 0:
            return None, False
        adj_node_ans = random.choice(adj_node_ans_list)
        grounded_entities_list[adjacency_node] = adj_node_ans
        grounded_ans = adj_node_ans
    for i in range(node_num):
        if not grounded_entities_list[i]:
            random_ground_candidate = random.sample(set(range(data_matrix.shape[0])), 1)[0]
            grounded_entities_list[i] = random_ground_candidate

    return grounded_entities_list, True
    

def ground_variable_v3(sample_query, sample_matrix, data_kg, index):
    """
    For positive requirements, node assignment are connected with relation satifying the alpha-necessity.
    """
    if np.sum(sample_matrix) == 0:
        left_node_num = sample_matrix.shape[0]
        random_candidate = random.sample(set(range(data_kg.num_entities)), left_node_num)
        return random_candidate, True
    leaf_node = get_matrix_leaf_node(sample_matrix)

    if leaf_node is not None:
        sub_matrix = remove_matrix_node(sample_matrix, leaf_node)
        sub_index = deepcopy(index)
        sub_index.pop(leaf_node)
        sub_answer, sub_exist = ground_variable_v3(sample_query, sub_matrix, data_kg, sub_index)
        if not sub_exist:
            return None, False
        else:
            if np.sum(sample_matrix[leaf_node]) == 0:  # leaf node don't have out edge
                sub_answer.insert(leaf_node, None)
                adjacency_node = np.where(sample_matrix[:, leaf_node] != 0)[0][0]
                adjacency_ans = sub_answer[adjacency_node]
                leaf_in_edge_num = sample_matrix[adjacency_node][leaf_node]
                necess_requirements = [tuple_[1] for tuple_ in sample_query.ht2rab[(index[adjacency_node], index[leaf_node])]]
                necess_index = int(necess_requirements[0][:-1]) // 25 - 1 # In a query ,the requirements have equal level!
                satisfy_leaf_candi = []
                for leaf_ans_candi in data_kg.h2t[adjacency_ans]:
                    exist_rs = data_kg.ht2r[(adjacency_ans, leaf_ans_candi)]
                    num_requirements = [0 for require in necess_requirements]
                    for exist_r in exist_rs:#TODO: Check the necessary
                        p = np.mean(data_kg.hrt2p[(adjacency_ans, exist_r, leaf_ans_candi)])
                        for i in range(len(necess_requirements)):
                            num_requirements[i] += (np.array(p).mean() >= data_kg.r2percentile[f"{exist_r}"][necess_index]) #TODO: use percentile index!
                    if np.all(num_requirements):
                        satisfy_leaf_candi.append(leaf_ans_candi)
                if not len(satisfy_leaf_candi):
                        return None, False
                leaf_node_ans = random.choice(satisfy_leaf_candi)
                sub_answer[leaf_node] = leaf_node_ans
                return sub_answer, True
            else:
                sub_answer.insert(leaf_node, None)
                adjacency_node = np.where(sample_matrix[leaf_node] != 0)[0][0]
                adjacency_ans = sub_answer[adjacency_node]
                necess_requirements = [tuple_[1] for tuple_ in sample_query.ht2rab[(index[leaf_node], index[adjacency_node])]]
                satisfy_leaf_candi = []
                for leaf_ans_candi in data_kg.t2h[adjacency_ans]:
                    exist_rs = data_kg.ht2r[(leaf_ans_candi, adjacency_ans)]
                    num_requirements = [0 for require in necess_requirements]
                    necess_index = int(necess_requirements[0][:-1]) // 25 - 1 # In a query ,the requirements have equal level!
                    for exist_r in exist_rs:#TODO: Check the necessary
                        p = np.mean(data_kg.hrt2p[(leaf_ans_candi, exist_r, adjacency_ans)])
                        for i in range(len(necess_requirements)):
                            num_requirements[i] += (p >= data_kg.r2percentile[f"{exist_r}"][necess_index]) #TODO: use percentile index!
                    if np.all(np.array(num_requirements) >= len(necess_requirements)):
                        satisfy_leaf_candi.append(leaf_ans_candi)
                if not len(satisfy_leaf_candi):
                        return None, False
                leaf_node_ans = random.choice(satisfy_leaf_candi)
                sub_answer[leaf_node] = leaf_node_ans
                return sub_answer, True
    else:
        node_num = sample_matrix.shape[0]
        if node_num == 1:
            random_ans = random.randint(0, data_kg.num_entities - 1)
            now_ans = [random_ans]
            return now_ans, True
        elif node_num == 3:
            try_time = 0
            while try_time < 30:
                try_time += 1
                guess_zero = random.randint(0, data_kg.num_entities - 1)
                candidate1_set = matrix_pair_filter_v2(0, 1, [guess_zero], sample_query, sample_matrix, data_kg, index)
                candidate2_set = matrix_pair_filter_v2(0, 2, [guess_zero], sample_query, sample_matrix, data_kg, index)
                for candidate1 in candidate1_set:
                    candidate2_by1_set = matrix_pair_filter_v2(1, 2, [candidate1], sample_query, sample_matrix, data_kg, index)
                    candidate2_refined = candidate2_set.intersection(candidate2_by1_set)
                    if candidate2_refined:
                        candidate2 = random.sample(candidate2_refined, 1)[0]
                        triangle_answer = [guess_zero, candidate1, candidate2]
                        return triangle_answer, True
            return None, False
        else:
            raise NotImplementedError


def matrix_pair_filter(node1, node2, candidate1_list, sample_matrix, data_matrix):
    candidate2_set = set()
    for candidate1 in candidate1_list:
        if sample_matrix[node1, node2] > 0: #TODO: Reverse?
            candidate2_list = np.where(data_matrix[:, candidate1] >= sample_matrix[node1, node2])[0]
        else:  # sample_matrix[node2, node1] > 0
            candidate2_list = np.where(data_matrix[candidate1] >= sample_matrix[node2, node1])[0]
        candidate2_set.update(set(candidate2_list))
    return candidate2_set

def matrix_pair_filter_v2(node1, node2, candidate1_list, sample_query, sample_matrix, data_kg, index):
    candidate2_set = set()
    for candidate1 in candidate1_list:
        if sample_matrix[node1, node2] > 0:
            necess_requirements = sample_query.ht2rab[(index[node1], index[node2])]
            tail_candidates = data_kg.h2t[candidate1]
            satisfy_candidate2_set = set()
            for tail_candidate in tail_candidates:
                r_list = data_kg.ht2r[(candidate1, tail_candidate)]
                num_requirements = [0 for require in necess_requirements]
                for i in range(len(necess_requirements)):
                    r_requirements = [np.mean(data_kg.hrt2p[(candidate1, r, tail_candidate)]) >= data_kg.r2percentile[f"{r}"][1] for r in r_list]
                    if np.any(r_requirements):
                        num_requirements[i] += 1
                if np.all(num_requirements):
                    satisfy_candidate2_set.add(tail_candidate)
        else:  # sample_matrix[node2, node1] > 0 TODO: elif?
            necess_requirements = sample_query.ht2rab[(index[node2], index[node1])]
            tail_candidates = data_kg.t2h[candidate1]
            satisfy_candidate2_set ={}
            for head_candidate in tail_candidates:
                r_list = data_kg.ht2r[(head_candidate, candidate1)]
                num_requirements = [0 for require in necess_requirements]
                for i in range(len(necess_requirements)):
                    r_requirements = [np.mean(data_kg.hrt2p[(head_candidate, r, candidate1)]) >= data_kg.r2percentile[f"{r}"][1] for r in r_list]
                    if np.any(r_requirements):
                        num_requirements[i] += 1
                if np.all(num_requirements):
                    satisfy_candidate2_set.add(head_candidate)
        candidate2_set.update(satisfy_candidate2_set)
    return candidate2_set


def ground_predicate_v3(grounded_entity_list: List, query_kg: KnowledgeGraph, data_kg: KnowledgeGraph):
    # For positive requirements, the predcit of every pair meet the requirements
    # Note. for multi-edge, the grounded predicate should be different.
    grounded_relation_dict = {}
    for (head, relation, tail, alpha, beta) in query_kg.facts:
        inner_relation_list = list(query_kg.ht2r[(head, tail)])
        necess_index = int(alpha[:-1]) // 25 -1
        grounded_head, grounded_tail = grounded_entity_list[head], grounded_entity_list[tail]
#        inner_relation_candidate = data_kg.ht2r[(grounded_head, grounded_tail)]

        inner_relation_candidate = set()
        for r in data_kg.ht2r[(grounded_head, grounded_tail)]:
                p = np.mean(data_kg.hrt2p[(grounded_head, r, grounded_tail)])
                if p >= data_kg.r2percentile[f"{r}"][necess_index]:
                    inner_relation_candidate.add(r)
        if len(query_kg.ht2r[head, tail]) > 1:
                chosed_r = {grounded_relation_dict[r] for r in query_kg.ht2r[head, tail] if r in grounded_relation_dict}
                inner_relation_candidate = inner_relation_candidate.difference(chosed_r)
        inner_relation_choice = random.sample(inner_relation_candidate, 1)[0]
        new_grounded_dict = {relation: inner_relation_choice}
        grounded_relation_dict.update(new_grounded_dict)
    return grounded_relation_dict

def ground_predicate_v2(grounded_entity_list: List, path: List, query_kg: KnowledgeGraph, data_kg: KnowledgeGraph):
    # For zero requirements, path is connected by relation, but other relations only connect the tail.
    path_pair = [(path[i], path[i+1]) for i in range(len(path)-1)]
    grounded_relation_dict = {}
    for (head, inner_relation, tail, alpha, beta) in query_kg.facts: #TODO: why data_keg is different with data_matrix.
        if (head, tail) in path_pair:
            grounded_head, grounded_tail = grounded_entity_list[head], grounded_entity_list[tail]
            inner_relation_candidate = data_kg.ht2r[(grounded_head, grounded_tail)]
        else:  
            exist_t = data_kg.h2t[grounded_entity_list[head]]
            random_tail = random.choice(list(exist_t))
            inner_relation_candidate = data_kg.ht2r[(grounded_entity_list[head], random_tail)]
        if len(query_kg.ht2r[head, tail]) > 1:
                chosed_r = {grounded_relation_dict[r] for r in query_kg.ht2r[head, tail] if r in grounded_relation_dict}
                inner_relation_candidate = inner_relation_candidate.difference(chosed_r)
        inner_relation_choice = random.sample(inner_relation_candidate, 1)[0]
        new_grounded_dict = {inner_relation: inner_relation_choice}
        grounded_relation_dict.update(new_grounded_dict)
    return grounded_relation_dict


def ground_predicate(grounded_entity_list: List, query_kg: KnowledgeGraph, data_kg: KnowledgeGraph):
    grounded_relation_dict = {}
    for (head, tail) in query_kg.ht2r:
        inner_relation_list = list(query_kg.ht2r[(head, tail)])
        grounded_head, grounded_tail = grounded_entity_list[head], grounded_entity_list[tail]
        inner_relation_candidate = data_kg.ht2r[(grounded_head, grounded_tail)]
        inner_relation_choices = random.sample(inner_relation_candidate, len(inner_relation_list))
        new_grounded_dict = {inner_relation_list[i]: inner_relation_choices[i] for i in range(len(inner_relation_list))}
        grounded_relation_dict.update(new_grounded_dict)
    return grounded_relation_dict

def get_matrix_leaf_node(matrix):
    boolean_matrix = (matrix != 0)
    col_sum, row_sum = np.sum(boolean_matrix, axis=0), np.sum(boolean_matrix, axis=1)
    edge_sum = col_sum + row_sum
    leaf_index_list = np.where(edge_sum == 1)[0]
    if len(leaf_index_list):
        sorted(leaf_index_list, key=lambda x: (np.sum(matrix[x]) + np.sum(matrix[:, x])))
        leaf_node = leaf_index_list[0]
        return leaf_node
    else:
        return None


def remove_matrix_node(matrix, node_index):
    new_matrix = np.delete(matrix, node_index, axis=0)
    new_matrix = np.delete(new_matrix, node_index, axis=1)
    return new_matrix


def kg2matrix(kg: KnowledgeGraph):
    """
    The nodes of kg should always be labeled by number first.
    """
    all_node_list = list(set(kg.node2or.keys()).union(set(kg.node2ir.keys())))
    node_num = len(all_node_list)
    kg_matrix = np.zeros((node_num, node_num), dtype=int)
    for fact in kg.facts:
        triple = fact[:3] if len(fact) > 3 else fact
        head, relation, tail = triple
        kg_matrix[head][tail] += 1
    return kg_matrix


def labeling_triples(triple_list):
    labeled_nodes = set()
    node2index = {}
    index2node = {}
    now_index = 0
    new_triple_list = []
    for fact in triple_list:
        if len(fact)>3:
            head, relation, tail, alpha, beta = fact
        else:
            head, relation, tail = fact
        if head not in labeled_nodes:
            labeled_nodes.add(head)
            node2index[head] = now_index
            index2node[now_index] = head
            now_index += 1
        if tail not in labeled_nodes:
            labeled_nodes.add(tail)
            node2index[tail] = now_index
            index2node[now_index] = tail
            now_index += 1
        if len(fact)>3:
            translated_triples = (node2index[head], relation, node2index[tail], alpha, beta)
        else:
            translated_triples = (node2index[head], relation, node2index[tail])
        new_triple_list.append(translated_triples)
    return new_triple_list, node2index, index2node


def label_triples_with_assign(triple_list, node2index):
    new_triple_list = []
    for triple in triple_list:
        head, relation, tail = triple
        translated_triples = (node2index[head], relation, node2index[tail])
        new_triple_list.append(translated_triples)
    return new_triple_list
