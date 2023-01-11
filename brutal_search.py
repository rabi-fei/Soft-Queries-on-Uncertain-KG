import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict
from typing import List
import copy

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import pickle
from torch import nn
from scipy.sparse import csc_matrix, diags

from src.language.tnorm import GodelTNorm, ProductTNorm, Tnorm
from src.language.fof import ConjunctiveFormula, DisjunctiveFormula
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph, kg_remove_node
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.utils.data import QueryAnsweringSeqDataLoader_v2
from src.utils.data_util import RaggedBatch
from lifted_embedding_estimation_with_truth_value import compute_evaluation_scores


torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", type=str, default='sparse/scipy_0.01_0.01.pickle')
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-betae')


def solve_conjunctive(positive_graph: KnowledgeGraph, negative_graph: KnowledgeGraph, relation_matrix,
                      now_candidate_set: dict, conjunctive_tnorm, existential_tnorm, now_variable):

    if not positive_graph.triples and not negative_graph.triples:
        return now_candidate_set[now_variable]
    if len(now_candidate_set) == 1:
        return now_candidate_set
    now_leaf_node, adjacency_node, being_asked_variable = \
        find_leaf_node(positive_graph, negative_graph, now_candidate_set, now_variable)
    if now_leaf_node:  # If there exists leaf node in the query graph, always possible to shrink into a sub_problem.
        adjacency_node_list = [adjacency_node]
        if being_asked_variable:
            next_variable = adjacency_node
            sub_pos_g, sub_neg_g = kg_remove_node(positive_graph, now_leaf_node), \
                                   kg_remove_node(negative_graph, now_leaf_node)
            sub_ans = solve_conjunctive(sub_pos_g, sub_neg_g, relation_matrix, now_candidate_set,
                                        conjunctive_tnorm, existential_tnorm, next_variable)
            final_ans = extend_ans(now_leaf_node, adjacency_node, positive_graph, negative_graph, relation_matrix,
                                   now_candidate_set[now_leaf_node], conjunctive_tnorm, existential_tnorm, sub_ans)
            return final_ans
        else:
            answer = cut_node_sub_problem(
                now_leaf_node, adjacency_node_list, positive_graph, negative_graph, relation_matrix, now_candidate_set,
                conjunctive_tnorm, existential_tnorm, now_variable)
            return answer
    else:
        '''
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
            '''
        return None


def find_leaf_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable):
    """
    Find a leaf node with least possible candidate. The now-asking variable is first.
    """
    return_candidate = [None, None, 0]
    for node in now_candidate:
        adjacency_node_set = set.union(
            *[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
              neg_sub_graph.t2h[node]])
        if len(adjacency_node_set) == 1:
            if node == now_variable:
                return node, list(adjacency_node_set)[0], True
            if not return_candidate[0] or np.count_nonzero(now_candidate[node]) < return_candidate[2]:
                return_candidate = [node, list(adjacency_node_set)[0], np.count_nonzero(now_candidate[node])]
    return return_candidate[0], return_candidate[1], False


def cut_node_sub_problem(to_cut_node, adjacency_node_list, sub_graph: KnowledgeGraph,
                         neg_sub_graph: KnowledgeGraph, now_candidate_set, data_graph: KnowledgeGraph):
    new_candidate_set = copy.deepcopy(now_candidate_set)
    for adjacency_node in adjacency_node_list:
        new_candidate_set, adj_exist_ans = node_pair_filtering(to_cut_node, adjacency_node, sub_graph, neg_sub_graph,
                                                               new_candidate_set, data_graph)
        all_adj_exist_ans = adj_exist_ans and all_adj_exist_ans
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    cut_node_candidate_set = new_candidate_set.pop(to_cut_node)
    sub_answer, sub_exist_answer = solve_conjunctive(new_sub_graph, new_sub_neg_graph, new_candidate_set,
                                                     data_graph)
    if sub_exist_answer:
        sub_answer[to_cut_node] = cut_node_candidate_set
        if len(cut_node_candidate_set) != 1:  # In this case, the reason to cut is leaf node, we double check the ans.
            assert len(adjacency_node_list) == 1
            adjacency_node = list(adjacency_node_list)[0]
            extended_answer, exist_answer = node_pair_filtering(adjacency_node, to_cut_node, sub_graph, neg_sub_graph,
                                                                sub_answer, data_graph)
            return extended_answer, exist_answer
        else:
            return sub_answer, True
    else:
        return None, False


def node_pair_filtering(leaf_node, adjacency_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                        relation_matrix, adj_candidate_set, conj_tnorm, exist_tnorm, leaf_candidate_set) -> dict:
    node_pair, reverse_node_pair = (leaf_node, adjacency_node), (adjacency_node, leaf_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]
    matrix_list = []
    for r in h2t_relation:
        matrix_list.append(relation_matrix[r])
    for r in t2h_relation:
        matrix_list.append(relation_matrix[r].transpose())
    for r in h2t_negation:
        matrix_list.append(1 - relation_matrix[r])
    for r in t2h_negation:
        matrix_list.append(1 - relation_matrix[r].transpose())
    if conj_tnorm == 'product':
        all_prob_matrix = matrix_list[0]
        for i in matrix_list[1:]:
            all_prob_matrix = all_prob_matrix * matrix_list[i]
        all_prob_matrix = all_prob_matrix * leaf_candidate_set
    else:
        raise NotImplementedError
    if exist_tnorm == 'Godel':
        prob_vec = all_prob_matrix.max(dim=0)
        if conj_tnorm == 'product':
            final_ans = leaf_candidate_set * prob_vec
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return final_ans


def extend_ans(leaf_node, adjacency_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                        relation_matrix, leaf_candidate, conj_tnorm, exist_tnorm, sub_ans):
    node_pair, reverse_node_pair = (adjacency_node, leaf_node), (leaf_node, adjacency_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]
    matrix_list = []
    for r in h2t_relation:
        matrix_list.append(relation_matrix[r])
    for r in t2h_relation:
        matrix_list.append(relation_matrix[r].transpose())
    for r in h2t_negation:
        matrix_list.append(1 - relation_matrix[r])
    for r in t2h_negation:
        matrix_list.append(1 - relation_matrix[r].transpose())
    if conj_tnorm == 'product':
        all_prob_matrix = matrix_list[0]
        for i in matrix_list[1:]:
            all_prob_matrix = all_prob_matrix.multipliy(matrix_list[i])
        all_prob_matrix = np.multiply(all_prob_matrix.todense(), np.expand_dims(sub_ans, axis=1))
    else:
        raise NotImplementedError
    if exist_tnorm == 'Godel':
        prob_vec = np.amax(all_prob_matrix, axis=0)  # prob*vec is 1*n  matrix
    else:
        raise NotImplementedError
    if conj_tnorm == 'product':
        final_ans = leaf_candidate * np.asarray(prob_vec).squeeze()
    else:
        raise NotImplementedError
    return final_ans


def construct_matrix_list():
    pass


def solve_EFO1(DNF_formula:DisjunctiveFormula, relation_matrix, conjunctive_tnorm, existential_tnorm, index):
    sub_ans_list = []
    n_entity = relation_matrix[0].shape[0]
    for sub_formula in DNF_formula.formula_list:
        all_candidates = {}
        for term_name in sub_formula.term_dict:
            if sub_formula.has_term_grounded_entity_id_list(term_name):
                all_candidates[term_name] = np.zeros(n_entity)
                all_candidates[term_name][sub_formula.term_grounded_entity_id_dict[term_name][index]] = 1
            else:
                all_candidates[term_name] = np.ones(n_entity)
        sub_graph_edge, sub_graph_negation_edge = [], []
        for pred in sub_formula.predicate_dict.values():
            pred_triples = (pred.head.name, sub_formula.pred_grounded_relation_id_dict[pred.name][index],
                            pred.tail.name)
            if pred.skolem_negation:
                sub_graph_negation_edge.append(pred_triples)
            else:
                sub_graph_edge.append(pred_triples)
        sub_kg_index = KGIndex()
        sub_kg_index.map_entity_name_to_id = {term: 0 for term in sub_formula.term_dict}
        sub_kg = KnowledgeGraph(sub_graph_edge, sub_kg_index)
        neg_kg = KnowledgeGraph(sub_graph_negation_edge, sub_kg_index)
        sub_kg_index.map_relation_name_to_id = {predicate: 0 for predicate in sub_formula.predicate_dict}
        sub_ans = solve_conjunctive(sub_kg, neg_kg, relation_matrix,
                                    all_candidates, conjunctive_tnorm, existential_tnorm, 'f')
        sub_ans_list.append(sub_ans)
    if len(sub_ans_list) == 1:
        return sub_ans_list[0]
    else:
        if conjunctive_tnorm == 'product':
            not_ans = 1 - sub_ans_list[0]
            for i in range(1, len(sub_ans_list)):
                not_ans = not_ans * (1 - sub_ans_list[i])
            return 1 - not_ans
        if conjunctive_tnorm == 'Godel':
            return None
        else:
            raise NotImplementedError


def compute_single_evaluation(fof, batch_ans, n_entity):
    k = 'f'
    metrics = defaultdict(float)
    for i, single_ans in enumerate(batch_ans):
        argsort = torch.argsort(torch.tensor(single_ans), descending=True)
        ranking = argsort.clone().to(torch.float)
        ranking = ranking.scatter_(0, argsort, torch.arange(n_entity).to(torch.float))
        hard_ans = fof.hard_answer_list[i][k]
        easy_ans = fof.easy_answer_list[i][k]
        num_hard = len(hard_ans)
        num_easy = len(easy_ans)
        cur_ranking = ranking[list(easy_ans) + list(hard_ans)]
        cur_ranking, indices = torch.sort(cur_ranking)
        masks = indices >= num_easy
        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
        cur_ranking = cur_ranking - answer_list + 1
        # filtered setting: +1 for start at 0, -answer_list for ignore other answers
        cur_ranking = cur_ranking[masks]
        # only take indices that belong to the hard answers
        mrr = torch.mean(1. / cur_ranking).item()
        h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
        h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
        h10 = torch.mean(
            (cur_ranking <= 10).to(torch.float)).item()
        add_hard_list = torch.arange(num_hard).to(torch.float)
        hard_ranking = cur_ranking + add_hard_list  # for all hard answer, consider other hard answer
        metrics['mrr'] += mrr
        metrics['hit1'] += h1
        metrics['hit3'] += h3
        metrics['hit10'] += h10
    metrics['num_query'] += len(batch_ans)
    return metrics



if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    with open(f'{args.pkl}', 'rb') as data:
        r_matrix_list = pickle.load(data)
    train_dataloader = QueryAnsweringSeqDataLoader_v2(
        osp.join(args.data_folder, 'test-qaa.json'),
        # size_limit=args.batch_size * 1,
        target_lstr=['r1(s1,f)'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)
    fof_list = train_dataloader.get_fof_list()
    t = tqdm.tqdm(enumerate(fof_list), total=len(fof_list))
    all_metrics = defaultdict(dict)
    for ifof, fof in t:
        batch_ans_list, metric = [], {}
        for query_index in range(args.batch_size):
            ans = solve_EFO1(fof, r_matrix_list, 'product', 'Godel', query_index)
            batch_ans_list.append(ans)
        batch_score = compute_single_evaluation(fof, batch_ans_list, 14505)
        for metric in batch_score:
            if metric not in all_metrics[fof.lstr]:
                all_metrics[fof.lstr][metric] = 0
            all_metrics[fof.lstr][metric] += batch_score[metric]
    for full_formula in all_metrics.keys():
        for log_metric in all_metrics[full_formula].keys():
            if log_metric != 'num_queries':
                all_metrics[full_formula][log_metric] /= all_metrics[full_formula]['num_queries']
    print(all_metrics)
