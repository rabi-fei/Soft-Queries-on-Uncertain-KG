import argparse
import json
import logging
import os
import math
import os.path as osp
import random
from collections import defaultdict
from typing import List
import copy

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
import scipy.stats as stats
import tqdm
import pickle
from torch import nn
from scipy.sparse import csc_matrix, diags, issparse

from src.language.foq import ConjunctiveFormula, DisjunctiveFormula
from src.structure.knowledge_graph import KnowledgeGraph, kg_remove_node
from src.structure.knowledge_graph_index import KGIndex
from src.utils.data import QueryAnsweringSeqDataLoader_v2
from src.utils.class_util import Writer
from src.utils.data_util import RaggedBatch
from train_lmpnn import compute_evaluation_scores

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--sleep", type=int, default=0)
parser.add_argument("--ckpt", type=str, default='checkpoints/ppi5k/full_matrix_list_0.01_0.001.ckpt')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--cuda", type=int, default=1)
parser.add_argument("--data_folder", type=str, default='data/ppi5k')
parser.add_argument("--mode", type=str, default='test', choices=['valid', 'test'])
parser.add_argument("--e_norm", type=str, default='Godel', choices=['Godel', 'product'])
parser.add_argument("--c_norm", type=str, default='Product', choices=['Plus', 'Godel', 'Product'])
parser.add_argument("--max", type=int, default=10)
parser.add_argument("--data_type", type=str, default='soft_EFO1')
parser.add_argument("--formula", type=list, default=["r1(s1,f1,25%,1.0)"])
negation_list = ['(r1(s1,f))&(!(r2(s2,f)))', '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))',
                 '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))', '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))',
                 '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))']
m_list = ['((r1(s1,e1))&(r2(e1,f)))&(r3(e1,f))', '((r1(s1,e1))&(r2(e1,f)))&(!(r3(e1,f)))', '(((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f)))&(r4(e1,e2))', '(((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f)))&(r4(e2,f))', '(((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f)))&(r4(e1,f))']

@torch.no_grad()
def solve_soft_conjunctive(positive_graph: KnowledgeGraph, negative_graph: KnowledgeGraph, relation_matrix, necess_confidence,
                      now_candidate_set: dict, conjunctive_tnorm, existential_tnorm, now_variable, device,
                      max_enumeration):
    n_entity = relation_matrix[0].shape[0]
    if not positive_graph.facts and not negative_graph.facts:
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
            sub_ans = solve_soft_conjunctive(sub_pos_g, sub_neg_g, relation_matrix, necess_confidence, now_candidate_set,
                                        conjunctive_tnorm, existential_tnorm, next_variable, device, max_enumeration)
            final_ans = extend_ans(now_leaf_node, adjacency_node, positive_graph, negative_graph, relation_matrix, necess_confidence,
                                   now_candidate_set[now_leaf_node], sub_ans, conjunctive_tnorm, existential_tnorm)
            return final_ans
        else:
            answer = cut_node_sub_problem_soft(now_leaf_node, adjacency_node_list, positive_graph, negative_graph,
                                          relation_matrix, necess_confidence, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                          now_variable, device, max_enumeration)
            return answer
    else:
        to_enumerate_node, adjacency_node_list = find_enumerate_node(positive_graph, negative_graph, now_candidate_set,
                                                                     now_variable)
        if max_enumeration:
            easy_candidate = torch.count_nonzero(now_candidate_set[to_enumerate_node] == 1)
            enumeration_num = torch.count_nonzero(now_candidate_set[to_enumerate_node])
            max_enumeration_here = max_enumeration + easy_candidate
            to_enumerate_candidates = torch.argsort(now_candidate_set[to_enumerate_node],
                                                    descending=True)[:min(max_enumeration_here, enumeration_num)]
        else:
            to_enumerate_candidates = now_candidate_set[to_enumerate_node].nonzero()
        this_node_candidates = copy.deepcopy(now_candidate_set[to_enumerate_node])
        all_enumerate_ans = torch.zeros((to_enumerate_candidates.shape[0], n_entity)).to(device)
        if to_enumerate_candidates.shape[0] == 0:
            return torch.zeros(n_entity).to(device)
        for i, enumerate_candidate in enumerate(to_enumerate_candidates):
            single_candidate = torch.zeros_like(now_candidate_set[to_enumerate_node]).to(device)
            candidate_truth_value = this_node_candidates[enumerate_candidate]
            single_candidate[enumerate_candidate] = 1
            now_candidate_set[to_enumerate_node] = single_candidate
            answer = cut_node_sub_problem_soft(to_enumerate_node, adjacency_node_list, positive_graph, negative_graph,
                                          relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                          now_variable, device, max_enumeration)
            if conjunctive_tnorm == 'product':
                enumerate_ans = candidate_truth_value * answer
            elif conjunctive_tnorm == 'Godel':
                enumerate_ans = torch.minimum(candidate_truth_value, answer)
            else:
                raise NotImplementedError
            all_enumerate_ans[i] = enumerate_ans
        if existential_tnorm == 'Godel':
            final_ans = torch.amax(all_enumerate_ans, dim=-2)
        else:
            raise NotImplementedError
        return final_ans


def find_leaf_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable):
    """
    Find a leaf node with least possible candidate. The now-asking variable is first.
    """
    return_candidate = [None, None, 0]
    for node in now_candidate:
        adjacency_node_set = set.union(
            *[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
              neg_sub_graph.t2h[node]])
        if "s" in node:
            return [node, list(adjacency_node_set)[0], False]
        if len(adjacency_node_set) == 1:
            if node == now_variable:
                continue
                return node, list(adjacency_node_set)[0], True
            candidate_num = torch.count_nonzero(now_candidate[node])
            if not return_candidate[0] or candidate_num < return_candidate[2]:
                return_candidate = [node, list(adjacency_node_set)[0], candidate_num]
    return return_candidate[0], return_candidate[1], False


def find_enumerate_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable):
    return_candidate = [None, 100, 100000]
    for node in now_candidate:
        if node == now_variable:
            continue
        adjacency_node_list = list(set.union(*[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
                                               neg_sub_graph.t2h[node]]))
        adjacency_node_num = len(adjacency_node_list)
        candidate_num = torch.count_nonzero(now_candidate[node])
        if not return_candidate[0] or adjacency_node_num < len(return_candidate[1]) or \
                (adjacency_node_num == len(return_candidate[1]) and candidate_num < return_candidate[2]):
            return_candidate = node, adjacency_node_list, candidate_num
    return return_candidate[0], return_candidate[1]


@torch.no_grad()
def cut_node_sub_problem_soft(to_cut_node, adjacency_node_list, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                         r_matrix_list, necess_confidence, now_candidate_set, conj_tnorm, exist_tnorm, now_variable, device,
                         max_enumeration):
    new_candidate_set = copy.deepcopy(now_candidate_set)
    for adjacency_node in adjacency_node_list:
        if isinstance(now_candidate_set[to_cut_node], int):
            adj_candidate_vec = constant_update(to_cut_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list, necess_confidence,
                                               new_candidate_set[to_cut_node], new_candidate_set[adjacency_node],
                                               conj_tnorm, exist_tnorm)
        else:
            adj_candidate_vec = existential_update(to_cut_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list, necess_confidence,
                                               new_candidate_set[to_cut_node], new_candidate_set[adjacency_node],
                                               conj_tnorm, exist_tnorm)       
        new_candidate_set[adjacency_node] = adj_candidate_vec
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    new_candidate_set.pop(to_cut_node)
    sub_answer = solve_soft_conjunctive(new_sub_graph, new_sub_neg_graph, r_matrix_list, necess_confidence, new_candidate_set, conj_tnorm,
                                   exist_tnorm, now_variable, device, max_enumeration)
    return sub_answer


@torch.no_grad()
def existential_update(leaf_node, adjacency_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                       r_matrix_list, necess_confidence, leaf_candidates, adj_candidates, conj_tnorm, exist_tnorm) -> dict:
    all_prob_matrix = construct_matrix_list(leaf_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list, necess_confindence,
                                            conj_tnorm)
    if conj_tnorm == 'Plus':
        all_prob_matrix.mul_(leaf_candidates.unsqueeze(-1))
        all_prob_matrix += adj_candidates.unsqueeze(-2)
    elif conj_tnorm == 'Product':
        all_prob_matrix.mul_(leaf_candidates.unsqueeze(-1))
        all_prob_matrix.mul_(adj_candidates.unsqueeze(-2))
    elif conj_tnorm == 'Godel':
        all_prob_matrix = torch.minimum(all_prob_matrix, leaf_candidates.unsqueeze(-1))
        all_prob_matrix = torch.minimum(all_prob_matrix, adj_candidates.unsqueeze(-2))
    else:
        raise NotImplementedError
    if exist_tnorm == 'Godel':
        prob_vec = torch.amax(all_prob_matrix, dim=-2).squeeze()
    else:
        raise NotImplementedError
    return prob_vec


@torch.no_grad()
def constant_update(constant_node, adjacency_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                       r_matrix_list, necess_confidence, constant_ground, adj_candidates, conj_tnorm, exist_tnorm) -> dict:
    node_pair, reverse_node_pair = (constant_node, adjacency_node), (adjacency_node, constant_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]

    candi_vec_list, alpha_list, beta_list = [], [], []

    for r in h2t_relation:
        ab = [(rab[1], rab[2]) for rab in sub_graph.ht2rab[(constant_node, adjacency_node)] if rab[0] ==r][0]
        alpha, beta =necess_confidence[f"{r}"][int(ab[0][:-1])//25-1], ab[1]
        alpha_list.append(alpha)
        beta_list.append(beta)
        relation_matrix = r_matrix_list[r].to_dense()
        candi_vec_list.append(relation_matrix[constant_ground])

    for r in t2h_relation:
        ab = [(rab[1], rab[2]) for rab in sub_graph.ht2rab[(constant_node, adjacency_node)] if rab[0] ==r][0]
        alpha, beta =necess_confidence[f"{r}"][int(ab[0][:-1])//25-1], ab[1]
        alpha_list.append(alpha)
        beta_list.append(beta)
        relation_matrix = r_matrix_list[r].to_dense().transpose_()
        candi_vec_list.append(relation_matrix[constant_ground])
        candi_vec_list.append(relation_matrix[constant_ground])

    for r in h2t_negation:
        ab = [(rab[1], rab[2]) for rab in neg_sub_graph.ht2rab[(constant_node, adjacency_node)] if rab[0] ==r][0]
        alpha, beta =necess_confidence[f"{r}"][int(ab[0][:-1])//25-1], ab[1]
        alpha_list.append(alpha)
        beta_list.append(beta)
        relation_matrix = r_matrix_list[r].to_dense()
        candi_vec_list.append(1 - relation_matrix[constant_ground])

    for r in t2h_negation:
        ab = [(rab[1], rab[2]) for rab in neg_sub_graph.ht2rab[(constant_node, adjacency_node)] if rab[0] ==r][0]
        alpha, beta =necess_confidence[f"{r}"][int(ab[0][:-1])//25-1], ab[1]
        alpha_list.append(alpha)
        beta_list.append(beta)
        relation_matrix = r_matrix_list[r].to_dense().transpose_()
        candi_vec_list.append(1 - relation_matrix[constant_ground])

    del relation_matrix

    if conj_tnorm == 'Plus':
        candi_vec = adj_candidates
        for i in range(0, len(candi_vec_list)):
            candi_vec += beta_list[i] * (candi_vec_list[i] >= alpha_list[i]) * candi_vec
    elif conj_tnorm == 'Product':
        candi_vec = adj_candidates
        for i in range(0, len(candi_vec_list)):
            a, b = alpha_list[i], beta_list[i]
            exp_candi_vec = torch.exp(b * candi_vec_list[i]) 
            candi_vec = candi_vec.multiply(exp_candi_vec * (exp_candi_vec >= math.exp(a * b)))
    else:
        raise NotImplementedError
    return candi_vec

@torch.no_grad()
def extend_ans(ans_node, sub_ans_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, relation_matrix, necess_confidence,
               leaf_candidate, sub_ans, conj_tnorm, exist_tnorm):
    all_prob_matrix = construct_matrix_list(sub_ans_node, ans_node, sub_graph, neg_sub_graph, relation_matrix, necess_confidence,
                                            conj_tnorm)
    all_prob_matrix.mul_(sub_ans.unsqueeze(-1))
    if exist_tnorm == 'Godel':
        prob_vec = (torch.amax(all_prob_matrix, dim=-2)).squeeze()  # prob*vec is 1*n  matrix
        del all_prob_matrix
    else:
        raise NotImplementedError
    if conj_tnorm == 'product':
        prob_vec = leaf_candidate * prob_vec
    elif conj_tnorm == 'Godel':
        prob_vec = torch.minimum(leaf_candidate, prob_vec)
    else:
        raise NotImplementedError
    return prob_vec


@torch.no_grad()
def construct_matrix_list(head_node, tail_node, sub_graph, neg_sub_graph, relation_matrix_list, necess_confidence, conj_tnorm):
    node_pair, reverse_node_pair = (head_node, tail_node), (tail_node, head_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]
    transit_matrix_list, alpha_list, beta_list = [], [], []

    for r in h2t_relation:
        ab = [(rab[1], rab[2]) for rab in sub_graph.ht2rab[(head_node, tail_node)] if rab[0] ==r][0]
        alpha, beta =necess_confidence[f"{r}"][int(ab[0][:-1])//25-1], ab[1]
        transit_matrix_list.append(relation_matrix_list[r])
        alpha_list.append(alpha)
        beta_list.append(beta)
    for r in t2h_relation:
        ab = [(rab[1], rab[2]) for rab in sub_graph.ht2rab[(head_node, tail_node)] if rab[0] ==r][0]
        alpha, beta =necess_confidence[f"{r}"][int(ab[0][:-1])//25-1], ab[1]
        transit_matrix_list.append(relation_matrix_list[r].transpose(-2, -1))
        alpha_list.append(alpha)
        beta_list.append(beta)
    for r in h2t_negation:
        ab = [(rab[1], rab[2]) for rab in neg_sub_graph.ht2rab[(head_node, tail_node)] if rab[0] ==r][0]
        alpha, beta =necess_confidence[f"{r}"][int(ab[0][:-1])//25-1], ab[1]
        transit_matrix_list.append(1 - relation_matrix_list[r].to_dense())
        alpha_list.append(alpha)
        beta_list.append(beta)
    for r in t2h_negation:
        ab = [(rab[1], rab[2]) for rab in sub_graph.ht2rab[(head_node, tail_node)] if rab[0] ==r][0]
        alpha, beta =necess_confidence[f"{r}"][int(ab[0][:-1])//25-1], ab[1]
        transit_matrix_list.append(1 - relation_matrix_list[r].transpose(-2, -1).to_dense())
        alpha_list.append(alpha)
        beta_list.append(beta)
    if conj_tnorm == 'Plus':
        all_prob_matrix = beta_list[0] * transit_matrix_list[0] * (transit_matrix_list[0].to_dense() >= alpha_list[0])
        for i in range(1, len(transit_matrix_list)):
            trans_matrix_i = transit_matrix_list[i].to_dense()
            all_prob_matrix += beta_list[i] * trans_matrix_i * (trans_matrix_i >= alpha_list[i])
    elif conj_tnorm == 'Product':
        exp_pro_matrix = torch.exp(beta_list[0] * transit_matrix_list[0].to_dense())
        all_prob_matrix = exp_pro_matrix * (exp_pro_matrix >= math.exp(alpha_list[0] * beta_list[0]))
        for i in range(1, len(transit_matrix_list)):
            exp_pro_matrix = torch.exp(beta_list[i] * transit_matrix_list[i].to_dense())
            all_prob_matrix = all_prob_matrix.multiply(exp_pro_matrix * (exp_pro_matrix >= math.exp(alpha_list[0] * beta_list[0])))
    else:
        raise NotImplementedError

    if all_prob_matrix.is_sparse:  # n*n sparse matrix or dense matrix (when only one negation edges)
        return all_prob_matrix.to_dense()
    else:
        return all_prob_matrix


@torch.no_grad()
def solve_soft_EFO1(DNF_formula: DisjunctiveFormula, relation_matrix, necess_confidence, conjunctive_tnorm, existential_tnorm, index, device,
               max_enumeration):
    torch.cuda.empty_cache()
    with torch.no_grad():
        sub_ans_list = []
        n_entity = relation_matrix[0].shape[0]
        for sub_formula in DNF_formula.formula_list:
            all_candidates = {}
            for term_name in sub_formula.term_dict:
                if sub_formula.has_term_grounded_entity_id_list(term_name):
                    all_candidates[term_name] = sub_formula.term_grounded_entity_id_dict[term_name][index]
                else:
                    if conjunctive_tnorm == "Plus":
                        all_candidates[term_name] = torch.zeros(n_entity).to(device)
                    else:
                        all_candidates[term_name] = torch.ones(n_entity).to(device)
            sub_graph_edge, sub_graph_negation_edge = [], []
            for pred in sub_formula.predicate_dict.values():
                pred_triples = (pred.head.name, sub_formula.pred_grounded_relation_id_dict[pred.name][index],
                                pred.tail.name, pred.alpha, pred.beta)
                if pred.negated:
                    sub_graph_negation_edge.append(pred_triples)
                else:
                    sub_graph_edge.append(pred_triples)
            sub_kg_index = KGIndex()
            sub_kg_index.map_entity_name_to_id = {term: 0 for term in sub_formula.term_dict}
            sub_kg = KnowledgeGraph(sub_graph_edge, sub_kg_index)
            neg_kg = KnowledgeGraph(sub_graph_negation_edge, sub_kg_index)
            sub_kg_index.map_relation_name_to_id = {predicate: 0 for predicate in sub_formula.predicate_dict}
            sub_ans = solve_soft_conjunctive(sub_kg, neg_kg, relation_matrix, necess_confidence,
                                        all_candidates, conjunctive_tnorm, existential_tnorm, 'f1', device,
                                        max_enumeration)

            sub_ans_list.append(sub_ans)
        if len(sub_ans_list) == 1:
            return sub_ans_list[0]
        else:

            final_ans = sub_ans_list[0]
            for i in range(1, len(sub_ans_list)):
                final_ans = torch.maximum(final_ans, sub_ans_list[i])
            return final_ans


@torch.no_grad()
def compute_single_evaluation(fof, batch_ans_tensor, n_entity):
    k = 'f1'
    metrics = defaultdict(float)
    argsort = torch.argsort(batch_ans_tensor, dim=1, descending=True)
    ranking = argsort.clone().to(torch.float).to(cuda_device)
    ranking = ranking.scatter_(1, argsort, torch.arange(n_entity).to(torch.float).
                               repeat(argsort.shape[0], 1).to(cuda_device))
    for i in range(batch_ans_tensor.shape[0]):
        # ranking = ranking.scatter_(0, argsort, torch.arange(n_entity).to(torch.float))

        ans = fof.easy_answer_list[i][f"{k}_answers"]
        value = fof.hard_answer_list[i][f"{k}_values"]
        num_ans = len(ans)
        
        ans = torch.tensor(ans).to(ranking.device)
        range_to_num = torch.arange(1, num_ans+1).to(torch.float32).to(ranking.device)

        cur_ranking = ranking[i, ans] + 1
        rr_score = 1. / cur_ranking
        predict_exist_ele = argsort[i][:num_ans].unsqueeze(-1) == ans.squeeze(0)
        predict_exist = torch.any(predict_exist_ele, dim=-1)

        '''
        if easy_ans:
            easy_mrr = torch.mean(1. / easy_ranking).item()
            metrics['easy_queries'] += 1
        else:
            easy_mrr = 0
        metrics['easy_MRR'] += easy_mrr
        '''
        dcg = torch.sum(rr_score / torch.log2(range_to_num+1), dim=-1).item()
        idcg = torch.sum(1 / (range_to_num * torch.log2(range_to_num+1)), dim=-1).item()
        ndcg = dcg / idcg
        map = torch.mean((torch.cumsum(predict_exist, dim=-1)  / range_to_num) * predict_exist, -1).item()

        kendalltau = stats.kendalltau(range_to_num.cpu(), cur_ranking.cpu())

        metrics['MAP'] += map
        metrics['DCG'] += dcg
        metrics['NDCG'] += ndcg
        if  not -1 <= kendalltau[0] <= 1:
            print(cur_ranking.item())
            metrics['tau'] += 1 / cur_ranking.item()
        else:
            metrics['tau'] += kendalltau[0]

    metrics['num_queries'] += batch_ans_tensor.shape[0]
    return metrics


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    torch.set_default_dtype(torch.float16)
    r_matrix_list = torch.load(args.ckpt)
    with open(osp.join(args.data_folder, "percentile_25_50_75.json"), "r") as f:
        necess_confindence = json.load(f)
    n_relation, n_entity = len(r_matrix_list), r_matrix_list[0].shape[0]
    requirement, importance = "normal", "equal"
    if args.cuda < 0 :
        cuda_device = torch.device('cpu')
    else:
        cuda_device = torch.device('cuda:{}'.format(args.cuda))
    for i in range(len(r_matrix_list)):
        r_matrix_list[i] = r_matrix_list[i].to(dtype=torch.float16).to(cuda_device)
    formula_path = osp.join(args.data_folder, 'test_type0000_soft_efo1_qaa.json')

    test_dataloader = QueryAnsweringSeqDataLoader_v2(
        formula_path,
        # size_limit=args.batch_size * 1,
        target_lstr=args.formula,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)
    writer = Writer(case_name=args.ckpt, config=args, log_path='results')
    fof_list = test_dataloader.get_fof_list_no_shuffle()
    t = tqdm.tqdm(enumerate(fof_list), total=len(fof_list))
    all_metrics = defaultdict(dict)
    # all_answers, now_formula_index = {}, {}
    # for lstr in test_dataloader.lstr_qaa:
    # all_answers[lstr] = torch.zeros((len(test_dataloader.lstr_qaa[lstr]), n_entity))
    # now_formula_index[lstr] = 0
    for ifof, fof in t:
        torch.cuda.empty_cache()
        batch_ans_list, metric = [], {}
        for query_index in range(len(fof.easy_answer_list)):
            ans = solve_soft_EFO1(fof, r_matrix_list, necess_confindence, args.c_norm, args.e_norm, query_index, cuda_device, args.max)
            batch_ans_list.append(ans)
        batch_ans_tensor = torch.stack(batch_ans_list, dim=0)
        # all_answers[fof.lstr][now_formula_index[fof.lstr]: now_formula_index[fof.lstr] + batch_ans_tensor.shape[0], :] \
        # = batch_ans_tensor
        # now_formula_index[fof.lstr] += batch_ans_tensor.shape[0]
        batch_score = compute_single_evaluation(fof, batch_ans_tensor, n_entity)
        for metric in batch_score:
            if metric not in all_metrics[fof.lstr]:
                all_metrics[fof.lstr][metric] = 0
            all_metrics[fof.lstr][metric] += batch_score[metric]
        del batch_score, batch_ans_tensor
    for full_formula in all_metrics.keys():
        for log_metric in all_metrics[full_formula].keys():
            if log_metric != 'num_queries':
                all_metrics[full_formula][log_metric] /= all_metrics[full_formula]['num_queries']
    print(all_metrics)
    # writer.save_torch(all_answers, 'all_answer_tensor.ckpt')
    writer.save_pickle(all_metrics, f"all_logging_{args.mode}_0.pickle")
