import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict
from typing import List

import torch
import scipy.sparse
import pickle

from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default='../cqd/models')
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-betae')


def create_matrix_statistics(scoring_matrix, observed_kg: KnowledgeGraph, latent_kg: KnowledgeGraph):
    n_rel, n_entity = scoring_matrix.shape[0], scoring_matrix.shape[1]
    only_hard_ans, easy_hard_ans, non_ans = [], [], []
    full_tail_prob = torch.softmax(scoring_matrix, dim=2)
    for rel_id in range(n_rel):
        for h_id in range(n_entity):
            tail_prob = full_tail_prob[rel_id][h_id]
            tail_set = observed_kg.hr2t[(h_id, rel_id)]
            hard_tail_set = latent_kg.hr2t[(h_id, rel_id)] - tail_set
            observed_t_num = len(tail_set)
            observed_edge_prob_vec = tail_prob[list(tail_set)]
            observed_edge_prob = torch.sum(observed_edge_prob_vec)
            hard_edge_prob_vec = tail_prob[list(hard_tail_set)]
            if tail_set and hard_tail_set:
                easy_hard_ans.append([observed_edge_prob_vec, hard_edge_prob_vec])
            if hard_tail_set and not tail_set:
                #max_hard_ans, least_hard_ans = min(hard_edge_prob_vec), max(hard_edge_prob_vec)
                only_hard_ans.append(hard_edge_prob_vec)
            if not hard_tail_set and not tail_set:
                max_non_answer = max(tail_prob)
                non_ans.append(max_non_answer)
    return only_hard_ans, easy_hard_ans, non_ans


def create_matrix_from_ckpt(scoring_matrix, observed_kg: KnowledgeGraph, threshold=0.01, epsilon=0.01):
    n_rel, n_entity = scoring_matrix.shape[0], scoring_matrix.shape[1]
    full_tail_prob = torch.softmax(scoring_matrix, dim=2)
    for rel_id in range(n_rel):
        for h_id in range(n_entity):
            tail_prob = full_tail_prob[rel_id][h_id]
            tail_set = observed_kg.hr2t[(h_id, rel_id)]
            observed_t_num = len(tail_set)
            scailing = observed_t_num/torch.sum(tail_prob[list(tail_set)]) if observed_t_num else 1
            full_tail_prob[rel_id][h_id] *= scailing
            full_tail_prob[rel_id][h_id] = torch.where(full_tail_prob[rel_id][h_id] > threshold,
                                                       full_tail_prob[rel_id][h_id], torch.zeros(n_entity))
            full_tail_prob[rel_id][h_id] = full_tail_prob[rel_id][h_id].clamp(0, 1-epsilon)
            full_tail_prob[rel_id][h_id][list(tail_set)] = 1
    return full_tail_prob


def collect_matrix_scipy_sparse(matrix_path):
    dense_matrix = torch.load(matrix_path)
    matrix_list = []
    for i in range(dense_matrix.shape[0]):
        sparse_matrix = scipy.sparse.csr_matrix(dense_matrix[i])
        matrix_list.append(sparse_matrix)
    return matrix_list


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    kgidx = KGIndex.load(osp.join(args.data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'train_kg.tsv'),
        kgindex=kgidx)
    '''
    test_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'test_kg.tsv'),
        kgindex=kgidx)
    '''
    threshold, epsilon = 0.01, 0.01
    split_num = int(79)
    split_each_num = int(train_kg.num_relations / split_num)
    all_matrix_list = []
    for split_id in range(split_num):
        #whole_prob_matrix = torch.zeros(split_each_num, train_kg.num_entities, train_kg.num_entities)
        matrix_path = f'matrix/matrix_{split_id}_{threshold}_{epsilon}.ckpt'
        if osp.exists(matrix_path):
            part_matrix_list = collect_matrix_scipy_sparse(matrix_path)
            all_matrix_list.extend(part_matrix_list)
            continue
        score_matrix = torch.load(osp.join(args.ckpt, f'matrix_{split_id}.ckpt'), map_location=None)
        prob_matrix = create_matrix_from_ckpt(score_matrix, train_kg, threshold, epsilon)
        #whole_prob_matrix[int(split_id * split_each_num): int(split_id * split_each_num + split_each_num)] = prob_matrix
        torch.save(prob_matrix, matrix_path)
    with open(f'sparse/scipy_{threshold}_{epsilon}.pickle', 'wb') as handle:
        pickle.dump(all_matrix_list)


