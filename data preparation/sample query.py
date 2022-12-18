import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from src.language.tnorm import GodelTNorm, ProductTNorm, Tnorm
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.utils.data_util import RaggedBatch
from lifted_embedding_estimation_with_truth_value import name2lstr, newlstr2name, lstr2name, DNF_lstr2name
from src.language.grammar import parse_lstr_to_disjunctive_formula
from src.language.fof import Disjunction, ConjunctiveFormula, DisjunctiveFormula
from src.utils.data import (QueryAnsweringMixDataLoader, QueryAnsweringSeqDataLoader,
                            QueryAnsweringSeqDataLoader_v2,
                            TrainRandomSentencePairDataLoader)
data_folder = 'data/FB15k-237-betae'
train_queries = list(name2lstr.values())
query_2in = 'r1(s1,f)&!r2(s2,f)'
query_2i = 'r1(s1,f)&r2(s2,f)'
parser = argparse.ArgumentParser()
#parser.add_argument("--output_name", type=str, default='new-qaa')
parser.add_argument("--output_folder", type=str, default='data')
parser.add_argument("--sample_num", type=int, default=3)
parser.add_argument('--mode', choices=['train', 'valid', 'test'], default='train')
parser.add_argument("--meaningful_negation", type=bool, default=True)


lstr_3pc = '((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2))'
lstr_3pnc = '((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(!(r5(e1,e2)))'
lstr_mi = '(((r1(s1,e1))&(r2(e1,f)))&(r3(e1,f)))&(r4(s2,f))'
lstr_2an = '(r1(e1,f))&(!(r2(s1,f)))'


def double_checking_answer(given_lstr, fof_qa_dict, kg: KnowledgeGraph):
    if kg is None:
        return None
    if given_lstr == lstr_3pc:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        e2_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r3'])]
        all_ans = set()
        for e1_c in e1_candidate:
            f_candidate_set = kg.hr2t[(e1_c, fof_qa_dict['r2'])]
            e2_c_set = kg.hr2t[(e1_c, fof_qa_dict['r5'])].intersection(e2_candidate)
            if e2_c_set:
                f_e2_candidate_set = set.union(*[kg.hr2t[(e2_c, fof_qa_dict['r4'])] for e2_c in e2_c_set])
            else:
                f_e2_candidate_set = {}
            f_final_candidate = f_candidate_set.intersection(f_e2_candidate_set)
            all_ans.update(f_final_candidate)
        return all_ans
    elif given_lstr == lstr_3pnc:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        e2_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r3'])]
        all_ans = set()
        for e1_c in e1_candidate:
            f_candidate_set = kg.hr2t[(e1_c, fof_qa_dict['r2'])]
            e2_c_set = e2_candidate - kg.hr2t[(e1_c, fof_qa_dict['r5'])]
            if e2_c_set:
                f_e2_candidate_set = set.union(*[kg.hr2t[(e2_c, fof_qa_dict['r4'])] for e2_c in e2_c_set])
            else:
                f_e2_candidate_set = {}
            f_final_candidate = f_candidate_set.intersection(f_e2_candidate_set)
            all_ans.update(f_final_candidate)
        return all_ans
    elif given_lstr == lstr_mi:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        f_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r4'])]
        f_candidate2 = set.union(*[kg.hr2t[(e1_c, fof_qa_dict['r2'])].intersection(kg.hr2t[(e1_c, fof_qa_dict['r3'])])
                                   for e1_c in e1_candidate])
        return f_candidate.intersection(f_candidate2)
    elif given_lstr == lstr_2an:
        f_candidate = kg.r2t[fof_qa_dict['r1']]
        f_candidate = f_candidate - kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r2'])]
        return f_candidate
    else:
        return None


def sample_one_formula_query(given_lstr, easy_kg: KnowledgeGraph, hard_kg: KnowledgeGraph, sample_num, sample_mode,
                             meaningful_negation):
    fof = parse_lstr_to_disjunctive_formula(given_lstr)
    now_sample_num = 0
    all_qa_dict = set()
    all_query_list = []
    while now_sample_num < sample_num:
        qa_dict = fof.sample_query(hard_kg, meaningful_negation)
        if qa_dict and str(qa_dict) not in all_qa_dict:  # We notice sampling may fail and return None
            all_qa_dict.add(str(qa_dict))  # remember it to avoid repeat
            fof.append_qa_instances(qa_dict)
            if sample_mode == 'train':
                hard_answer = fof.deterministic_query(now_sample_num, hard_kg)
                easy_answer = set()
            else:
                hard_answer = fof.deterministic_query(now_sample_num, hard_kg)
                easy_answer = fof.deterministic_query(now_sample_num, easy_kg)
            check_easy_ans, check_hard_ans = double_checking_answer(given_lstr, qa_dict, easy_kg), \
                double_checking_answer(given_lstr, qa_dict, hard_kg)
            if check_hard_ans is not None:
                assert hard_answer == check_hard_ans
            if check_easy_ans is not None:
                assert easy_answer == check_easy_ans
            if hard_answer - easy_answer is not None:
                if sample_mode == 'train':
                    new_query = [qa_dict, {'f': list(hard_answer)}, []]
                else:
                    new_query = [qa_dict, {'f': list(easy_answer)}, {'f': list(hard_answer)}]
                all_query_list.append(new_query)
                now_sample_num += 1
    return all_query_list


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    kgidx = KGIndex.load(osp.join(data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(data_folder, 'train_kg.tsv'),
        kgindex=kgidx)
    valid_kg = KnowledgeGraph.create(
        triple_files=osp.join(data_folder, 'valid_kg.tsv'),
        kgindex=kgidx)
    test_kg = KnowledgeGraph.create(
        triple_files=osp.join(data_folder, 'test_kg.tsv'),
        kgindex=kgidx)
    """
    for lstr in DNF_lstr2name:
        test_sample_query(lstr, train_kg)
    """
    all_query_dict = {}
    for lstr in newlstr2name:
        if args.mode == 'train':
            all_query = sample_one_formula_query(lstr, None, train_kg, args.sample_num, args.mode,
                                                 args.meaningful_negation)
        elif args.mode == 'valid':
            all_query = sample_one_formula_query(lstr, train_kg, valid_kg, args.sample_num, args.mode,
                                                 args.meaningful_negation)
        elif args.mode == 'test':
            all_query = sample_one_formula_query(lstr, valid_kg, test_kg, args.sample_num, args.mode,
                                                 args.meaningful_negation)
        else:
            raise NotImplementedError
        all_query_dict[lstr] = all_query
    with open(osp.join(args.output_folder, f'{args.mode}_real_EFO1_qaa.json'), 'wt') as f:
        print([k for k in all_query_dict])
        json.dump(all_query_dict, f)
