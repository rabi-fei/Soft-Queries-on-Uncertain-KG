import argparse
from collections import defaultdict
import logging
import os
import os.path as osp

import torch
import tqdm
from torch import nn
import torch.nn.functional as F
import numpy as np

from src.language.tnorm import GodelTNorm, ProductTNorm
from src.pipeline.reasoning_machine import GradientReasoningMachine
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import NeuralBinaryPredicate, TransE
from src.utils.data import (QueryAnsweringSeqDataLoader,
                            TrainRandomSentencePairDataLoader)

parser = argparse.ArgumentParser()

# base environment
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--output_dir", type=str, default='log')

# input task folder, defines knowledge graph, index, and formulas
parser.add_argument("--task_folder", type=str, default='data/FB15k-237-betae')

# model, defines the neural binary predicate
parser.add_argument("--model_name", type=str, default='transe')
parser.add_argument("--embedding_dim", type=int, default=300)
parser.add_argument("--margin", type=float, default=20)
parser.add_argument("--p", type=int, default=1)

# optimization
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=1e-1)
parser.add_argument("--reasoning_rate", type=float, default=1e-1)
parser.add_argument("--objective", type=str, choices=['kvsall', 'noisy', 'none'], default='none')
parser.add_argument("--noisy_sample_size", type=int, default=128)


def train_epoch_noisy_v2(desc, train_dataloader, nbp: NeuralBinaryPredicate, grm: GradientReasoningMachine, args):
    optimizer = torch.optim.Adam(nbp.parameters(), args.learning_rate)

    with tqdm.tqdm(enumerate(train_dataloader), desc=desc, total=len(train_dataloader)) as t:
        trajectory = defaultdict(list)
        for i, (pos_fof, neg_fof) in t:
            metric_step = defaultdict(list)
            ####################
            optimizer.zero_grad()
            pos_fetched, neg_fetched = grm.reasoning((pos_fof, neg_fof), all_candidates=False, infer_free=False)

            pos_answer_sizes = [len(gdict['f']) for gdict in pos_fof.grounding_dict_list]
            pos_tv_list = torch.split(pos_fetched['tv'], pos_answer_sizes)

            neg_answer_sizes = [len(gdict['f']) for gdict in neg_fof.grounding_dict_list]
            neg_tv_list = torch.split(neg_fetched['tv'], neg_answer_sizes)

            batch_size = len(pos_tv_list)

            pos_loss, neg_loss = 0, 0
            for ptv, ntv in zip(pos_tv_list, neg_tv_list):
                pos_loss -= torch.mean(torch.log(ptv + 1e-20)) / batch_size
                neg_loss -= torch.mean(torch.log(1 - ntv + 1e-20)) / batch_size

            loss = pos_loss + neg_loss

            loss.backward()
            optimizer.step()
            ####################
            metric_step['loss'].append(loss.item())

            postfix = {'step': i+1}
            for k in metric_step:
                postfix[k] = np.mean(metric_step[k])
                trajectory[k].append(postfix[k])
            logging.info(f"[{desc}] {postfix}")
            postfix['acc_loss'] = np.mean(trajectory['loss'])
            t.set_postfix(postfix)

        metric = {'step': i+1}
        for k in trajectory:
            metric[k] = np.mean(trajectory[k])
    return metric



def train_epoch_noisy(desc, train_dataloader, nbp: NeuralBinaryPredicate, grm: GradientReasoningMachine, args):
    optimizer = torch.optim.Adam(nbp.parameters(), args.learning_rate)

    with tqdm.tqdm(enumerate(train_dataloader), desc=desc, total=len(train_dataloader)) as t:
        trajectory = defaultdict(list)
        for i, fofs in t:
            ####################
            optimizer.zero_grad()
            fetched = grm.reasoning(fofs, all_candidates=True)

            metric_step = defaultdict(list)
            loss = 0

            for fof, fof_reasoning_kv in zip(fofs, fetched):
                # batch_size, all entities
                batch_truth_value_of_grounded_cases = torch.transpose(
                    fof_reasoning_kv['tv'], dim0=0, dim1=1)
                batch_size, answer_size = batch_truth_value_of_grounded_cases.shape

                # only works for single free variable with name
                true_answers = []
                tv4target = 0
                tv4noisy = 0
                for j, easy_answer in enumerate(fof.easy_answer_list):
                    target_index = torch.tensor(easy_answer['f'],
                                                device=args.device)
                    target_one_hot = torch.sum(
                        F.one_hot(target_index, num_classes=answer_size),
                        dim=0,
                        keepdim=True
                    )
                    true_answers.append(target_one_hot)

                    noisy_index = torch.randint(low=0,
                                                high=nbp.num_entities,
                                                size=(128,),
                                                device=args.device)
                    tv4target -= torch.mean(torch.log(
                        batch_truth_value_of_grounded_cases[j, target_index] + 1e-10))
                    tv4noisy -= torch.mean(torch.log( 1 -
                        batch_truth_value_of_grounded_cases[j, noisy_index] + 1e-10))

                    this_loss = tv4target + tv4noisy

                multi_true_answer_tensor = torch.cat(true_answers, dim=0)

                loss += this_loss / len(fof.easy_answer_list)
                metric_step['loss'].append(this_loss.item())

                true_positive = torch.sum((multi_true_answer_tensor * batch_truth_value_of_grounded_cases) > 0.5, -1).tolist()
                all_true = torch.sum(multi_true_answer_tensor, -1).tolist()
                all_positive = torch.sum(batch_truth_value_of_grounded_cases > 0.5, -1).tolist()
                precision, recall = [], []
                for tp, at, ap in zip(true_positive, all_true, all_positive):
                    precision.append(tp / ap if ap > 0 else 0)
                    recall.append(tp / at)
                metric_step['precision'].extend(precision)
                metric_step['recall'].extend(recall)
                metric_step['all_possitive'].extend(all_positive)

            loss.backward()
            optimizer.step()
            ####################

            postfix = {'step': i+1}
            for k in metric_step:
                postfix[k] = np.mean(metric_step[k])
                trajectory[k].append(postfix[k])
            logging.info(f"[{desc}] {postfix}")
            postfix['acc_loss'] = np.mean(trajectory['loss'])
            t.set_postfix(postfix)

        metric = {'step': i+1}
        for k in trajectory:
            metric[k] = np.mean(trajectory[k])
    return metric


# The K-verses-all objective, shown to be suboptimal
def train_epoch_K_verses_All(desc, train_dataloader, nbp: NeuralBinaryPredicate, grm: GradientReasoningMachine, args):
    optimizer = torch.optim.Adam(nbp.parameters(), args.learning_rate)

    with tqdm.tqdm(enumerate(train_dataloader), desc=desc, total=len(train_dataloader)) as t:
        trajectory = defaultdict(list)
        for i, fofs in t:
            ####################
            optimizer.zero_grad()
            fetched = grm.reasoning(fofs, all_candidates=True)

            metric_step = defaultdict(list)
            loss = 0

            for fof, fof_reasoning_kv in zip(fofs, fetched):
                # batch_size, all entities
                batch_truth_value_of_grounded_cases = torch.transpose(
                    fof_reasoning_kv['tv'], dim0=0, dim1=1)
                batch_size, answer_size = batch_truth_value_of_grounded_cases.shape
                # only works for single free variable with name
                true_answers = []
                for easy_answer in fof.easy_answer_list:
                    target_sparse = torch.tensor(easy_answer['f'], device=args.device)
                    target_one_hot = torch.sum(
                        F.one_hot(target_sparse, num_classes=answer_size),
                        dim=0,
                        keepdim=True
                    )
                    true_answers.append(target_one_hot)

                multi_true_answer_tensor = torch.cat(true_answers, dim=0)
                this_loss = - multi_true_answer_tensor * torch.log(batch_truth_value_of_grounded_cases + 1e-10)
                this_loss -= (1-multi_true_answer_tensor) * torch.log(1-batch_truth_value_of_grounded_cases + 1e-10)
                loss += this_loss.mean()

                metric_step['loss'].append(this_loss.mean().item())

                true_positive = torch.sum((multi_true_answer_tensor * batch_truth_value_of_grounded_cases) > 0.5, -1).tolist()
                all_true = torch.sum(multi_true_answer_tensor, -1).tolist()
                all_positive = torch.sum(batch_truth_value_of_grounded_cases > 0.5, -1).tolist()
                precision, recall = [], []
                for tp, at, ap in zip(true_positive, all_true, all_positive):
                    precision.append(tp / ap if ap > 0 else 0)
                    recall.append(tp / at)

                metric_step['precision'].extend(precision)
                metric_step['recall'].extend(recall)
                metric_step['all_possitive'].extend(all_positive)

            loss.backward()
            optimizer.step()
            ####################


            postfix = {'step': i+1}
            for k in metric_step:
                postfix[k] = np.mean(metric_step[k])
                trajectory[k].append(postfix[k])
            logging.info(f"[{desc}] {postfix}")
            postfix['acc_loss'] = np.mean(trajectory['loss'])
            t.set_postfix(postfix)

        metric = {'step': i+1}
        for k in trajectory:
            metric[k] = np.mean(trajectory[k])
    return metric


def evaluate(desc, dataloader, nbp:NeuralBinaryPredicate, grm: GradientReasoningMachine):
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    with tqdm.tqdm(dataloader, desc=desc, total=len(dataloader)) as t:
        for i, fofs in enumerate(t):
            fetched = grm.reasoning(fofs, all_candidates=False, infer_free=True)
            for fof, fof_reasoning_kv in zip(fofs, fetched):
                fvar_local_emb_dict = fof_reasoning_kv['fvar_local_emb_dict']

                # for each free variable name
                for k, batch_est_emb in fvar_local_emb_dict.items():
                    # [batch_size, num_entities]
                    batch_entity_rankings = nbp.get_all_entity_rankings(batch_est_emb)
                    for i, ranking in enumerate(torch.split(batch_entity_rankings, 1)):
                        ranking = ranking.squeeze()
                        # [1, num_entities]
                        hard_answers = torch.tensor(fof.hard_answer_list[i][k],
                                                    device=nbp.device)
                        hard_answer_rank = ranking[hard_answers]
                        # [1, num_entities]

                        # remove better easy answers from its rankings
                        if fof.easy_answer_list[i][k]:
                            easy_answers = torch.tensor(fof.easy_answer_list[i][k],
                                                        device=nbp.device)
                            easy_answer_rank = ranking[easy_answers].view(-1, 1)

                            num_skipped_answers = torch.sum(
                                hard_answer_rank > easy_answer_rank, dim=0)
                            pure_hard_ans_rank = hard_answer_rank - num_skipped_answers
                        else:
                            pure_hard_ans_rank = hard_answer_rank.squeeze()

                        # remove better hard answers from its ranking
                        _reference_hard_ans_rank = pure_hard_ans_rank.reshape(-1, 1)
                        num_skipped_answers = torch.sum(
                            pure_hard_ans_rank > _reference_hard_ans_rank, dim=0
                        )
                        pure_hard_ans_rank -= num_skipped_answers.reshape(pure_hard_ans_rank.shape)

                        rr = (1 / (1+pure_hard_ans_rank)).detach().cpu().numpy()
                        hit1 = (pure_hard_ans_rank < 1).detach().cpu().numpy()
                        hit3 =  (pure_hard_ans_rank < 3).detach().cpu().numpy()
                        hit10 =  (pure_hard_ans_rank < 3).detach().cpu().numpy()

                        metric[fof.lstr()]['rr'].append(rr.mean())
                        metric[fof.lstr()]['hit1'].append(hit1.mean())
                        metric[fof.lstr()]['hit3'].append(hit3.mean())
                        metric[fof.lstr()]['hit10'].append(hit10.mean())

    sum_metric = defaultdict(dict)
    for k1 in metric:
        for k2 in metric[k1]:
            sum_metric[k1][k2] = np.mean(metric[k1][k2])

    logging.info(f"[{desc}] {sum_metric}")



if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=osp.join(args.output_dir, 'output.log'),
                        format='%(asctime)s %(message)s',
                        level=logging.INFO,
                        filemode='wt')

    kgidx = KGIndex.load(
        osp.join(args.task_folder, "kgindex.json"))

    # train_kg = KnowledgeGraph.create(
    #     osp.join(args.task_folder, "train_kg.tsv"),
    #     kgidx,
    #     device=args.device)
    # valid_kg = KnowledgeGraph.create(
    #     osp.join(args.task_folder, "valid_kg.tsv"),
    #     kgidx,
    #     device=args.device)
    # test_kg = KnowledgeGraph.create(
    #     osp.join(args.task_folder, "test_kg.tsv"),
    #     kgidx,
    #     device=args.device)

    nbp = TransE(
        num_entities=kgidx.num_entities,
        num_relations=kgidx.num_relations,
        embedding_dim=args.embedding_dim,
        p=args.p,
        margin=args.margin,
        device=args.device)

    nbp.to(args.device)

    # this dataloader is for k-vs-all objective. works for noisy v1
    # train_dataloader = QueryAnsweringSeqDataLoader(
    #     osp.join(args.task_folder, 'train-qaa.json'),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=0)

    # for noisy objective
    train_dataloader = TrainRandomSentencePairDataLoader(
        osp.join(args.task_folder, 'train-qaa.json'),
        batch_size=args.batch_size,
        shuffle=True,
        answer_size=kgidx.num_entities,
        noisy_sample_size=args.noisy_sample_size,
        num_workers=0)


    valid_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'valid-qaa.json'),
        batch_size=512,
        shuffle=False,
        num_workers=0
    )

    test_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'test-qaa.json'),
        batch_size=512,
        shuffle=False,
        num_workers=0
    )


    # train_epoch_K_verses_All(f"initial from cold start",
                            #    train_dataloader, nbp, grm0, args)
    for e in range(args.epoch):

        train_grm = GradientReasoningMachine(
            reasoning_rate=args.reasoning_rate,
            reasoning_steps=int(30 * ((e + 1) // args.epoch)),
            reasoning_optimizer='Adam',
            nbp=nbp,
            tnorm=ProductTNorm)
        if args.objective.lower() == 'kvsall':
            train_epoch_K_verses_All(f"training epoch {e}",
                                     train_dataloader, nbp, train_grm, args)
        elif args.objective.lower() == 'noisy':
            train_epoch_noisy_v2(f"training epoch {e}",
                              train_dataloader, nbp, train_grm, args)


        eval_grm = GradientReasoningMachine(
            reasoning_rate=args.reasoning_rate,
            reasoning_steps=30,
            reasoning_optimizer='Adam',
            nbp=nbp,
            tnorm=ProductTNorm)
        evaluate(f"validate epoch {e}",
                 valid_dataloader, nbp, eval_grm)
        evaluate(f"test epoch {e}",
                 test_dataloader, nbp, eval_grm)
