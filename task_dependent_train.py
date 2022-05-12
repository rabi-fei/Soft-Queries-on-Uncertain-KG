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
parser.add_argument("--embedding_dim", type=int, default=100)
parser.add_argument("--margin", type=float, default=5)
parser.add_argument("--p", type=int, default=1)

# optimization
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=1e-1)


def train_epoch_K_verses_All(desc, train_dataloader, nbp: NeuralBinaryPredicate, grm: GradientReasoningMachine, args):
    loss_func = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(nbp.parameters(), args.learning_rate)

    with tqdm.tqdm(enumerate(train_dataloader), desc=desc) as t:
        for i, fofs in t:
            metric_step = defaultdict(list)
            ####################
            optimizer.zero_grad()
            fetched = grm.reasoning(fofs, all_candidates=True)
            loss = 0
            for fof, fof_reasoning_kv in zip(fofs, fetched):
                # batch_size, all entities
                batch_truth_value_of_grounded_cases = torch.transpose(
                    fof_reasoning_kv['tv'], dim0=0, dim1=1)
                batch_size, answer_size = batch_truth_value_of_grounded_cases.shape
                # only works for single free variable with name f
                true_answers = []
                for easy_answer in fof.easy_answer_list:
                    target_sparse = torch.tensor(easy_answer['f'])
                    target_one_hot = torch.sum(
                        F.one_hot(target_sparse, num_classes=answer_size),
                        dim=0,
                        keepdim=True
                    )
                    true_answers.append(target_one_hot)

                multi_true_answer_tensor = torch.cat(true_answers, dim=0)

                loss += loss_func(batch_truth_value_of_grounded_cases,
                                    multi_true_answer_tensor)
            loss.backward()
            optimizer.step()
            ####################
            metric_step['loss'].append(loss.item())

            logging.info(f"[Train]"
                        f"batch_step: {i+1};"
                        f"average K verses All loss: {np.mean(metric_step['loss'])}")
            postfix = {
                "step": i+1,
                "average loss": np.mean(metric_step['loss'])
            }
            t.set_postfix(postfix)
    return postfix


def evaluate(desc, dataloader, nbp:NeuralBinaryPredicate, grm: GradientReasoningMachine):
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    with tqdm.tqdm(dataloader, desc=desc) as t:
        for i, fofs in enumerate(t):
            fetched = grm.reasoning(fofs, all_candidates=False)
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
                        if fof.easy_answer_list[i][k]:
                            easy_answers = torch.tensor(fof.easy_answer_list[i][k],
                                                        device=nbp.device)
                            easy_answer_rank = ranking[easy_answers].view(-1, 1)

                            num_skipped_answers = torch.sum(
                                hard_answer_rank > easy_answer_rank, dim=0)
                            pure_hard_ans_rank = hard_answer_rank - num_skipped_answers
                        else:
                            pure_hard_ans_rank = hard_answer_rank.squeeze()

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
        for k2 in metric:
            metric[k1][k2] = np.mean(metric[k1][k2])

    logging.info(f"{sum_metric}")






if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=osp.join(args.output_dir, 'output.log'),
                        format='%(asctime)s %(message)s',
                        level=logging.INFO)

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

    # may change to other ways of teps
    train_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'train-qaa.json'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0)

    valid_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'valid-qaa.json'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'test-qaa.json'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    grm = GradientReasoningMachine(
        reasoning_rate=1e-1,
        reasoning_steps=20,
        reasoning_optimizer='Adam',
        nbp=nbp)

    for e in range(args.epoch):
        # train_epoch_K_verses_All(f"training epoch {e}",
                                #  train_dataloader, nbp, grm, args)
        evaluate(f"validate epoch {e}",
                 valid_dataloader, nbp, grm)
        evaluate(f"test epoch {e}",
                 test_dataloader, nbp, grm)
