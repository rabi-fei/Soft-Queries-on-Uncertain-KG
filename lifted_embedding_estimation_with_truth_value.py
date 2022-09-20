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
from src.pipeline.reasoning_machine import (DeepsetEFOReasoner,
                                            GradientEFOReasoner, Reasoner,
                                            GNNEFOReasonerComplEx,
                                            RelationalDeepSet,
                                            LogicalGNNLayerComplEx)
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.utils.data import (QueryAnsweringMixDataLoader,
                            QueryAnsweringSeqDataLoader, RaggedBatch,
                            TrainRandomSentencePairDataLoader)

torch.autograd.set_detect_anomaly(True)

lstr2name = {
    'r1(s1,f)': '1p',
    '(r1(s1,e1))&(r2(e1,f))': '2p',
    '((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f))': '3p',
    '(r1(s1,f))&(r2(s2,f))': '2i',
    '((r1(s1,f))&(r2(s2,f)))&(r3(s3,f))': '3i',
    '((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f))': 'ip',
    '((r1(s1,e1))&(r2(e1,f)))&(r3(s2,f))': 'pi',
    '(r1(s1,f))&(!(r2(s2,f)))': '2in',
    '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))': '3in',
    '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))': 'inp',
    '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))': 'pin',
    '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))': 'pni',
    '(r1(s1,f))|(r2(s2,f))': '2u',
    '((r1(s1,e1))|(r2(s2,e1)))&(r3(e1,f))': 'up',
    '!((!(r1(s1,f)))&(!(r2(s2,f))))': '2u-dm',
    '!(((!(r1(s1,e1)))|(r2(s2,e1)))&(r3(e1,f)))': 'up-dm'
}

name2lstr = {
    "1p": "r1(s1,f)",
    "2p": "r1(s1,e1)&r2(e1,f)",  # 2p
    "3p": "r1(s1,e1)&r2(e1,e2)&r3(e2,f)",  # 3p
    "2i": "r1(s1,f)&r2(s2,f)",  # 2i
    "3i": "r1(s1,f)&r2(s2,f)&r3(s3,f)",  # 3i
    "ip": "r1(s1,e1)&r2(s2,e1)&r3(e1,f)",  # ip
    "pi": "r1(s1,e1)&r2(e1,f)&r3(s2,f)",  # pi
    "2in": "r1(s1,f)&!r2(s2,f)",  # 2in
    "3in": "r1(s1,f)&r2(s2,f)&!r3(s3,f)",  # 3in
    "inp": "r1(s1,e1)&!r2(s2,e1)&r3(e1,f)",  # inp
    "pin": "r1(s1,e1)&r2(e1,f)&!r3(s2,f)",  # pin
    "pni": "r1(s1,e1)&!r2(e1,f)&r3(s2,f)",  # pni
    "2u": "r1(s1,f)|r2(s2,f)",  # 2u
    "up": "(r1(s1,e1)|r2(s2,e1))&r3(e1,f)",  # up
    "2u-dm": "!(!r1(s1,f)&!r2(s2,f))",  # 2u-dm
    "up-dm": "!(!r1(s1,e1)|r2(s2,e1))&r3(e1,f)",  # up-dm
}


negation_query = [
    "r1(s1,f)&!r2(s2,f)",  # 2in
    "r1(s1,f)&r2(s2,f)&!r3(s3,f)",  # 3in
    "r1(s1,e1)&!r2(s2,e1)&r3(e1,f)",  # inp
    "r1(s1,e1)&r2(e1,f)&!r3(s2,f)",  # pin
    "r1(s1,e1)&!r2(e1,f)&r3(s2,f)",  # pni
]


parser = argparse.ArgumentParser()

# base environment
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--output_dir", type=str, default='log')

# input task folder, defines knowledge graph, index, and formulas
parser.add_argument("--task_folder", type=str, default='data/FB15k-237-betae')
parser.add_argument("--train_queries", action='append')
parser.add_argument("--eval_queries", action='append')

# model, defines the neural binary predicate
parser.add_argument("--model_name", type=str, default='complex')
parser.add_argument("--embedding_dim", type=int, default=1000)
parser.add_argument("--margin", type=float, default=0.1)
parser.add_argument("--scale", type=float, default=0.1)
parser.add_argument("--p", type=int, default=1)
parser.add_argument("--checkpoint_path")

# optimization for the entire process
parser.add_argument("--optimizer", type=str, default='AdamW')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--pretrain_epoch", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--batch_size_eval", type=int, default=8)
# need justification
parser.add_argument("--learning_rate", type=float, default=1e-4)
# need justification
parser.add_argument("--weight_decay", type=float, default=1e-4)
# need justification
parser.add_argument("--noisy_sample_size", type=int, default=32)
# contrastive learning temperature
parser.add_argument("--temp", type=float, default=0.1)
# neg sampling distance margin
parser.add_argument("--dist_margin", type=float, default=10.0)

parser.add_argument("--objective", type=str, default='lift-contrastive_cosine')
# reasoning machine
parser.add_argument("--reasoner", type=str, default='gnn', choices=['gnn', 'deepset', 'gradient'])
parser.add_argument("--tnorm", type=str, default='product', choices=['product', 'godel'])
# reasoner = gradient
parser.add_argument("--reasoning_rate", type=float, default=1e-1)
parser.add_argument("--reasoning_steps", type=int, default=1000)
parser.add_argument("--reasoning_optimizer", type=str, default='AdamW')
parser.add_argument("--reasoning_steps_eval", type=int, default=1000)
# reasoner = gnn
parser.add_argument("--num_layers", type=int, default=1)


def train_neural_binary_predicate(
        desc: str,
        train_dataloader: QueryAnsweringSeqDataLoader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner,
        optimizer: torch.optim.Optimizer,
        args):

    trajectory = defaultdict(list)

    fof_list = train_dataloader.get_fof_list()
    t = tqdm.tqdm(enumerate(fof_list), desc=desc, total=len(fof_list))

    # for each batch
    for ifof, fof in t:
        if fof.lstr != 'r1(s1,f)':
            continue
        ####################
        loss = 0
        metric_step = {}

        reasoner.initialize_with_formula(fof)
        reasoner.initialize_local_embedding()

        pos_1answer_list = []
        neg_answers_list = []

        for i, pos_answer_dict in enumerate(fof.easy_answer_list):
            # this iteration is somehow redundant since there is only one free
            # variable in current case, i.e., fname='f'
            assert 'f' in pos_answer_dict
            pos_1answer_list.append(random.choice(pos_answer_dict['f']))
            neg_answers_list.append(torch.randint(0, nbp.num_entities,
                                                  (args.noisy_sample_size, 1)))

        batch_pos_emb = nbp.get_entity_emb(pos_1answer_list)
        batch_neg_emb = nbp.get_entity_emb(
            torch.cat(neg_answers_list, dim=1))

        pos_tv = reasoner.evaluate_truth_values({'f': batch_pos_emb})
        pos_nll = - torch.log(pos_tv + 1e-10).mean()
        neg_tv = reasoner.evaluate_truth_values({'f': batch_neg_emb})
        neg_nll = - torch.log(1 - neg_tv + 1e-10).mean()
        metric_step['pos_tv'] = pos_tv.mean().item()
        metric_step['pos_nll'] = pos_nll.item()
        metric_step['neg_tv'] = neg_tv.mean().item()
        metric_step['neg_nll'] = neg_nll.item()
        loss += pos_nll + neg_nll

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ####################
        metric_step['loss'] = loss.item()

        postfix = {'step': ifof+1}
        for k in metric_step:
            postfix[k] = np.mean(metric_step[k])
            trajectory[k].append(postfix[k])
        postfix['acc_loss'] = np.mean(trajectory['loss'])
        t.set_postfix(postfix)

        metric_step['acc_loss'] = postfix['acc_loss']
        metric_step['lstr'] = fof.lstr

        logging.info(f"[train neural link predictor {desc}] {json.dumps(metric_step)}")

    t.close()

    metric = {}
    for k in trajectory:
        metric[k] = np.mean(trajectory[k])
    return metric


def train_lifted_estimator(
        desc: str,
        train_dataloader: QueryAnsweringSeqDataLoader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner,
        optimizer: torch.optim.Optimizer,
        args):

    T = args.temp
    trajectory = defaultdict(list)

    fof_list = train_dataloader.get_fof_list()
    t = tqdm.tqdm(enumerate(fof_list), desc=desc, total=len(fof_list))

    # for each batch
    for ifof, fof in t:
        ####################
        loss = 0
        metric_step = {}

        reasoner.initialize_with_formula(fof)

        # this procedure is somewhat of low efficiency
        # ? can we change it to batch implementation ?

        reasoner.estimate_lifted_embeddings()
        batch_fvar_emb = reasoner.get_embedding('f')

        if 'lift' in args.objective:
            lifted_tv = reasoner.evaluate_truth_values()
            lifted_nll = - torch.log(lifted_tv + 1e-10).mean()
            metric_step['lifted_tv'] = lifted_tv.mean().item()
            metric_step['lifted_tv_nll'] = lifted_nll.mean().item()
            loss += lifted_nll

        pos_1answer_list = []
        neg_answers_list = []

        for i, pos_answer_dict in enumerate(fof.easy_answer_list):
            # this iteration is somehow redundant since there is only one free
            # variable in current case, i.e., fname='f'
            assert 'f' in pos_answer_dict
            pos_1answer_list.append(random.choice(pos_answer_dict['f']))
            neg_answers_list.append(torch.randint(0, nbp.num_entities,
                                                  (args.noisy_sample_size, 1)))

        batch_pos_emb = nbp.get_entity_emb(pos_1answer_list)
        batch_neg_emb = nbp.get_entity_emb(
            torch.cat(neg_answers_list, dim=1))

        pos_tv = reasoner.evaluate_truth_values({'f': batch_pos_emb})
        pos_nll = - torch.log(pos_tv + 1e-10).mean()
        neg_tv = reasoner.evaluate_truth_values({'f': batch_neg_emb})
        neg_nll = - torch.log(1 - neg_tv + 1e-10).mean()
        metric_step['pos_tv'] = pos_tv.mean().item()
        metric_step['pos_tv_nll'] = pos_nll.item()
        metric_step['neg_tv'] = neg_tv.mean().item()
        metric_step['neg_tv_nll'] = neg_nll.item()
        loss += pos_nll + neg_nll

        if 'contrastive_cosine' in args.objective:
            contrastive_pos_score = torch.exp(torch.cosine_similarity(
                batch_pos_emb, batch_fvar_emb, dim=-1) / T)
            contrastive_neg_score = torch.exp(torch.cosine_similarity(
                batch_neg_emb, batch_fvar_emb, dim=-1) / T)

            contrastive_nll = - torch.log(
                contrastive_pos_score / (contrastive_pos_score + contrastive_neg_score.sum(0))
            ).mean()
            metric_step['contrastive_pos_score'] = contrastive_pos_score.mean().item()
            metric_step['contrastive_neg_score'] = contrastive_neg_score.mean().item()
            metric_step['contrastive_nll'] = contrastive_nll.item()
            loss += contrastive_nll

        if 'neg_sample_dist' in args.objective:
            pos_sample_score = torch.sigmoid(
                args.dist_margin - torch.sum((batch_pos_emb-batch_fvar_emb)**2, dim=-1) / T)
            neg_sample_score = torch.sigmoid(
                args.dist_margin - torch.sum((batch_neg_emb-batch_fvar_emb)**2, dim=-1) / T)

            neg_sample_nll = - torch.log(pos_sample_score + 1e-10).mean() \
                             - torch.log(1 - neg_sample_score + 1e-10).mean()

            metric_step['pos_sample_score'] = pos_sample_score.mean().item()
            metric_step['neg_sample_score'] = neg_sample_score.mean().item()
            metric_step['neg_sample_nll'] = neg_sample_nll.item()
            loss += neg_sample_nll

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ####################
        metric_step['loss'] = loss.item()

        postfix = {'step': ifof+1}
        for k in metric_step:
            postfix[k] = np.mean(metric_step[k])
            trajectory[k].append(postfix[k])
        postfix['acc_loss'] = np.mean(trajectory['loss'])
        t.set_postfix(postfix)

        metric_step['acc_loss'] = postfix['acc_loss']
        metric_step['lstr'] = fof.lstr

        logging.info(f"[train lifted estimator {desc}] {json.dumps(metric_step)}")

    t.close()

    metric = {}
    for k in trajectory:
        metric[k] = np.mean(trajectory[k])
    return metric


def compute_evaluation_scores(fof, batch_entity_rankings, metric):
    k = 'f'
    for i, ranking in enumerate(torch.split(batch_entity_rankings, 1)):
        ranking = ranking.squeeze()
        if fof.hard_answer_list[i]:
            # [1, num_entities]
            hard_answers = torch.tensor(fof.hard_answer_list[i][k],
                                        device=nbp.device)
            hard_answer_rank = ranking[hard_answers]

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

        else:
            pure_hard_ans_rank = ranking[
                torch.tensor(fof.easy_answer_list[i][k], device=nbp.device)]

        # remove better hard answers from its ranking
        _reference_hard_ans_rank = pure_hard_ans_rank.reshape(-1, 1)
        num_skipped_answers = torch.sum(
            pure_hard_ans_rank > _reference_hard_ans_rank, dim=0
        )
        pure_hard_ans_rank -= num_skipped_answers.reshape(
            pure_hard_ans_rank.shape)

        rr = (1 / (1+pure_hard_ans_rank)).detach().cpu().float().numpy()
        hit1 = (pure_hard_ans_rank < 1).detach().cpu().float().numpy()
        hit3 = (pure_hard_ans_rank < 3).detach().cpu().float().numpy()
        hit10 = (pure_hard_ans_rank < 10).detach().cpu().float().numpy()

        metric['mrr'].extend(rr.ravel().tolist())
        metric['hit1'].extend(hit1.ravel().tolist())
        metric['hit3'].extend(hit3.ravel().tolist())
        metric['hit10'].extend(hit10.ravel().tolist())


def evaluate_by_search_emb_then_rank_truth_value(
        e,
        desc,
        dataloader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner):
    """
    Evaluation used in CQD, two phase computation
    1. continuous optimiation of embeddings quant. + free
    2. evaluate all sentences with intermediate optimized
    """
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    fofs = dataloader.get_fof_list()

    # conduct reasoning
    with tqdm.tqdm(fofs, desc=desc) as t:
        for fof in t:
            reasoner.initialize_with_formula(fof)
            reasoner.estimate_lifted_embeddings()
            with torch.no_grad():
                truth_value_entity_batch = reasoner.evaluate_truth_values(
                    free_var_emb_dict={
                        'f': nbp.entity_embedding.unsqueeze(1)
                    },
                    batch_size_eval=args.batch_size_eval)  # [num_entities batch_size]
            ranking_score = torch.transpose(truth_value_entity_batch, 0, 1)
            ranked_entity_ids = torch.argsort(
                ranking_score, dim=-1, descending=True)
            batch_entity_rankings = torch.argsort(
                ranked_entity_ids, dim=-1, descending=False)
            compute_evaluation_scores(
                fof, batch_entity_rankings, metric[fof.lstr])

            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = float(
                        np.mean(metric[lstr][score_name]))

            postfix = {}
            # postfix['reasoning_steps'] = len(traj)
            postfix['lstr'] = fof.lstr
            for name in ['1p', '2p', '3p', '2i', 'inp']:
                if name in sum_metric:
                    postfix[name + '_hit3'] = sum_metric[name]['hit3']
            t.set_postfix(postfix)

    sum_metric['epoch'] = e
    logging.info(f"[{desc}][final] {json.dumps(sum_metric)}")
    torch.cuda.empty_cache()


def evaluate_by_nearest_search(
        e,
        desc,
        dataloader,
        nbp: NeuralBinaryPredicate,
        reasoner: GradientEFOReasoner):
    """
    Evaluation used by nearest neighbor
    1. continuous optimiation of embeddings quant. + free
    2. evaluate all sentences with intermediate optimized
    """
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    fofs = dataloader.get_fof_list()

    # conduct reasoning
    with tqdm.tqdm(fofs, desc=desc) as t:
        for fof in t:
            reasoner.initialize_with_formula(fof)
            reasoner.estimate_lifted_embeddings()
            batch_fvar_emb = reasoner.get_embedding('f')
            batch_entity_rankings = nbp.get_all_entity_rankings(
                batch_fvar_emb)
            # [batch_size, num_entities]
            compute_evaluation_scores(
                fof, batch_entity_rankings, metric[fof.lstr])
            t.set_postfix({'lstr': fof.lstr})

        print("sum metric")
        sum_metric = defaultdict(dict)
        for lstr in metric:
            for score_name in metric[lstr]:
                sum_metric[lstr2name[lstr]][score_name] = float(
                    np.mean(metric[lstr][score_name]))

        postfix = {}
        for name in ['1p', '2p', '3p', '2i', 'inp']:
            if name in sum_metric:
                postfix[name + '_hit3'] = sum_metric[name]['hit3']

    sum_metric['epoch'] = e
    logging.info(f"[{desc}][final] {json.dumps(sum_metric)}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # * parse argument
    args = parser.parse_args()

    # * prepare the logger
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=osp.join(args.output_dir, 'output.log'),
                        format='%(asctime)s %(message)s',
                        level=logging.INFO,
                        filemode='wt')

    # * initialize the kgindex
    kgidx = KGIndex.load(
        osp.join(args.task_folder, "kgindex.json"))

    # * load neural binary predicate
    print(f"loading the nbp {args.model_name}")
    nbp = get_nbp_class(args.model_name)(
        num_entities=kgidx.num_entities,
        num_relations=kgidx.num_relations,
        embedding_dim=args.embedding_dim,
        p=args.p,
        margin=args.margin,
        device=args.device)

    optimizer_nbp = getattr(torch.optim, args.optimizer)(
        nbp.parameters(),
        lr=1e-3,
        weight_decay=args.weight_decay)


    if args.checkpoint_path:
        nbp.load_state_dict(torch.load(args.checkpoint_path), strict=False)

    nbp.to(args.device)
    print(f"model loaded from {args.checkpoint_path}")

    # * initialize reasoning machine
    if args.tnorm == 'product':
        tnorm = ProductTNorm
    else:
        tnorm = GodelTNorm

    if args.reasoner == 'deepset':
        ent_dim = nbp.entity_embedding.size(1)
        rel_dim = nbp.relation_embedding.size(1)
        rds = RelationalDeepSet(
            ent_dim, rel_dim, num_layers=args.num_layers).to(nbp.device)
        reasoner = DeepsetEFOReasoner(nbp, tnorm, rds)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            # list(rds.parameters()) + list(nbp.parameters()),
            list(rds.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)

    elif args.reasoner == 'gnn':
        lgnn_layer = LogicalGNNLayerComplEx(nbp.embedding_dim,
                                            hidden_dim=4096,
                                            num_entities=nbp.num_entities,
                                            eps=0.1)
        lgnn_layer.to(nbp.device)
        reasoner = GNNEFOReasonerComplEx(nbp, tnorm, lgnn_layer)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            # list(lgnn_layer.parameters()) + list(nbp._entity_embedding.parameters()),
            list(lgnn_layer.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_estimator, 50, 0.1)

    elif args.reasoner == 'gradient':
        reasoner = GradientEFOReasoner(nbp, tnorm,
                                       reasoning_rate=args.reasoning_rate,
                                       reasoning_steps=args.reasoning_steps,
                                       reasoning_optimizer=args.reasoning_optimizer)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            nbp.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)

    # * prepare dataset``
    print("loading dataset")
    if args.train_queries:
        train_queries = [name2lstr[tq] for tq in args.train_queries]
    else:
        train_queries = list(name2lstr.values())
    print("train queries", train_queries)

    if args.eval_queries:
        eval_queries = [name2lstr[tq] for tq in args.eval_queries]
    else:
        eval_queries = list(name2lstr.values())
    print("eval queries", eval_queries)

    train_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'train-qaa.json'),
        # size_limit=args.batch_size * 1,
        target_lstr=train_queries,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0)

    valid_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'valid-qaa.json'),
        target_lstr=eval_queries,
        batch_size=5000,
        shuffle=False,
        num_workers=0)

    # valid_1p_dataloader = QueryAnsweringSeqDataLoader(
    #     osp.join(args.task_folder, 'valid-qaa.json'),
    #     target_lstr=['r1(s1,f)'],
    #     batch_size=5000,
    #     shuffle=False,
    #     num_workers=0)

    test_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'test-qaa.json'),
        target_lstr=eval_queries,
        batch_size=5000,
        shuffle=False,
        num_workers=0)

    # test_1p_dataloader = QueryAnsweringSeqDataLoader(
    #     osp.join(args.task_folder, 'test-qaa.json'),
    #     target_lstr=['r1(s1,f)'],
    #     batch_size=5000,
    #     shuffle=False,
    #     num_workers=0)
    print("dataset prepared")

    evaluate_by_search_emb_then_rank_truth_value(
        -1, f"CQD evaluate validate set",
        valid_dataloader, nbp,
        reasoner=GradientEFOReasoner(nbp, tnorm,
                                     reasoning_rate=args.reasoning_rate,
                                     reasoning_steps=args.reasoning_steps,
                                     reasoning_optimizer=args.reasoning_optimizer)
        )
    evaluate_by_search_emb_then_rank_truth_value(
        -1, f"CQD evaluate test set",
        valid_dataloader, nbp,
        reasoner=GradientEFOReasoner(nbp, tnorm,
                                     reasoning_rate=args.reasoning_rate,
                                     reasoning_steps=args.reasoning_steps,
                                     reasoning_optimizer=args.reasoning_optimizer)
        )

    for e in range(args.pretrain_epoch):
        train_neural_binary_predicate(f"epoch {e}",
                                      train_dataloader, nbp, reasoner, optimizer_nbp, args)
        evaluate_by_search_emb_then_rank_truth_value(
            e, f"CQD evaluate validate set pretrain {e+1}",
            valid_dataloader, nbp,
            reasoner=GradientEFOReasoner(nbp, tnorm,
                                         reasoning_rate=args.reasoning_rate,
                                         reasoning_steps=args.reasoning_steps,
                                         reasoning_optimizer=args.reasoning_optimizer))
        evaluate_by_search_emb_then_rank_truth_value(
            e, f"CQD evaluate test set pretrain {e+1}",
            test_dataloader, nbp,
            reasoner=GradientEFOReasoner(nbp, tnorm,
                                         reasoning_rate=args.reasoning_rate,
                                         reasoning_steps=args.reasoning_steps,
                                         reasoning_optimizer=args.reasoning_optimizer))

    for e in range(args.epoch):
        train_lifted_estimator(f"epoch {e}",
                               train_dataloader, nbp, reasoner, optimizer_estimator, args)
        scheduler.step()
        if (e+1) % 5 == 0:
            evaluate_by_search_emb_then_rank_truth_value(e, f"tv evaluate validate set epoch {e+1}",
                                                         valid_dataloader, nbp, reasoner)
            evaluate_by_search_emb_then_rank_truth_value(e, f"tv evaluate test set epoch {e+1}",
                                                         test_dataloader, nbp, reasoner)
            evaluate_by_nearest_search(e, f"NN evaluate validate set epoch {e+1}",
                                       valid_dataloader, nbp, reasoner)
            evaluate_by_nearest_search(e, f"NN evaluate test set epoch {e+1}",
                                       test_dataloader, nbp, reasoner)