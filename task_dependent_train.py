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
from src.structure.neural_binary_predicate import ComplEx, NeuralBinaryPredicate, TransE
from src.utils.data import (QueryAnsweringSeqDataLoader, QueryAnsweringMixDataLoader,
                            TrainRandomSentencePairDataLoader,
                            RaggedBatch)

lstr2name = {'r1(s1,f)': '1p', '(r1(s1,e1))&(r2(e1,f))': '2p', '((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f))': '3p', '(r1(s1,f))&(r2(s2,f))': '2i', '((r1(s1,f))&(r2(s2,f)))&(r3(s3,f))': '3i', '((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f))': 'ip', '((r1(s1,e1))&(r2(e1,f)))&(r3(s2,f))': 'pi', '(r1(s1,f))&(!(r2(s2,f)))': '2in', '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))': '3in', '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))': 'inp', '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))': 'pin', '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))': 'pni', '(r1(s1,f))|(r2(s2,f))': '2u', '((r1(s1,e1))|(r2(s2,e1))))&(r3(e1,f))': 'up', '!((!(r1(s1,f)))&(!(r2(s2,f))))': '2u-dnf', '!(((!(r1(s1,e1)))|(r2(s2,e1)))&(r3(e1,f)))': 'up-dnf'}

parser = argparse.ArgumentParser()

# base environment
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--output_dir", type=str, default='log')

# input task folder, defines knowledge graph, index, and formulas
parser.add_argument("--task_folder", type=str, default='data/FB15k-237-q2b')

# model, defines the neural binary predicate
parser.add_argument("--model_name", type=str, default='complex')
parser.add_argument("--embedding_dim", type=int, default=500)
parser.add_argument("--margin", type=float, default=20)
parser.add_argument("--p", type=int, default=1)
parser.add_argument("--checkpoint_path")

# optimization
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--reasoning_rate", type=float, default=1e-1)
parser.add_argument("--objective", type=str, choices=['kvsall', 'noisy', 'none'], default='none')
parser.add_argument("--noisy_sample_size", type=int, default=32)

parser.add_argument("--metric_margin", type=float, default=50)
parser.add_argument("--sigma", type=float, default=10)
parser.add_argument("--v", type=float, default=.9)

def train_epoch_noisy_v2(desc, train_dataloader: TrainRandomSentencePairDataLoader, nbp: NeuralBinaryPredicate, grm: GradientReasoningMachine, args):
    optimizer = torch.optim.Adam(nbp.parameters(), args.learning_rate)

    with tqdm.tqdm(enumerate(train_dataloader), desc=desc, total=len(train_dataloader)) as t:
        trajectory = defaultdict(list)
        for i, (pos_fof, neg_fof) in t:
            metric_step = defaultdict(list)
            ####################
            optimizer.zero_grad()
            pos_fetched, neg_fetched = grm.reasoning((pos_fof, neg_fof))
            plogtv = - torch.log(pos_fetched['tv'] + 1e-20)

            # plogtv_list = torch.split(plogtv, pos_answer_sizes)
            ragged_plogtv = RaggedBatch(
                flatten=plogtv,
                sizes=[len(gdict['f']) for gdict in pos_fof.grounding_dict_list])
            padded_plogtv = ragged_plogtv.to_dense_matrix(padding_value=0)
            pos_losses = torch.sum(padded_plogtv, dim=-1) / torch.tensor(ragged_plogtv.sizes, device=nbp.device)

            nlog1mtv = - torch.log(1 - neg_fetched['tv'] + 1e-20)
            ragged_nlog1mtv = RaggedBatch(
                flatten=nlog1mtv,
                sizes=[len(gdict['f']) for gdict in neg_fof.grounding_dict_list])
            padded_nlog1mtv = ragged_nlog1mtv.to_dense_matrix(padding_value=0)
            neg_losses = torch.sum(padded_nlog1mtv, dim=-1) / torch.tensor(ragged_nlog1mtv.sizes, device=nbp.device)

            loss = pos_losses.mean() + neg_losses.mean()
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

def train_lower_bound_noisy_margin(
    desc,
    train_dataloader: QueryAnsweringSeqDataLoader,
    nbp:NeuralBinaryPredicate,
    grm: GradientReasoningMachine,
    args):
    optimizer = torch.optim.Adam(nbp.parameters(), args.learning_rate)

    metric_margin = torch.tensor(args.metric_margin, device=nbp.device)
    v = torch.tensor(args.v, device=nbp.device)

    with tqdm.tqdm(enumerate(train_dataloader), desc=desc, total=len(train_dataloader)) as t:
        trajectory = defaultdict(list)
        for ii, fofs in t:
            metric_step = defaultdict(list)
            ####################
            optimizer.zero_grad()
            fetched = grm.reasoning(fofs, infer_free=True)
            loss = 0

            pos_tv_list = []
            neg_tv_list = []
            mle_loss_list = []
            embedding_regularization_list = []

            # for each formula
            for fof, fetch in zip(fofs, fetched):
                batch_fvar_local_emb_dict = fetch['fvar_local_emb_dict']
                batch_tv = fetch['tv']

                for i, pos_answer_dict in enumerate(fof.easy_answer_list):
                    tv = batch_tv[i]
                    pos_tv = tv
                    neg_tv = tv
                    for f in pos_answer_dict:
                        fvar_emb = batch_fvar_local_emb_dict[f][i]
                        pos_answer = pos_answer_dict[f]

                        pos_embs = nbp.get_head_emb(pos_answer)
                        neg_embs = nbp.get_head_emb(torch.randint(0, nbp.num_entities, (args.noisy_sample_size,)))

                        pos_ans_tv = torch.exp(- torch.sum((pos_embs - fvar_emb)**2, dim=-1) / args.sigma ** 2)
                        pos_tv = grm.tnorm.conjunction(pos_ans_tv, pos_tv)
                        pos_tv_list.append(pos_tv.mean().item())

                        neg_ans_tv = torch.exp(- torch.sum((neg_embs - fvar_emb)**2, dim=-1) / args.sigma ** 2)
                        neg_tv = grm.tnorm.conjunction(neg_ans_tv, neg_tv)
                        neg_tv_list.append(neg_tv.mean().item())

                        mle = - torch.log(pos_tv).mean() - torch.log(1-neg_tv).mean()
                        mle_loss_list.append(mle)


                        embedding_reg = 0
                        for symb in fof.symbol_dict:
                            symb_emb = nbp.get_head_emb(
                                fof.get_term_grounded_entity_id_list(symb)[i]
                            )
                            embedding_reg += torch.sum(nbp.regularization(symb_emb) ** 3, -1)

                        for pred in fof.predicate_dict:
                            pred_emb = nbp.get_head_emb(
                                fof.get_pred_grounded_relation_id_list(pred)[i]
                            )
                            embedding_reg += torch.sum(nbp.regularization(pred_emb) ** 3, -1)

                        embedding_regularization_list.append(embedding_reg)


            mle_loss = torch.mean(torch.stack(mle_loss_list))
            embedding_regularization = torch.mean(torch.stack(embedding_regularization_list)) * .5

            loss = mle_loss + embedding_regularization
            loss.backward()
            optimizer.step()

            ####################
            metric_step['loss'].append(loss.item())
            metric_step['pos_tv'].append(np.mean(pos_tv_list))
            metric_step['neg_tv'].append(np.mean(neg_tv_list))
            metric_step['mle_loss'].append(mle_loss.item())
            metric_step['emb_reg'].append(embedding_regularization.item())

            postfix = {'step': ii+1}
            for k in metric_step:
                postfix[k] = np.mean(metric_step[k])
                trajectory[k].append(postfix[k])
            logging.info(f"[{desc}] {postfix}")
            postfix['acc_loss'] = np.mean(trajectory['loss'])
            t.set_postfix(postfix)

        metric = {'step': ii+1}
        for k in trajectory:
            metric[k] = np.mean(trajectory[k])
    return metric

# def train_epoch_noisy(desc, train_dataloader, nbp: NeuralBinaryPredicate, grm: GradientReasoningMachine, args):
#     optimizer = torch.optim.Adam(nbp.parameters(), args.learning_rate)

#     with tqdm.tqdm(enumerate(train_dataloader), desc=desc, total=len(train_dataloader)) as t:
#         trajectory = defaultdict(list)
#         for i, fofs in t:
#             ####################
#             optimizer.zero_grad()
#             fetched = grm.reasoning(fofs, all_candidates=True)

#             metric_step = defaultdict(list)
#             loss = 0

#             for fof, fof_reasoning_kv in zip(fofs, fetched):
#                 # batch_size, all entities
#                 batch_truth_value_of_grounded_cases = torch.transpose(
#                     fof_reasoning_kv['tv'], dim0=0, dim1=1)
#                 batch_size, answer_size = batch_truth_value_of_grounded_cases.shape

#                 # only works for single free variable with name
#                 true_answers = []
#                 tv4target = 0
#                 tv4noisy = 0
#                 for j, easy_answer in enumerate(fof.easy_answer_list):
#                     target_index = torch.tensor(easy_answer['f'],
#                                                 device=args.device)
#                     target_one_hot = torch.sum(
#                         F.one_hot(target_index, num_classes=answer_size),
#                         dim=0,
#                         keepdim=True
#                     )
#                     true_answers.append(target_one_hot)

#                     noisy_index = torch.randint(low=0,
#                                                 high=nbp.num_entities,
#                                                 size=(128,),
#                                                 device=args.device)
#                     tv4target -= torch.mean(torch.log(
#                         batch_truth_value_of_grounded_cases[j, target_index] + 1e-10))
#                     tv4noisy -= torch.mean(torch.log( 1 -
#                         batch_truth_value_of_grounded_cases[j, noisy_index] + 1e-10))

#                     this_loss = tv4target + tv4noisy

#                 multi_true_answer_tensor = torch.cat(true_answers, dim=0)

#                 loss += this_loss / len(fof.easy_answer_list)
#                 metric_step['loss'].append(this_loss.item())

#                 true_positive = torch.sum((multi_true_answer_tensor * batch_truth_value_of_grounded_cases) > 0.5, -1).tolist()
#                 all_true = torch.sum(multi_true_answer_tensor, -1).tolist()
#                 all_positive = torch.sum(batch_truth_value_of_grounded_cases > 0.5, -1).tolist()
#                 precision, recall = [], []
#                 for tp, at, ap in zip(true_positive, all_true, all_positive):
#                     precision.append(tp / ap if ap > 0 else 0)
#                     recall.append(tp / at)
#                 metric_step['precision'].extend(precision)
#                 metric_step['recall'].extend(recall)
#                 metric_step['all_possitive'].extend(all_positive)

#             loss.backward()
#             optimizer.step()
#             ####################

#             postfix = {'step': i+1}
#             for k in metric_step:
#                 postfix[k] = np.mean(metric_step[k])
#                 trajectory[k].append(postfix[k])
#             logging.info(f"[{desc}] {postfix}")
#             postfix['acc_loss'] = np.mean(trajectory['loss'])
#             t.set_postfix(postfix)

#         metric = {'step': i+1}
#         for k in trajectory:
#             metric[k] = np.mean(trajectory[k])
#     return metric


# # The K-verses-all objective, shown to be suboptimal
# def train_epoch_K_verses_All(desc, train_dataloader, nbp: NeuralBinaryPredicate, grm: GradientReasoningMachine, args):
#     optimizer = torch.optim.Adam(nbp.parameters(), args.learning_rate)

#     with tqdm.tqdm(enumerate(train_dataloader), desc=desc, total=len(train_dataloader)) as t:
#         trajectory = defaultdict(list)
#         for i, fofs in t:
#             ####################
#             optimizer.zero_grad()
#             fetched = grm.reasoning(fofs, all_candidates=True)

#             metric_step = defaultdict(list)
#             loss = 0

#             for fof, fof_reasoning_kv in zip(fofs, fetched):
#                 # batch_size, all entities
#                 batch_truth_value_of_grounded_cases = torch.transpose(
#                     fof_reasoning_kv['tv'], dim0=0, dim1=1)
#                 batch_size, answer_size = batch_truth_value_of_grounded_cases.shape
#                 # only works for single free variable with name
#                 true_answers = []
#                 for easy_answer in fof.easy_answer_list:
#                     target_sparse = torch.tensor(easy_answer['f'], device=args.device)
#                     target_one_hot = torch.sum(
#                         F.one_hot(target_sparse, num_classes=answer_size),
#                         dim=0,
#                         keepdim=True
#                     )
#                     true_answers.append(target_one_hot)

#                 multi_true_answer_tensor = torch.cat(true_answers, dim=0)
#                 this_loss = - multi_true_answer_tensor * torch.log(batch_truth_value_of_grounded_cases + 1e-10)
#                 this_loss -= (1-multi_true_answer_tensor) * torch.log(1-batch_truth_value_of_grounded_cases + 1e-10)
#                 loss += this_loss.mean()

#                 metric_step['loss'].append(this_loss.mean().item())

#                 true_positive = torch.sum((multi_true_answer_tensor * batch_truth_value_of_grounded_cases) > 0.5, -1).tolist()
#                 all_true = torch.sum(multi_true_answer_tensor, -1).tolist()
#                 all_positive = torch.sum(batch_truth_value_of_grounded_cases > 0.5, -1).tolist()
#                 precision, recall = [], []
#                 for tp, at, ap in zip(true_positive, all_true, all_positive):
#                     precision.append(tp / ap if ap > 0 else 0)
#                     recall.append(tp / at)

#                 metric_step['precision'].extend(precision)
#                 metric_step['recall'].extend(recall)
#                 metric_step['all_possitive'].extend(all_positive)

#             loss.backward()
#             optimizer.step()
#             ####################


#             postfix = {'step': i+1}
#             for k in metric_step:
#                 postfix[k] = np.mean(metric_step[k])
#                 trajectory[k].append(postfix[k])
#             logging.info(f"[{desc}] {postfix}")
#             postfix['acc_loss'] = np.mean(trajectory['loss'])
#             t.set_postfix(postfix)

#         metric = {'step': i+1}
#         for k in trajectory:
#             metric[k] = np.mean(trajectory[k])
#     return metric


def compute_evaluation_scores(fof, batch_entity_rankings, metric):
    k = 'f'
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
        hit10 =  (pure_hard_ans_rank < 10).detach().cpu().numpy()

        metric['mrr'].append(rr.mean())
        metric['hit1'].append(hit1.mean())
        metric['hit3'].append(hit3.mean())
        metric['hit10'].append(hit10.mean())


def evaluate_by_search_emb_then_rank_truth_value(
    desc, dataloader, nbp:NeuralBinaryPredicate, grm: GradientReasoningMachine, target_lstr=[]):
    """
    Evaluation used in CQD, two phase computation
    1. continuous optimiation of embeddings quant. + free
    2. evaluate all sentences with intermediate optimized
    """
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    with tqdm.tqdm(dataloader, desc=desc, total=len(dataloader)) as t:
        for i, _fofs in enumerate(t):
            # filter the desired lstr
            fofs = []
            for f in _fofs:
                if (((len(target_lstr) == 0) or
                     (f.lstr() in target_lstr))
                    and len(f.free_variable_dict) == 1):
                    fofs.append(f)

            # conduct reasoning
            fetched = grm.reasoning(fofs, infer_free=False)
            for fof, fof_reasoning_kv in zip(fofs, fetched):
                truth_value_entity_batch = fof_reasoning_kv['tv']  # [num_entities batch_size]
                ranking_score = torch.transpose(truth_value_entity_batch, 0, 1)
                # batch_entity_rankings = nbp.get_all_entity_rankings(batch_est_emb)
                # ranked_entity_ids[ranking] = {entity_id} at the {rankings}-th place
                ranked_entity_ids = torch.argsort(ranking_score, dim=-1, descending=True)
                # entity_rankings[entity_id] = {rankings} of the entity
                batch_entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
                # [batch_size, num_entities]
                compute_evaluation_scores(fof, batch_entity_rankings, metric[fof.lstr()])


            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = np.mean(metric[lstr][score_name])

            postfix = {}
            for name in ['1p', '2p', '3p', '2i', 'inp']:
                if name in sum_metric:
                    postfix[name + '_hit3'] = sum_metric[name]['hit3']
            t.set_postfix(postfix)

    logging.info(f"[{desc}][final] {sum_metric}")

def evaluate_by_nearest_search(
    desc, dataloader, nbp:NeuralBinaryPredicate, grm: GradientReasoningMachine, target_lstr=[]):
    """
    Evaluation used in CQD, two phase computation
    1. continuous optimiation of embeddings quant. + free
    2. evaluate all sentences with intermediate optimized
    """
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    with tqdm.tqdm(dataloader, desc=desc, total=len(dataloader)) as t:
        for i, _fofs in enumerate(t):
            # filter the desired lstr
            fofs = []
            for f in _fofs:
                if (((len(target_lstr) == 0) or
                     (f.lstr() in target_lstr))
                    and len(f.free_variable_dict) == 1):
                    fofs.append(f)

            # conduct reasoning
            fetched = grm.reasoning(fofs, infer_free=False)
            for fof, fof_reasoning_kv in zip(fofs, fetched):
                fvar_emb_dict = fof_reasoning_kv['fvar_local_emb_dict']  # [num_entities batch_size]
                batch_entity_rankings = nbp.get_all_entity_rankings(fvar_emb_dict['f'])
                # [batch_size, num_entities]
                compute_evaluation_scores(fof, batch_entity_rankings, metric[fof.lstr()])


            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = np.mean(metric[lstr][score_name])

            postfix = {}
            for name in ['1p', '2p', '3p', '2i', 'inp']:
                if name in sum_metric:
                    postfix[name + '_hit3'] = sum_metric[name]['hit3']
            t.set_postfix(postfix)

    logging.info(f"[{desc}][final] {sum_metric}")


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

    if args.model_name.lower() == 'transe':
        nbp_class = TransE
    elif args.model_name.lower() == 'complex':
        nbp_class = ComplEx
    else:
        raise NotImplementedError

    nbp = nbp_class(
        num_entities=kgidx.num_entities,
        num_relations=kgidx.num_relations,
        embedding_dim=args.embedding_dim,
        p=args.p,
        margin=args.margin,
        device=args.device)

    if args.checkpoint_path:
        nbp.load_state_dict(torch.load(args.checkpoint_path))
        print(f"model loaded from {args.checkpoint_path}")

    nbp.to(args.device)

    # this dataloader is for k-vs-all objective. works for noisy v1
    # depreciated
    train_dataloader = QueryAnsweringMixDataLoader(
        osp.join(args.task_folder, 'train-qaa.json'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0)

    # for noisy objective v2
    # train_dataloader = TrainNoisyAnswerDataLoader(
    #     osp.join(args.task_folder, 'train-qaa.json'),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     answer_size=kgidx.num_entities,
    #     noisy_sample_size=args.noisy_sample_size,
    #     num_workers=2)

    valid_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'valid-qaa.json'),
        batch_size=5000,
        shuffle=False,
        num_workers=1
    )

    test_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'test-qaa.json'),
        batch_size=5000,
        shuffle=False,
        num_workers=1
    )


    # train_epoch_K_verses_All(f"initial from cold start",
                            #    train_dataloader, nbp, grm0, args)
    train_grm = GradientReasoningMachine(
        reasoning_rate=args.reasoning_rate,
        reasoning_steps=3,
        reasoning_optimizer='Adam',
        nbp=nbp,
        tnorm=ProductTNorm)

    eval_grm = GradientReasoningMachine(
        reasoning_rate=args.reasoning_rate,
        reasoning_steps=1000,
        reasoning_optimizer='Adam',
        nbp=nbp,
        tnorm=ProductTNorm)

    # evaluate_by_nearest_search(f"validate epoch {0}",
    #                            valid_dataloader, nbp, eval_grm)
    # evaluate_by_nearest_search(f"test epoch {0}",
    #                            test_dataloader, nbp, eval_grm)

    eval_only = False
    for e in range(args.epoch):

        if args.objective.lower() == 'noisy':
            # # train_epoch_noisy_v2(f"training epoch {e}",
            #                      train_dataloader, nbp, train_grm, args)
            train_lower_bound_noisy_margin(f"train lower bound noisy", train_dataloader, nbp, train_grm, args)
        else:
            print("no training")
            eval_only = True

        if (e+1) % 1 == 0:
            evaluate_by_nearest_search(f"validate epoch {e}",
                     valid_dataloader, nbp, eval_grm)
            evaluate_by_nearest_search(f"test epoch {e}",
                     test_dataloader, nbp, eval_grm)

            if eval_only:
                break
