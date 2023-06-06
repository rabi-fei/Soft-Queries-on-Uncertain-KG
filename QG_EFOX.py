import argparse
import math
import os.path as osp
from collections import defaultdict

import torch
import tqdm
import json
import pandas as pd
import numpy as np

from FIT import solve_EFO1
from src.utils.data import QueryAnsweringSeqDataLoader_v2
from src.utils.class_util import Writer
from src.structure.geometric_graph import QueryGraph
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from fol import BetaEstimator4V, BoxEstimator, LogicEstimator, NLKEstimator, ConEstimator, FuzzQEstiamtor


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/LogicE_FB15k-237_EFOX.yaml")


def read_from_yaml(yaml_path):
    import yaml
    with open(yaml_path, 'r') as fd:
        return yaml.load(fd, Loader=yaml.FullLoader)


def load_model(step, checkpoint_path, model, opt, load_device):
    full_ckpt_pth = osp.join(checkpoint_path, f'{step}.ckpt')
    print('Loading checkpoint %s...' % full_ckpt_pth)
    checkpoint = torch.load(full_ckpt_pth, map_location=load_device)
    model.load_state_dict(checkpoint['model_parameter'])
    opt.load_state_dict(checkpoint['optimizer_parameter'])
    current_learning_rate = checkpoint['learning_rate']
    warm_up_steps = checkpoint['warm_up_steps']
    return current_learning_rate, warm_up_steps


def load_beta_model(checkpoint_path, model, optimizer):
    print('Loading checkpoint %s...' % checkpoint_path)
    checkpoint = torch.load(osp.join(
        checkpoint_path, 'checkpoint'))
    init_step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'])
    current_learning_rate = checkpoint['current_learning_rate']
    warm_up_steps = checkpoint['warm_up_steps']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return current_learning_rate, warm_up_steps, init_step


def ranking2metrics(ranking, easy_ans, hard_ans, ranking_device):
    num_hard = len(hard_ans)
    num_easy = len(easy_ans)
    assert len(set(hard_ans).intersection(set(easy_ans))) == 0
    # only take those answers' rank
    cur_ranking = ranking[list(easy_ans) + list(hard_ans)]
    cur_ranking, indices = torch.sort(cur_ranking)
    masks = indices >= num_easy
    answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(ranking_device)
    cur_ranking = cur_ranking - answer_list + 1
    # filtered setting: +1 for start at 0, -answer_list for ignore other answers
    cur_ranking = cur_ranking[masks]
    # only take indices that belong to the hard answers
    mrr = torch.mean(1. / cur_ranking).item()
    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
    h10 = torch.mean(
        (cur_ranking <= 10).to(torch.float)).item()
    return mrr, h1, h3, h10


def eval_batch_query(model, pred_emb_list, easy_ans_list, hard_ans_list):
    """
    eval a batch of query of the same formula, the pred_emb of the query has been given.
    pred_emb:  batch*emb_dim
    easy_ans_list: list of easy_ans
    """
    device = model.device
    two_marginal_logs = defaultdict(float)
    one_marginal_logs, no_marginal_logs = defaultdict(float), defaultdict(float)
    f_str_list = [f'f{i + 1}' for i in range(len(pred_emb_list))]
    f_str = '_'.join(f_str_list)
    if len(pred_emb_list) == 1:
        with torch.no_grad():
            all_logit = model.compute_all_entity_logit(pred_emb_list[0], union=False)
            # batch*nentity
            argsort = torch.argsort(all_logit, dim=1, descending=True)
            ranking = argsort.clone().to(torch.float)
            #  create a new torch Tensor for batch_entity_range
            ranking = ranking.scatter_(1, argsort, torch.arange(model.n_entity).to(torch.float).
                                       repeat(argsort.shape[0], 1).to(device))
            # achieve the ranking of all entities
            for i in range(all_logit.shape[0]):
                easy_ans = [instance[0] for instance in easy_ans_list[i][f_str]]
                hard_ans = [instance[0] for instance in hard_ans_list[i][f_str]]
                mrr, h1, h3, h10 = ranking2metrics(ranking[i], easy_ans, hard_ans, device)
                two_marginal_logs['MRR'] += mrr
                two_marginal_logs['HITS1'] += h1
                two_marginal_logs['HITS3'] += h3
                two_marginal_logs['HITS10'] += h10
            num_query = all_logit.shape[0]
            two_marginal_logs['num_queries'] += num_query
    else:
        with torch.no_grad():
            final_ranking_list = []
            for pred_emb in pred_emb_list:
                all_logit = model.compute_all_entity_logit(pred_emb, union=False)
                argsort = torch.argsort(all_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                #  create a new torch Tensor for batch_entity_range
                ranking = ranking.scatter_(1, argsort, torch.arange(model.n_entity).to(torch.float).
                                           repeat(argsort.shape[0], 1).to(device))
                final_ranking_list.append(ranking)
            final_ranking = torch.stack(final_ranking_list, dim=1).to(device)  # batch * free_num * nentity
            two_marginal_logs, one_marginal_logs, no_marginal_logs = evaluate_batch_joint(final_ranking, easy_ans_list,
                                                                                          hard_ans_list, device, f_str)
    return two_marginal_logs, one_marginal_logs, no_marginal_logs


def log_add_metric(add_log, mrr, h1, h3, h10, mul_mrr, h1_1, h3_3, h10_10):
    add_log['MRR'] += mrr
    add_log['HITS1'] += h1
    add_log['HITS3'] += h3
    add_log['HITS10'] += h10
    add_log['num_queries'] += 1
    add_log['couple_MRR'] += mul_mrr
    add_log['HITS1*1'] += h1_1
    add_log['HITS3*3'] += h3_3
    add_log['HITS10*10'] += h10_10
    return add_log


def evaluate_batch_joint(final_ranking, easy_ans_list, hard_ans_list, device, f_str):
    """
    final_ranking: batch * free_num * nentity
    """
    two_marginal_logs = defaultdict(float)
    one_marginal_logs, no_marginal_logs = defaultdict(float), defaultdict(float)
    for i in range(final_ranking.shape[0]):
        easy_ans = easy_ans_list[i][f_str]  # A list of list, each list is an instance.
        hard_ans = hard_ans_list[i][f_str]
        num_easy, num_hard = len(easy_ans), len(hard_ans)
        #  assert len(set(hard_ans).intersection(set(easy_ans))) == 0
        full_ans = easy_ans + hard_ans
        full_ans_tensor = torch.tensor(full_ans).to(device).transpose(0, 1)
        marginal_easy_ans_list, marginal_hard_ans_list = [], []
        couple_filtered_list = []
        marginal_exist_num = 0
        marginal_stored = []
        for j in range(final_ranking.shape[1]):
            marginal_easy_ans, marginal_full_ans = set([easy_instance[j] for easy_instance in easy_ans]), \
                set([full_instance[j] for full_instance in full_ans])
            marginal_hard_ans = marginal_full_ans - marginal_easy_ans
            marginal_easy_ans_list.append(marginal_easy_ans)
            marginal_hard_ans_list.append(marginal_hard_ans)
            marginal_ans_ranking = final_ranking[i, j][list(marginal_easy_ans) + list(marginal_hard_ans)]
            marginal_num_hard, marginal_num_easy = len(marginal_hard_ans), len(marginal_easy_ans)
            sort_marginal_ranking, marginal_indices = torch.sort(marginal_ans_ranking)
            marginal_masks = marginal_indices >= marginal_num_easy
            marginal_answer_list = torch.arange(
                marginal_num_hard + marginal_num_easy).to(torch.float).to(device)
            filtered_marginal_ranking = sort_marginal_ranking - marginal_answer_list + 1
            # filtered setting: +1 for start at 0, -answer_list for ignore other answers
            adjusted_marginal_all_ranking = final_ranking[i, j].clone().to(device)
            adjusted_marginal_all_ranking[list(marginal_easy_ans) + list(marginal_hard_ans)] = \
                torch.gather(filtered_marginal_ranking, dim=0, index=marginal_indices.argsort())
            couple_filtered_list.append(adjusted_marginal_all_ranking)
            #  Compute the marginal ranking first
            if len(marginal_hard_ans) == 0:  # There is really possibility that no marginal hard answer
                pass
            else:
                marginal_exist_num += 1
                marginal_hard_ranking = filtered_marginal_ranking[marginal_masks]
                marginal_mrr = torch.mean(1. / marginal_hard_ranking).item()
                marginal_h1 = torch.mean((marginal_hard_ranking <= 1).to(torch.float)).item()
                marginal_h3 = torch.mean((marginal_hard_ranking <= 3).to(torch.float)).item()
                marginal_h10 = torch.mean(
                    (marginal_hard_ranking <= 10).to(torch.float)).item()
                marginal_stored.append([marginal_mrr, marginal_h1, marginal_h3, marginal_h10])
        marginal_filtered_joint_rank = torch.stack(couple_filtered_list, dim=0)  # free_num * nentity
        m_j_ranking = torch.gather(
            marginal_filtered_joint_rank, dim=1, index=torch.tensor(hard_ans).to(device).transpose(0, 1))
        #  free_num * hard_num

        couple_mrr = torch.mean(torch.sqrt(torch.prod((1. / m_j_ranking), dim=0))).item()
        couple_h1 = torch.mean(torch.prod((m_j_ranking <= 1).to(torch.float), dim=0)).item()
        couple_h3 = torch.mean(torch.prod((m_j_ranking <= 3).to(torch.float), dim=0)).item()
        couple_h10 = torch.mean(torch.prod((m_j_ranking <= 10).to(torch.float), dim=0)).item()

        #  Compute the hard joint ranking
        couple_ans_ranking = torch.gather(final_ranking[i], dim=1, index=full_ans_tensor)  # free_num * ans
        add_ans_ranking = torch.sum(couple_ans_ranking, dim=0)  # ans
        final_ans_ranking = add_ans_ranking * (add_ans_ranking + 1) / 2 + couple_ans_ranking[0]
        sort_ans_ranking, indices = torch.sort(final_ans_ranking)
        masks = indices >= num_easy
        answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(device)
        filtered_ans_ranking = sort_ans_ranking - answer_list + 1
        cur_ranking = filtered_ans_ranking[masks]
        # if math.isinf(mrr):
        # print("warning: mrr is inf")
        mrr = torch.mean(1. / cur_ranking).item()
        h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
        h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
        h10 = torch.mean(
            (cur_ranking <= 10).to(torch.float)).item()
        if marginal_exist_num == 0:
            no_marginal_logs = log_add_metric(
                no_marginal_logs, mrr, h1, h3, h10, couple_mrr, couple_h1, couple_h3, couple_h10)
        elif marginal_exist_num == 1:
            one_marginal_logs = log_add_metric(
                one_marginal_logs, mrr, h1, h3, h10, couple_mrr, couple_h1, couple_h3, couple_h10)
            one_marginal_logs['marginal_MRR'] += marginal_stored[0][0]
            one_marginal_logs['marginal_HITS1'] += marginal_stored[0][1]
            one_marginal_logs['marginal_HITS3'] += marginal_stored[0][2]
            one_marginal_logs['marginal_HITS10'] += marginal_stored[0][3]
        else:
            two_marginal_logs = log_add_metric(
                two_marginal_logs, mrr, h1, h3, h10, couple_mrr, couple_h1, couple_h3, couple_h10)
            # assert h10 <= couple_h10
            two_marginal_logs['marginal_MRR'] += marginal_stored[0][0] / 2
            two_marginal_logs['marginal_HITS1'] += marginal_stored[1][0] / 2
            two_marginal_logs['marginal_HITS1'] += marginal_stored[0][1] / 2
            two_marginal_logs['marginal_HITS3'] += marginal_stored[1][1] / 2
            two_marginal_logs['marginal_HITS3'] += marginal_stored[0][2] / 2
            two_marginal_logs['marginal_HITS10'] += marginal_stored[1][2] / 2
            two_marginal_logs['marginal_HITS10'] += marginal_stored[0][3] / 2
            two_marginal_logs['marginal_HITS10'] += marginal_stored[1][3] / 2
    return two_marginal_logs, one_marginal_logs, no_marginal_logs



if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    configure = read_from_yaml(args.config)
    if configure['cuda'] < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(configure['cuda']))
    all_formula_data = pd.read_csv(osp.join('data', 'DNF_EFO2_23_4123166.csv'))
    case_name = configure['output']['output_path'] if configure['output']['output_path'] else \
        args.config.split("config")[-1][1:]
    writer = Writer(case_name=case_name, config=configure, log_path=configure["output"]["prefix"])
    data_folder = configure['data']['data_folder']
    kgidx = KGIndex.load(osp.join(data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(data_folder, 'train_kg.tsv'),
        kgindex=kgidx)
    # get model
    train_config = configure['train']
    model_name = configure['estimator']['embedding']
    model_params = configure['estimator'][model_name]
    model_params['n_entity'], model_params['n_relation'] = train_kg.num_entities, train_kg.num_relations
    model_params['negative_sample_size'] = train_config['negative_sample_size']
    model_params['device'] = device

    if model_name == 'beta':
        model = BetaEstimator4V(**model_params)
        allowed_norm = ['DeMorgan', 'DNF+MultiIU']
    elif model_name == 'box':
        model = BoxEstimator(**model_params)
        allowed_norm = ['DNF+MultiIU']
    elif model_name == 'logic':
        model = LogicEstimator(**model_params)
        allowed_norm = ['DeMorgan+MultiI', 'DNF+MultiIU']
    elif model_name == 'NewLook':
        model = NLKEstimator(**model_params)
        model.setup_relation_tensor(train_kg.hr2t)
        allowed_norm = ['DNF+MultiIUD']
    elif model_name == 'ConE':
        model = ConEstimator(**model_params)
        allowed_norm = ['DeMorgan+MultiI', 'DNF+MultiIU']
    elif model_name == 'FuzzQE':
        model = FuzzQEstiamtor(**model_params)
    else:
        assert False, 'Not valid model name!'
    model.to(device)

    lr = train_config['learning_rate']
    if model.name == 'FuzzQE' and train_config['optimizer'] == 'AdamW':
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, list(model.parameters())),
            lr=lr, eps=1e-06, weight_decay=train_config['L2_reg'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_config['steps'], eta_min=0,
                                                               last_epoch=-1)
    else:
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = None
    init_step = 1
    loss_function = train_config['loss_function'] if 'loss_function' in train_config else 'original'

    if configure['load']['load_model']:
        checkpoint_path, checkpoint_step = configure['load']['checkpoint_path'], configure['load']['step']
        if checkpoint_step != 0:
            lr_dict, train_config['warm_up_steps'] = load_model(checkpoint_step, checkpoint_path, model, opt,
                                                                    device)
            lr = lr_dict
            init_step = checkpoint_step + 1  # I think there should be + 1 for train is before then save
        else:
            lr, train_config['warm_up_steps'], init_step = load_beta_model(checkpoint_path, model, opt)
            init_step += 1
    if 'train' not in configure['action']:
        assert train_config['steps'] == init_step

    if 'test' in configure['action']:
        all_metrics = defaultdict(dict)
        for i, row in tqdm.tqdm(all_formula_data.iterrows(), total=len(all_formula_data)):
            formula_id = row['formula_id']
            formula = row['formula']
            # data_path = osp.join(configure['data']['data_folder'], f'test_type{i:04d}_EFOX_qaa.json')
            data_path = osp.join(configure['data']['data_folder'], f'test_{formula_id}_EFOX_qaa.json')
            if not osp.exists(data_path):
                print(f'Warnings,{data_path} not exists!')
                continue
            test_dataloader = QueryAnsweringSeqDataLoader_v2(
                data_path,
                target_lstr=None,
                batch_size=configure['evaluate']['batch_size'],
                shuffle=False,
                num_workers=0)
            fof_list = test_dataloader.get_fof_list_no_shuffle()
            t = tqdm.tqdm(enumerate(fof_list), total=len(fof_list))
            # all_answers, now_formula_index = {}, {}
            # for lstr in test_dataloader.lstr_qaa:
            # all_answers[lstr] = torch.zeros((len(test_dataloader.lstr_qaa[lstr]), n_entity))
            # now_formula_index[lstr] = 0
            all_two_log, all_one_log, all_no_log = defaultdict(float), defaultdict(float), defaultdict(float)
            for ifof, fof in t:
                QG_instance = QueryGraph(fof.formula_list[0], device)
                QG_embedding_list = QG_instance.get_whole_graph_embedding(model=model)
                two_mar_log, mar_log, no_mar_logs = \
                    eval_batch_query(model, QG_embedding_list, fof.easy_answer_list, fof.hard_answer_list)
                for metric in two_mar_log:
                    all_two_log[metric] += two_mar_log[metric]
                for metric in mar_log:
                    all_one_log[metric] += mar_log[metric]
                for metric in no_mar_logs.keys():
                    all_no_log[metric] += no_mar_logs[metric]
            '''
            for log_metric in all_two_log.keys():
                if log_metric != 'num_queries':
                    all_two_log[log_metric] /= all_two_log['num_queries']
            for log_metric in all_one_log.keys():
                if log_metric != 'num_queries':
                    all_one_log[log_metric] /= all_one_log['num_queries']
            for log_metric in all_no_log.keys():
                if log_metric != 'num_queries':
                    all_no_log[log_metric] /= all_no_log['num_queries']
            '''
            print(all_two_log)
            all_metrics[formula] = {formula: [all_two_log, all_one_log, all_no_log]}
        #  writer.save_torch(all_answers, 'all_answer_tensor.ckpt')
            writer.save_pickle({formula: [all_two_log, all_one_log, all_no_log]}, f"all_logging_test_0_{formula_id}.pickle")
        writer.save_pickle(all_metrics, f"all_logging_test_0.pickle")




