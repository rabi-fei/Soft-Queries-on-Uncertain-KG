import argparse
import os.path as osp
from collections import defaultdict

import torch
import tqdm
import pandas as pd

from FIT import solve_conjunctive
from src.utils.data import QueryAnsweringSeqDataLoader_v2
from src.utils.class_util import Writer
from src.structure.knowledge_graph import KnowledgeGraph, kg_remove_node
from src.structure.knowledge_graph_index import KGIndex

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--sleep", type=int, default=0)
parser.add_argument("--ckpt", type=str, default='sparse/237/torch_0.005_0.001.ckpt')
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-EFO1')
parser.add_argument("--mode", type=str, default='test', choices=['valid', 'test'])
parser.add_argument("--e_norm", type=str, default='Godel', choices=['Godel', 'product'])
parser.add_argument("--c_norm", type=str, default='product', choices=['Godel', 'product'])
parser.add_argument("--max", type=int, default=1)
parser.add_argument("--data_type", type=str, default='EFOX', choices=['BetaE', 'EFO1', 'EFO1_l, EFOX'])
parser.add_argument("--formula", type=list, default=None)
negation_list = ['(r1(s1,f))&(!(r2(s2,f)))', '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))', '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))', '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))', '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))']


@torch.no_grad()
def solve_EFOX(conj_formula, relation_matrix, conjunctive_tnorm, existential_tnorm, index, device,
               max_enumeration):
    torch.cuda.empty_cache()
    with torch.no_grad():
        n_entity = relation_matrix[0].shape[0]
        all_candidates = {}
        for term_name in conj_formula.term_dict:
            if conj_formula.has_term_grounded_entity_id_list(term_name):
                all_candidates[term_name] = torch.zeros(n_entity).to(device)
                all_candidates[term_name][conj_formula.term_grounded_entity_id_dict[term_name][index]] = 1
            else:
                all_candidates[term_name] = torch.ones(n_entity).to(device)
        sub_graph_edge, sub_graph_negation_edge = [], []
        for pred in conj_formula.predicate_dict.values():
            pred_triples = (pred.head.name, conj_formula.pred_grounded_relation_id_dict[pred.name][index],
                            pred.tail.name)
            if pred.skolem_negation:
                sub_graph_negation_edge.append(pred_triples)
            else:
                sub_graph_edge.append(pred_triples)
        sub_kg_index = KGIndex()
        sub_kg_index.map_entity_name_to_id = {term: 0 for term in conj_formula.term_dict}
        sub_kg = KnowledgeGraph(sub_graph_edge, sub_kg_index)
        neg_kg = KnowledgeGraph(sub_graph_negation_edge, sub_kg_index)
        sub_kg_index.map_relation_name_to_id = {predicate: 0 for predicate in conj_formula.predicate_dict}
        sub_ans = solve_conjunctive(sub_kg, neg_kg, relation_matrix,
                                    all_candidates, conjunctive_tnorm, existential_tnorm, 'f', device,
                                    max_enumeration)


@torch.no_grad()
def eval_batch_query(model, pred_emb_list, easy_ans_list, hard_ans_list):
    """
    eval a batch of query of the same formula, the pred_emb of the query has been given.
    pred_emb:  batch*emb_dim
    easy_ans_list: list of easy_ans
    """
    device = model.device
    logs = defaultdict(float)
    marginal_logs = defaultdict(float)
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
                num_hard = len(hard_ans)
                num_easy = len(easy_ans)
                assert len(set(hard_ans).intersection(set(easy_ans))) == 0
                # only take those answers' rank
                cur_ranking = ranking[i, list(easy_ans) + list(hard_ans)]
                cur_ranking, indices = torch.sort(cur_ranking)
                masks = indices >= num_easy
                answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(device)
                cur_ranking = cur_ranking - answer_list + 1
                # filtered setting: +1 for start at 0, -answer_list for ignore other answers
                cur_ranking = cur_ranking[masks]
                # only take indices that belong to the hard answers
                mrr = torch.mean(1. / cur_ranking).item()
                h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                h10 = torch.mean(
                    (cur_ranking <= 10).to(torch.float)).item()
                #add_hard_list = torch.arange(num_hard).to(torch.float).to(device)
                #hard_ranking = cur_ranking + add_hard_list  # for all hard answer, consider other hard answer
                #logs['retrieval_accuracy'] += torch.mean((hard_ranking <= num_hard).to(torch.float)).item()
                logs['MRR'] += mrr
                logs['HITS1'] += h1
                logs['HITS3'] += h3
                logs['HITS10'] += h10
            num_query = all_logit.shape[0]
            logs['num_queries'] += num_query
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
            for i in range(all_logit.shape[0]):
                easy_ans = easy_ans_list[i][f_str]  # A list of list, each list is an instance.
                hard_ans = hard_ans_list[i][f_str]
                num_easy, num_hard = len(easy_ans), len(hard_ans)
                #  assert len(set(hard_ans).intersection(set(easy_ans))) == 0
                full_ans = easy_ans + hard_ans
                full_ans_tensor = torch.tensor(full_ans).to(device).transpose(0, 1)
                marginal_easy_ans_list, marginal_hard_ans_list = [], []
                for j in range(len(pred_emb_list)):
                    marginal_easy_ans, marginal_full_ans = set([easy_instance[j] for easy_instance in easy_ans]), \
                        set([full_instance[j] for full_instance in full_ans])
                    marginal_hard_ans = marginal_full_ans - marginal_easy_ans
                    marginal_easy_ans_list.append(marginal_easy_ans)
                    marginal_hard_ans_list.append(marginal_hard_ans)
                    #  Compute the marginal ranking first
                    if len(marginal_hard_ans) == 0:  # There is really possibility that no marginal hard answer
                        marginal_logs['num_queries'] -= 1
                    else:
                        marginal_metric_metrics = ranking2metrics(final_ranking_list[j][i], marginal_easy_ans,
                                                                  marginal_hard_ans)
                        mrr, h1, h3, h10 = marginal_metric_metrics
                        marginal_logs['MRR'] += mrr / len(pred_emb_list)
                        marginal_logs['HITS1'] += h1 / len(pred_emb_list)
                        marginal_logs['HITS3'] += h3 / len(pred_emb_list)
                        marginal_logs['HITS10'] += h10 / len(pred_emb_list)
                #  Compute the hard joint ranking
                couple_ans_ranking = torch.gather(final_ranking[i], dim=1, index=full_ans_tensor)  # free_num * ans
                add_ans_ranking = torch.sum(couple_ans_ranking, dim=0)  # ans
                final_ans_ranking = add_ans_ranking * (add_ans_ranking + 1) / 2 + couple_ans_ranking[0]
                sort_ans_ranking, indices = torch.sort(final_ans_ranking)
                masks = indices >= num_easy
                answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(device)
                filtered_ans_ranking = sort_ans_ranking - answer_list + 1
                cur_ranking = filtered_ans_ranking[masks]
                #if math.isinf(mrr):
                    #print("warning: mrr is inf")
                mrr = torch.mean(1. / cur_ranking).item()
                h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                h10 = torch.mean(
                    (cur_ranking <= 10).to(torch.float)).item()
                logs['MRR'] += mrr
                #if math.isinf(logs['MRR']):
                    #print("warning: mrr is inf")
                logs['HITS1'] += h1
                logs['HITS3'] += h3
                logs['HITS10'] += h10
            num_query = all_logit.shape[0]
            logs['num_queries'] += num_query
            marginal_logs['num_queries'] += num_query
    return marginal_logs, logs


def compute_single_evaluation(fof, batch_ans_tensor, n_entity):
    metrics = defaultdict(float)
    argsort = torch.argsort(batch_ans_tensor, dim=1, descending=True)
    ranking = argsort.clone().to(torch.float).to(cuda_device)
    ranking = ranking.scatter_(1, argsort, torch.arange(n_entity).to(torch.float).
                               repeat(argsort.shape[0], 1).to(cuda_device))
    logs = defaultdict(float)
    marginal_logs = defaultdict(float)
    f_str_list = [f'f{i + 1}' for i in range(len(pred_emb_list))]
    f_str = '_'.join(f_str_list)
    for i in range(batch_ans_tensor.shape[0]):
        #ranking = ranking.scatter_(0, argsort, torch.arange(n_entity).to(torch.float))
        hard_ans = fof.hard_answer_list[i][k]
        easy_ans = fof.easy_answer_list[i][k]
        num_hard = len(hard_ans)
        num_easy = len(easy_ans)
        real_ans_num = num_easy + num_hard
        pred_ans_num = torch.sum(batch_ans_tensor[i])
        cur_ranking = ranking[i, list(easy_ans) + list(hard_ans)]
        cur_ranking, indices = torch.sort(cur_ranking)
        masks = indices >= num_easy
        # easy_masks = indices < num_easy
        answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(cuda_device)
        cur_ranking = cur_ranking - answer_list + 1
        # filtered setting: +1 for start at 0, -answer_list for ignore other answers
        # easy_ranking = cur_ranking[easy_masks]
        hard_ranking = cur_ranking[masks]
        # only take indices that belong to the hard answers
        '''
        if easy_ans:
            easy_mrr = torch.mean(1. / easy_ranking).item()
            metrics['easy_queries'] += 1
        else:
            easy_mrr = 0
        metrics['easy_MRR'] += easy_mrr
        '''
        mrr = torch.mean(1. / hard_ranking).item()
        h1 = torch.mean((hard_ranking <= 1).to(torch.float)).item()
        h3 = torch.mean((hard_ranking <= 3).to(torch.float)).item()
        h10 = torch.mean(
            (hard_ranking <= 10).to(torch.float)).item()
        mae = torch.abs(pred_ans_num - real_ans_num).item()
        mape = mae / real_ans_num
        metrics['MAE'] += mae
        metrics['MAPE'] += mape
        metrics['MRR'] += mrr
        metrics['HITS1'] += h1
        metrics['HITS3'] += h3
        metrics['HITS10'] += h10
    metrics['num_queries'] += batch_ans_tensor.shape[0]
    return metrics


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    relation_matrix_list = torch.load(args.ckpt)
    n_relation, n_entity = len(relation_matrix_list), relation_matrix_list[0].shape[0]
    if args.cuda < 0:
        cuda_device = torch.device('cpu')
    else:
        cuda_device = torch.device('cuda:{}'.format(args.cuda))
    for i in range(len(relation_matrix_list)):
        relation_matrix_list[i] = relation_matrix_list[i].to(cuda_device)
    all_metrics = defaultdict(dict)
    all_formula_data = pd.read_csv(osp.join('data', 'DNF_EFO2_23_412316.csv'))
    for i, row in tqdm.tqdm(all_formula_data.iterrows(), total=len(all_formula_data)):
        formula_id = row['formula_id']
        formula = row['formula']
        # data_path = osp.join(configure['data']['data_folder'], f'test_type{i:04d}_EFOX_qaa.json')
        formula_path = osp.join(args.data_folder, f'test_{formula_id}_EFOX_qaa.json')
        if not osp.exists(formula_path):
            print(f'Warnings,{formula_path} not exists!')
            continue
        test_dataloader = QueryAnsweringSeqDataLoader_v2(
            formula_path,
            # size_limit=args.batch_size * 1,
            target_lstr=args.formula,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0)
        fof_list = test_dataloader.get_fof_list_no_shuffle()
        t = tqdm.tqdm(enumerate(fof_list), total=len(fof_list))
        all_metrics = defaultdict(dict)
        for ifof, fof in t:
            torch.cuda.empty_cache()
            batch_ans_list, metric = [], {}
            for query_index in range(len(fof.easy_answer_list)):
                ans = solve_EFOX(fof, relation_matrix_list, args.c_norm, args.e_norm, query_index, cuda_device,
                                 args.max)
                batch_ans_list.append(ans)
            batch_ans_tensor = torch.stack(batch_ans_list, dim=0)
            batch_score = compute_single_evaluation(fof, batch_ans_tensor, n_entity)
            for metric in batch_score:
                if metric not in all_metrics[fof.lstr]:
                    all_metrics[fof.lstr][metric] = 0
                all_metrics[fof.lstr][metric] += batch_score[metric]
            del batch_score, batch_ans_tensor
            for metric in log:
                all_log[metric] += log[metric]
            for metric in mar_log:
                all_log[f'marginal_{metric}'] += mar_log[metric]
        for log_metric in all_log.keys():
            if log_metric != 'num_queries':
                all_log[log_metric] /= all_log['num_queries']
        print(all_log)
        all_metrics[formula] = all_log
        #  writer.save_torch(all_answers, 'all_answer_tensor.ckpt')\
    writer = Writer(case_name=args.ckpt, config=args, log_path='results')

    # all_answers, now_formula_index = {}, {}
    # for lstr in test_dataloader.lstr_qaa:
        # all_answers[lstr] = torch.zeros((len(test_dataloader.lstr_qaa[lstr]), n_entity))
        # now_formula_index[lstr] = 0
    for ifof, fof in t:
        torch.cuda.empty_cache()
        batch_ans_list, metric = [], {}
        for query_index in range(len(fof.easy_answer_list)):
            ans = solve_EFO1(fof, relation_matrix_list, args.c_norm, args.e_norm, query_index, cuda_device, args.max)
            batch_ans_list.append(ans)
        batch_ans_tensor = torch.stack(batch_ans_list, dim=0)
        #all_answers[fof.lstr][now_formula_index[fof.lstr]: now_formula_index[fof.lstr] + batch_ans_tensor.shape[0], :] \
            #= batch_ans_tensor
        #now_formula_index[fof.lstr] += batch_ans_tensor.shape[0]
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
    #writer.save_torch(all_answers, 'all_answer_tensor.ckpt')
    writer.save_pickle(all_metrics, f"all_logging_{args.mode}_0.pickle")
