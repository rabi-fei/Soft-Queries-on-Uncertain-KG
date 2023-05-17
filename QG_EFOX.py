import argparse
import os.path as osp
from collections import defaultdict

import torch
import tqdm

from FIT import solve_EFO1
from src.utils.data import QueryAnsweringSeqDataLoader_v2
from src.utils.class_util import Writer


parser = argparse.ArgumentParser()
parser.add_argument("--sleep", type=int, default=0)
parser.add_argument("--ckpt", type=str, default='sparse/237/torch_0.005_0.001.ckpt')
parser.add_argument("--model_name", type=, default=10)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-EFO1')
parser.add_argument("--mode", type=str, default='test', choices=['valid', 'test'])
parser.add_argument("--e_norm", type=str, default='Godel', choices=['Godel', 'product'])
parser.add_argument("--c_norm", type=str, default='product', choices=['Godel', 'product'])
parser.add_argument("--max", type=int, default=10)
parser.add_argument("--data_type", type=str, default='EFO1', choices=['BetaE', 'EFO1', 'EFO1_l'])
parser.add_argument("--formula", type=list, default=None)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    model_ckpt = torch.load(args.ckpt)
    n_relation, n_entity = len(relation_matrix_list), relation_matrix_list[0].shape[0]
    if args.cuda < 0:
        cuda_device = torch.device('cpu')
    else:
        cuda_device = torch.device('cuda:{}'.format(args.cuda))
    if args.data_type == 'BetaE':
        formula_path = osp.join(args.data_folder, f'{args.mode}-qaa.json')
    elif args.data_type == 'EFO1':
        formula_path = osp.join(args.data_folder, f'{args.mode}_real_EFO1_qaa.json')
    elif args.data_type == 'EFO1_l':
        formula_path = osp.join(args.data_folder, f'{args.mode}_1000_real_EFO1_qaa.json')
    else:
        raise NotImplementedError
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

