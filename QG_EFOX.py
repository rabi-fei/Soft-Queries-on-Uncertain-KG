import argparse
import os.path as osp
from collections import defaultdict

import torch
import tqdm
import json

from FIT import solve_EFO1
from src.utils.data import QueryAnsweringSeqDataLoader_v2
from src.utils.class_util import Writer


parser = argparse.ArgumentParser()
parser.add_argument("--sleep", type=int, default=0)
parser.add_argument("--config", type=str, default="config/LogicE_FB15k-237_EFOX.yaml")
parser.add_argument("--ckpt", type=str, default='sparse/237/torch_0.005_0.001.ckpt')
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-EFO1')
parser.add_argument("--formula", type=list, default=None)


def read_from_yaml(yaml_path):
    import yaml
    with open(yaml_path, 'r') as fd:
        return yaml.load(fd, Loader=yaml.FullLoader)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    configure = read_from_yaml(args.config)
    if configure['cuda'] < 0:
        cuda_device = torch.device('cpu')
    else:
        cuda_device = torch.device('cuda:{}'.format(configure['cuda']))
    all_formula_data = json.load(open(configure['evaluate']['formula_id_file'], 'r'))
    case_name = configure['output']['output_path'] if configure['output']['output_path'] else \
        args.config.split("config")[-1][1:]
    writer = Writer(case_name=case_name, config=configure, log_path=configure["output"]["prefix"])
    if 'test' in configure['action']:
        for i, row in tqdm.tqdm(all_formula_data.iterrows(), total=len(all_formula_data)):
            formula = row['formula']
            data_path = osp.join(args.data_folder, f'{args.mode}_type{i:04d}_EFOX_qaa.json')
            if not osp.exists(data_path):
                print(f'Warnings,{data_path} not exists!')
            test_dataloader = QueryAnsweringSeqDataLoader_v2(
                data_path,
                target_lstr=None,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0)
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
                    pass
                batch_ans_tensor = torch.stack(batch_ans_list, dim=0)
                # all_answers[fof.lstr][now_formula_index[fof.lstr]: now_formula_index[fof.lstr] + batch_ans_tensor.shape[0], :] \
                # = batch_ans_tensor
                # now_formula_index[fof.lstr] += batch_ans_tensor.shape[0]
            for full_formula in all_metrics.keys():
                for log_metric in all_metrics[full_formula].keys():
                    if log_metric != 'num_queries':
                        all_metrics[full_formula][log_metric] /= all_metrics[full_formula]['num_queries']
            print(all_metrics)
            # writer.save_torch(all_answers, 'all_answer_tensor.ckpt')
            # mwriter.save_pickle(all_metrics, f"all_logging_{args.mode}_0.pickle")
        model_ckpt = torch.load(args.ckpt)




