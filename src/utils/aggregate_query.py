import argparse
import json
import os.path as osp

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='test', choices=['valid', 'test'])
parser.add_argument("--data_folder", type=str, default='data/NELL-EFO1')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    all_query_data = {}
    for i in [0,1,2,3,4,5,7,8,9]:
        query_file = open(osp.join(args.data_folder, f'{args.mode}_{i}_real_EFO1_qaa.json'))
        query_data = json.load(query_file)
        for query in query_data:
            all_query_data[query] = query_data[query]
    with open(osp.join(args.data_folder, f'{args.mode}_real_EFO1_qaa.json'), 'wt') as f:
        json.dump(all_query_data, f)
