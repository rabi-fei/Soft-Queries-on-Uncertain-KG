import argparse
import json
import os.path as osp
import pandas as pd

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', choices=['train', 'valid', 'test'])
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-EFO1ex')
parser.add_argument("--max", type=int, default=None)
parser.add_argument("--action", type=str, default='split', choices=['aggregate', 'split'])
parser.add_argument("--data_file", type=str, default="data/DNF_train_EFO1.csv")
parser.add_argument("--data_type", type=str, default="EFO1ex", choices=['EFO1', 'EFO1ex'])

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    all_query_data = {}
    abstract_query_data = pd.read_csv(args.data_file)
    if args.action == 'aggregate':
        for formula_id in abstract_query_data['formula_id']:
            i = int(formula_id.split[4:])
            query_file = open(osp.join(args.data_folder, f'{args.mode}_{i}_real_EFO1_qaa.json'))
            query_data = json.load(query_file)
            for query in query_data:
                if args.max:
                    all_query_data[query] = query_data[query][:args.max]
                else:
                    all_query_data[query] = query_data[query]
        if args.max:
            output_path = osp.join(args.data_folder, f'{args.mode}_{args.max}_qaa.json')
        else:
            output_path = osp.join(args.data_folder, f'{args.mode}_qaa.json')
        with open(output_path, 'wt') as f:
            json.dump(all_query_data, f)
    else:
        query_file = open(osp.join(args.data_folder, f'{args.mode}_qaa.json'))
        all_query_data = json.load(query_file)
        for query in all_query_data:
            if query in abstract_query_data['formula']:
                specific_query_data = {}
                formula_id = abstract_query_data['formula_id'][abstract_query_data['formula'].index(query)]
                if args.max:
                    specific_query_data[query] = all_query_data[query][:args.max]
                else:
                    specific_query_data[query] = all_query_data[query]
                if args.max:
                    output_path = osp.join(args.data_folder,
                                           f'{args.mode}_{args.max}_{formula_id}_{args.data_type}_qaa.json')
                else:
                    output_path = osp.join(args.data_folder, f'{args.mode}_{formula_id}_{args.data_type}_qaa.json')
                with open(output_path, 'wt') as f:
                    json.dump(specific_query_data, f)



