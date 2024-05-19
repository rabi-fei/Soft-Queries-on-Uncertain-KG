import pandas as pd
import numpy as np
import pickle
import argparse
import csv
import json
import os
import os.path as osp
import scipy.stats as stats
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument("--out_folder", type=str, default="result")
parser.add_argument("--dataset", type=str, default="onet20k")
parser.add_argument("--model", type=str, default="ConE")
parser.add_argument("--mode", type=str, default="test")
parser.add_argument("--folder", type=str, default="soft_EFO-1_log/train/FIT_CN15k_soft_Godel.yaml240201.23:32:362fc31387")
parser.add_argument("--construct", type=int, default=1)
parser.add_argument("--step", type=int, default=1000)
parser.add_argument("--group_by_e", type=int, default=0)
# "constrcuct" will construct the csv file to record the results of different types of queries.
# If not, read from a constructed csv file.


formulas = pd.read_csv("data/DNF_train_soft_EFO1.csv")

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model

    if args.construct:
        for i, formula in enumerate(formulas["formula"]):
            fid = "{:0>4d}".format(i)
            #path = osp.join(args.folder, f"test_type{fid}_soft_efo1_qaa.json")
            path = osp.join(args.folder, f"all_logging_test_{args.step}_type{fid}.json")
            #path = f"results/onet20k/main_box/test_type{fid}_soft_efo1_qaa.json"
            with open(path, "r") as f:
                log = json.load(f)
            for key in log.keys():
                if key not in formulas:
                    formulas[key] = ""
                    formulas[key][i] = log[key]
                else:
                    formulas[key][i] = log[key]

        formulas.to_csv(osp.join(args.folder, "agg.csv"))
        formulas_with_metric = formulas 
    else:
        formulas_with_metric = pd.read_csv(osp.join(args.folder, "agg.csv"))

    if args.group_by_e:
        scores_by_metric = {"MAP":[], "NDCG":[], "spearmanr":[], "kendalltau":[]}
        index_without_e = formulas_with_metric["e_num"] == 0
        index_with_e = formulas_with_metric["e_num"] > 0

        for metric in scores_by_metric:
            avg_without_e = (formulas_with_metric[metric][index_without_e]).sum() / (formulas_with_metric["num_queries"][index_without_e]).sum()
            avg_with_e = ( formulas_with_metric[metric][index_with_e]).sum() / (formulas_with_metric["num_queries"][index_with_e]).sum()
            scores_by_metric[metric].extend(["{:.1%}".format(avg_without_e)[:-1], "{:.1%}".format(avg_with_e)[:-1]])
        scores4_latex = ["&".join(scores_by_metric[metric]) for metric in scores_by_metric]
        print(scores4_latex)

    else:
        scores_by_metric = {"MAP":[], "NDCG":[], "spearmanr":[], "kendalltau":[]}
        scores = [[] for i in range(3)]
        scores_str = ["" for i in range(3)]
        for metric in scores_by_metric:
            for i, formula in enumerate(formulas["formula"]):
                s = "{:.1%}".format(formulas[metric][i] / formulas["num_queries"][i])[:-1]
                #s = "{:.1%}".format(formulas[metric][i])[:-1]
                scores_by_metric[metric].append(s)
            s_mean = "{:.1%}".format((formulas[metric]).sum() / formulas["num_queries"].sum())[:-1]
            #s_mean = "{:.1%}".format((formulas[metric] * formulas["num_queries"]).sum() / formulas["num_queries"].sum())[:-1]
            scores_by_metric[metric].append(s_mean)
        print("tau Mean Score:", scores_by_metric["kendalltau"][-1])
        print("spearmanr Mean Score:", scores_by_metric["spearmanr"][-1])
        print("MAP Mean Score:", scores_by_metric["MAP"][-1])
        print("NDCG Mean Score:", scores_by_metric["NDCG"][-1])
        scores4_latex = ["&".join(scores_by_metric[metric]) for metric in scores_by_metric]
        print(scores4_latex)


