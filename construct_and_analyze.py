import pandas as pd
import numpy as np
import pickle
import argparse
import csv
import json
import os
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument("--out_folder", type=str, default="result")
parser.add_argument("--dataset", type=str, default="ppi5k")
parser.add_argument("--model", type=str, default="BetaE")
parser.add_argument("--task", type=str, default="low")
parser.add_argument("--construct", type=int, default=1)
# "constrcuct" will construct the csv file to record the results of different types of queries.
# If not, read from a constructed csv file.


formulas = pd.read_csv("data/DNF_train_soft_EFO1.csv")

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    task = args.task

    if args.construct:
        for i, formula in enumerate(formulas["formula"]):
            with open("results/{}/{}/{}.json".format(dataset, task, formula), "rb") as f:
                log = json.load(f)
            for key in log.keys():
                if key not in formulas:
                    formulas[key] = ""
                    formulas[key][i] = log[key]
                else:
                    formulas[key][i] = log[key]

        formulas.to_csv(f"results/{dataset}_{task}_test_box_agg.csv")
    else:
        formulas_with_metric = pd.read_csv(f"result/{dataset}_{task}_test_box_agg.csv")


    scores_by_metric = {"MAP":[], "NDCG":[], "spearmanr":[], "kendalltau":[]}
    scores = [[] for i in range(3)]
    scores_str = ["" for i in range(3)]
    for i, formula in enumerate(formulas["formula"]):

        for metric in scores_by_metric:
            s = "{:.1%}".format(formulas[metric][i])[:-1]
            scores_by_metric[metric].append(s)
    scores4_latex = ["&".join(scores_by_metric[metric]) for metric in scores_by_metric]
    print(scores4_latex)


