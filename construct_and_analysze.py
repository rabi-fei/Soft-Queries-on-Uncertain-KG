import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
toDo = ["constrcuct", "analyze"]
task = "FB15k237"
model = "beta"
variable = [2]
formulas = pd.read_csv("data/DNF_EFO2_23_4123166.csv")

formulas_1 = formulas[formulas["f_num"]==1].copy()
formulas_2 = formulas[formulas["f_num"]==2]

if "constrcuct" in toDo:
    for i in formulas_1["index"]:
        with open("result/{}_result/{}_test/all_logging_test_0_type{:0>4d}.pickle".format(task, model, i), "rb") as f:
            log = pickle.load(f)
        for key in log[formulas_1["formula"][i]][0].keys():
            if key not in formulas_1:
                formulas_1[key] = ""
                formulas_1[key][i] = log[formulas_1["formula"][i]][0][key]
            else:
                formulas_1[key][i] = log[formulas_1["formula"][i]][0][key]
    formulas_1.to_csv(f"result/{task}_agg/DNF_{model}_1.csv")
    for i in formulas_2["index"]:
        with open("EFO-log/{}_result/{}_test/all_logging_test_0_type{:0>4d}.pickle".format(task, model, i), "rb") as f:
            log = pickle.load(f)
        for j in range(3):
            for key in log[formulas_2["formula"][i]][j].keys():
                if f"{key}_{j}" not in formulas_2:
                    formulas_2[f"{key}_{j}"] = ""
                    formulas_2[f"{key}_{j}"][i] = log[formulas_2["formula"][i]][j][key]
                else:
                    formulas_2[f"{key}_{j}"][i] = log[formulas_2["formula"][i]][j][key]
    formulas_2.to_csv(f"result/{task}_agg/DNF_{model}_2.csv")
else:
    if 1 in variable:
        formulas_1 = pd.read_csv(f"result/{task}_agg/DNF_{model}_{1}.csv")
    if 2 in variable:
        formulas_2 = pd.read_csv(f"result/{task}_agg/DNF_{model}_{2}.csv")
if 1 in variable:
    metric = "MRR"
    scores = [[] for i in range(3)]
    scores_str = ["" for i in range(3)]
    for c in range(1,4):
            for e  in range(3):
                r_e = np.array(formulas_1["e_num"] == e) & np.array(formulas_1["c_num"] == c)
                if np.any(r_e & ~np.array(formulas_1["multi"]) & ~np.array(formulas_1["cyclic"])):
                    index_red = r_e & ~np.array(formulas_1["multi"]) & ~np.array(formulas_1["cyclic"])
                    avg_r_e_DAG = formulas_1[metric][index_red].sum() / formulas_1["num_queries"][index_red].sum()
                    scores[c-1].append(avg_r_e_DAG)
                    scores_str[c-1] += "&{:.1%}".format(avg_r_e_DAG)[:-1]
                if np.any(r_e & np.array(formulas_1["multi"])) :
                    index_rem = r_e & np.array(formulas_1["multi"])
                    avg_r_e_M = formulas_1[metric][index_rem].sum() / formulas_1["num_queries"][index_rem].sum()
                    scores[c-1].append(avg_r_e_M)
                    scores_str[c-1] += "&{:.1%}".format(avg_r_e_M)[:-1]
                if np.any(r_e & np.array(formulas_1["cyclic"])):
                    index_rec = r_e & np.array(formulas_1["cyclic"])
                    avg_r_e_C = formulas_1[metric][index_rec].sum() / formulas_1["num_queries"][index_rec].sum()
                    scores[c-1].append(avg_r_e_C)
                    scores_str[c-1] += "&{:.1%}".format(avg_r_e_C)[:-1]
            index_c = np.array(formulas_1["c_num"] == c)
            avg_c = formulas_1[metric][index_c].sum() / formulas_1["num_queries"][index_c].sum()
            scores_str[c-1] += "&{:.1%}".format(avg_c)[:-1]
            scores[c-1].append(avg_c)

    scores_avge_str = ""
    for e in range(3):
        index_e = np.array(formulas_1["e_num"] == e)
        if np.any(index_e & ~np.array(formulas_1["multi"]) & ~np.array(formulas_1["cyclic"])):
            index_ed = index_e & ~np.array(formulas_1["multi"]) & ~np.array(formulas_1["cyclic"])
            avg_e_DAG = formulas_1[metric][index_ed].sum() / formulas_1["num_queries"][index_ed].sum()
            scores_avge_str += "&{:.1%}".format(avg_e_DAG)[:-1]
        if np.any(index_e & np.array(formulas_1["multi"])):
            index_em = index_e & np.array(formulas_1["multi"])
            avg_e_M = formulas_1[metric][index_em].sum() / formulas_1["num_queries"][index_em].sum()
            scores_avge_str += "&{:.1%}".format(avg_e_M)[:-1]
        if np.any(index_e & np.array(formulas_1["cyclic"])):
            index_ec = index_e & np.array(formulas_1["cyclic"])
            avg_e_C = formulas_1[metric][index_ec].sum() / formulas_1["num_queries"][index_ec].sum()
            scores_avge_str += "&{:.1%}".format(avg_e_C)[:-1]
    avg = formulas_1[metric].sum() / formulas_1["num_queries"].sum()
    formulas_1.to_csv(f"data/DNF_{model}_1.csv")

if 2 in variable:
    metric2 = ["marginal_HITS10_0", "HITS10*10_0","HITS10_0"]
    scores = [[] for i in range(3)]
    label = [[] for i in range(3)]
    scores_str = ["" for i in range(len(metric2))]
    id_fc = np.array(formulas_2["c_num"] == 3) & np.array(formulas_2["f_num"] == 2)
    for i in range(len(metric2)):
        metric_index = metric2[i]
        for e  in range(3):
                    id_fce = id_fc & np.array(formulas_2["e_num"] == e)
                    if np.any(id_fce & ~np.array(formulas_2["multi"]) & ~np.array(formulas_2["cyclic"])):
                        id_fced = id_fce & ~np.array(formulas_2["multi"]) & ~np.array(formulas_2["cyclic"])
                        avg_fced = formulas_2[metric_index][id_fced].sum() / formulas_2["num_queries_0"][id_fced].sum()
                        label[i].append("DAG")
                        scores[i].append(avg_fced)
                        scores_str[i] += "&{:.1%}".format(avg_fced)[:-1]
#                    else:
#                         scores_str[i] += "&"

                    if np.any(id_fce & np.array(formulas_2["multi"])) :
                        id_fcem = id_fce & np.array(formulas_2["multi"])
                        avg_fcem = formulas_2[metric_index][id_fcem].sum() / formulas_2["num_queries_0"][id_fcem].sum()
                        label[i].append("Multi")
                        scores[i].append(avg_fcem)
                        scores_str[i] += "&{:.1%}".format(avg_fcem)[:-1]
#                    else:
#                         scores_str[i] += "&"

                    if np.any(id_fce & np.array(formulas_2["cyclic"])):
                        id_fcec = id_fce & np.array(formulas_2["cyclic"])
                        avg_fcec = formulas_2[metric_index][id_fcec].sum() / formulas_2["num_queries_0"][id_fcec].sum()
                        label[i].append("Cyclic")
                        scores[i].append(avg_fcec)
                        scores_str[i] += "&{:.1%}".format(avg_fcec)[:-1]
#                    else:
#                         scores_str[i] += "&"

        avg_fc_metric =  formulas_2[metric_index][id_fc].sum() / formulas_2["num_queries_0"][id_fc].sum()
        scores_str[i] += "&{:.1%}".format(avg_fc_metric)[:-1]
    print("here")

