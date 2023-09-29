import tqdm
import json
import os
import numpy as np
graph_paths = ["data/ppi5k"]
#graph_paths = ["data/FB15k", "data/FB15k-237", "data/NELL"]

#Target: get the entity number, relation number
for graph_path in graph_paths:
    files = ["train", "valid", "test"]
    rel2uncertain = {}
    for file in files:
        target_file = graph_path + "/" + file + ".tsv"
        with open(target_file, "r",  encoding='utf-8') as f:
            for fact in tqdm.tqdm(f.readlines()):
                h, r, t, p = fact.rstrip().split("\t")
                if (int(h), int(r), int(t)) == (2217, 2, 2286):
                    print((h,r,t,p))
                if int(r) not in rel2uncertain:
                    rel2uncertain[int(r)] = [float(p)]
                else:
                    rel2uncertain[int(r)].append(float(p))
    rel2percentile = {}
    for rel in sorted(rel2uncertain.keys()):
            pre_25 = np.percentile(rel2uncertain[rel], 25)
            pre_50 = np.percentile(rel2uncertain[rel], 50)
            pre_75 = np.percentile(rel2uncertain[rel], 75)
            rel2percentile[rel] = [pre_25, pre_50, pre_75]
    print(len(rel2percentile))
    with open(os.path.join(graph_path, 'percentile_25_50_75.json'), 'w') as f:
            json.dump(rel2percentile, f)
#        print(f"the number of 1p queries is {len(queries_1p)}")