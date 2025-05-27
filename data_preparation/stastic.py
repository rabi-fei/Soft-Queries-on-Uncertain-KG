import tqdm
import json
import os
import numpy as np
graph_paths = ["data/cn15k"]
#graph_paths = ["data/FB15k", "data/FB15k-237", "data/NELL"]

#Target: get the entity number, relation number
for graph_path in graph_paths:
    files = ["train", "val", "test"]
    rel2uncertain = {}
    for file in files:
        target_file = graph_path + "/" + file + ".txt"
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
            values = np.array(rel2uncertain[rel])
            filted_values = values[values > 0]
            pre_25 = float("{:.4f}".format(np.percentile(filted_values, 25)))
            pre_50 = float("{:.4f}".format(np.percentile(filted_values, 50)))
            pre_75 = float("{:.4f}".format(np.percentile(filted_values, 75)))
            rel2percentile[rel] = [pre_25, pre_50, pre_75]
    print(len(rel2percentile))
    with open(os.path.join(graph_path, 'percentile_25_50_75.json'), 'w') as f:
            json.dump(rel2percentile, f)
#        print(f"the number of 1p queries is {len(queries_1p)}")
