import pickle
import pandas as pd
import json
import os

def parse4latex(df):
    final_scores = df[1250].to_list()
    scores_str =  "&".join(["{:.1%}".format(s)[:-1] for s in final_scores])
    return scores_str

step, max_step = 3000, 450000
target_metric = "NDCG" #["MAP", "NDCG", "spearmanr", "kendalltau"]
cur_lstr = ['r1(s1,f1,0%,1.0)', '(r1(s1,e1,0%,1.0))&(r2(e1,f1,0%,1.0))', '(r1(s1,f1,0%,1.0))&(r2(s2,f1,0%,1.0))', '(!(r1(s1,f1,0%,1.0)))&(r2(s2,f1,0%,1.0))', '(r1(s1,f1,0%,1.0))&(r2(e1,f1,0%,1.0))', '(r1(s1,e1,0%,1.0))&((r2(e1,f1,0%,1.0))&(r3(e1,f1,0%,1.0)))', '(r1(s1,f1,0%,1.0))|(r2(s2,f1,0%,1.0))', '(!(r1(s1,f1,0%,1.0)))&((r2(s2,f1,0%,1.0))&(r3(s3,f1,0%,1.0)))', '(r1(s1,e1,0%,1.0))&((r2(s2,e1,0%,1.0))&(r3(e1,f1,0%,1.0)))', '(r1(s1,e1,0%,1.0))&((r2(s2,e1,0%,1.0))&((r3(e1,f1,0%,1.0))&(r4(e1,f1,0%,1.0))))', '(!(r1(s1,e1,0%,1.0)))&((r2(s2,e1,0%,1.0))&(r3(e1,f1,0%,1.0)))']
df = pd.DataFrame(columns=["lstr"])
collecte_scores = []
df["lstr"] = cur_lstr
for check_step in range(step, max_step+step, step):
    results = {}
    scores = []
    for i in range(12):
        fid = "{:0>4d}".format(i)
        file = f"soft_EFO-1_log/train/ConE_ONET20k_soft_random_zero.yaml240121.23:18:2507c6882d/all_logging_test_{check_step}.json"
        if not os.path.exists(file):
            continue
        with open(file, "rb") as f:
            score = json.load(f)

    #df[check_step] = scores
    collecte_scores.append(scores)
    print(f"sucessfully parse the result of {check_step}")

socres_str = parse4latex(df)

df.to_csv(f"soft_EFO-1_log/train_embedding_models/FIT_CN15k_soft_Godel.yaml240107.12:16:19bd4d4330/{metric}.csv")
print(f"sucessfully parsed the results")