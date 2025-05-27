import json
import pandas as pd
import os


df = pd.read_csv("data/DNF_train_soft_EFO1.csv")
data_folder = "data/processed_from_2335/onet20k"

all_data = {}
for type_id in df["formula_id"]:
    if type_id == "type0007":
        continue
    single_formula_name = f"test_{type_id}_soft_efo1_qaa.json"
    single_formula_path = os.path.join(data_folder, single_formula_name)
    if not os.path.exists(single_formula_path):
        continue

    with open(single_formula_path, "r") as f:
        single_data = json.load(f)
    all_data.update(single_data)
save_path = os.path.join(data_folder, "all_data.json")
with open(save_path, "w") as f:
        single_data = json.dump(all_data, f)

print("finished")