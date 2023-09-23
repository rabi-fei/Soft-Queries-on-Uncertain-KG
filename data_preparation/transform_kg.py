import csv
import os
import json

read_folder = "data/nl27k"
save_folder = "data/processed/nl27k"
file_names = ["train.tsv", "val.tsv", "test.tsv"]


names = ["entity_id.csv", "relation_id.csv"]
saved_dict = {}
for name in names:
    map_id = {}
    read_path = os.path.join(read_folder, name)
    saved_path = os.path.join(save_folder, "kgindex.json")
    with open(read_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            if i==0:
                continue
            id, key = row
            map_id[key] = int(id)
    saved_dict[name[0]] = map_id
with open(saved_path, "w") as f:
    json.dump(saved_dict, f)

saved_names = ["train.txt", "valid.txt", "test.txt"]
for i, file_name in enumerate(file_names):
    read_path = os.path.join(read_folder, file_name)
    saved_path = os.path.join(save_folder, saved_names[i])
    with open(saved_path, "w") as f:
        with open(read_path, 'r', encoding='utf-8') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')  # 指定制表符为分隔符
            for row in reader:
                h,r,t,p = row
                f.writelines("\t".join(row)+"\n")
    print("finish a file")

#json.load()



