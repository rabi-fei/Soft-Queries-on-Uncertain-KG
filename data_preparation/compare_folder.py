


folder1, folder2 = "data/processed/cn15k", "data/cn15k"
files_1 = ["valid.txt", "test.txt"]
files_2 = ["val.tsv", "test.tsv"]

origin_collect = []
for file in files_1:
    with open(folder1 + "/" + file, "r") as f:
        facts = f.readlines()
    handled_facts = []
    for fact in facts:
        h,r,t,p = fact.split("\t")
        handled_facts.append([int(h), int(r), int(t)])
    origin_collect.append(handled_facts)

new_collect = []
for file in files_1:
    with open(folder1 + "/" + file, "r") as f:
        facts = f.readlines()
    handled_facts = []
    for fact in facts:
        h,r,t,p = fact.split("\t")
        handled_facts.append([int(h), int(r), int(t)])
    new_collect.append(handled_facts)

for i in range(2):
    for triple in origin_collect[i]:
        if triple not in new_collect[i]:
            print("what")


        

