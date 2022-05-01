from src.utils.data import QueryAnsweringMixDataLoader

qad = QueryAnsweringMixDataLoader('data/FB15k-237-betae/test-qaa.json',
                            batch_size=7,
                            shuffle=True,
                            num_workers=0)
for fof_list in qad:
    for fof in fof_list:
        print(fof.formula.to_lstr())
        print(len(fof.easy_answer))
        print(len(fof.hard_answer))

    print()