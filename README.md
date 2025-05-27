# Extending Complex Logical Queries on Uncertain Knowledge Graph

This repository is the implementation for the paper "Extending Complex Logical Queries on Uncertain Knowledge Graph".



## 1 Preparation

### 1.1 Environment

We have utilized a CSP solver provided in the python-constraint package, please install it by:
```
pip install python-constraint
```

We have also utilized the pytorch-geometric and networkx package, please install it by:
```
conda install pyg -c pyg
conda install networkx
```


### 1.1 Data Preparation


Please download the Soft Queries on Uncertain Knowledge graph (SQUK) dataset from [here](https://drive.google.com/file/d/1cGu4zMfAtkK3VyNyL151ejGptugmJq3X/view?usp=sharing), 
the data of three knowledge graphs can be downloaded separately and put it in the `data` folder.

Then, after unzipping the query data. an example data folder should look like this:
```
data/cn15k/
  - kgindex.json
  - train_kg.txt
  - valid_kg.txt
  - test_kg.txt
  - percentile_25_50_75.json
  - test_type0000_soft_efo1_qaa.json
  - ......
```

The `kgindex.json` and `percentile_25_50_75.json` are the index file and percentile file for the uncertain knowledge graph respectively, 
the `train_kg.txt`, `valid_kg.txt`, and `test_kg.txt` are the training graph,  validation graph, and  test graph respectively. They are used for data generation.

The following are the source of uncertain knowledge graphs.

```
cn15k from [here] (https://github.com/stasl0217/beurre/tree/main/data/cn15k)
ppi5k from [here] (https://github.com/stasl0217/UKGE/tree/master/data/ppi5k)
onet20k from [here] (https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/onet20k.zip)
```

To get the `kgindex.json` and `percentile_25_50_75.json`, run the follow command:
```angular2html
python data_preparation/stastic.py
python data_preparation/transform_kg.py
```

### 1.2 Checkpoint Preparation

To reproduce the experiment in the paper, we have provided the checkpoint for each model foreach knowledge graph, we
offer the checkpoint for six representative model (LogicE, ConE, SIU), which can be downloaded from [here](https://drive.google.com/file/d/1PGtpCqUVXOTQH5MSHfkwCDYzyObq3T29/view?usp=sharing),


It should be unzipped and put in the `ckpt` folder.

An example of the `ckpt` sub folder, which includes the model trained on the knowledge graph ``FB15k-237'' should look like this:
```
ckpt/cn15k
  - LogicE_full/450000.ckpt
  - ConE_full/300000.ckpt
  - SIU/beurre.pt
  - SIU/ukge.pt
```

where each sub folder is the checkpoint for each model, and the name of the sub folder is the name of the model.

To generate the matrix list used for SIU with UKGE, please run the command:
```angular2html
python create_matrix_for_UKG.py --ckpt_path ckpt/onet20k/ukge.pt --data_folder data/processed/onet20k --output_folder ckpt/onet20k
```
## 2. Sample the data yourself

We have the powerful frame that supports several key functionalities for the task of soft query answering, you can also sample the query by yourself following the instruction. 

```angular2html
python sample_hybrid_soft_queries.py --sample_formula_scope zero_soft_efo1 --mode valid --a_mode zero --b_mode equal

```

If you have downloaded the SQUK dataset, you can also skip this section.


## 2. Reproduce the result of the paper.

### 2.1 main experiments
Please run the following commands to reproduce our main results.
```

iterations=11
for ((i=0; i<=$iterations; i++))
do
    path=$(printf "test_type%04d_soft_efo1_qaa.json" $i)
    python solve_soft_EFO1.py --cuda 0 --data_folder data/processed/onet20k --out_folder results/onet20k/main_box --query_path "$path" --ckpt checkpoints/onet20k/full_matrix_list_0.1_0.001.ckpt
done
```


