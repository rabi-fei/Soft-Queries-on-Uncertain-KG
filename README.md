# EFO<sub>k</sub>-CQA: Towards Knowledge Graph Complex Query Answering beyond Set Operation

This repository is for implementation for the paper "EFO<sub>k</sub>-CQA: Towards Knowledge Graph Complex Query Answering beyond Set Operation".



## 1 Preparation

### 1.1 Environment

We have utilized a CSP solver provided in the python-constraint package:
```
pip install python-constraint
```


### 1.1 Data Preparation


Please download the data from [here](https://drive.google.com/drive/folders/1kqnRdpcnVdBfbY8eVRIoVUgXdgkU8qd4?usp=sharing), 
the data of three knowledge graphs can be downloaded separately and put it in the `data` folder, as well as a file named 
`DNF_EFO2_23_4123166.csv` which is used to store the abstract query graph(query type) for the EFOX experiment.

The `DNF_EFO2_23_4123166.csv` should be put into the `data` folder.

Then, after unzipping the query data. an example data folder should look like this:
```
data/FB15k-237-EFOX-final/
  - kgindex.json
  - train_kg.tsv
  - valid_kg.tsv
  - test_kg.tsv
  - test-type0000-EFOX-qaa.json 
  - test-type0001-EFOX-qaa.json
  - ......
```

where the `test-type0000-EFOX-qaa.json` is used for the EFOX experiment, containing the data for query type0000. 

The `kgindex.json` and `train_kg.tsv` are the index file and the training graph for the knowledge graph respectively, 
the `valid_kg.tsv` and `test_kg.tsv` are the validation graph and the test graph respectively. They are used for data generation.






### 1.2 Checkpoint Preparation

To reproduce the experiment in the paper, we have provided the checkpoint for each model foreach knowledge graph.
The checkpoint for six representative model can be downloaded from [here](https://drive.google.com/drive/folders/13S3wpcsZ9t02aOgA11Qd8lvO0JGGENZ2?usp=sharing),


It should be unzipped and put in the `ckpt` folder.

An example of the `ckpt` sub folder should look like this:
```
sparse/FB15k-237
  - BetaE/450000.ckpt
  - torch_0.005_0.001.ckpt
```

## 2. Sample the data yourself

We have the powerful frame that supports several key functionalities for the task of complex query answering, 
you can also sample the query by yourself



### 2.1 Enumerate the abstract query graph

```angular2html
python data_preparation/create_qg.py
```

### 2.2 Sample the query graph

```angular2html
python sample_query.py
```

## 3. Reproduce the result of the paper.

### 3.1 

For query embedding method, including BetaE, LogicE, ConE, please run the following command:

```angular2html
python QG_EFOX.py --config config/LogicE_FB15k-237.yaml
```

which is an example for LogicE method on FB15k-237 dataset.

For CQD and LMPNN, please run the following command:

```angular2html
python train_lmpnn.py 
```

which is an example for LMPNN method on FB15k-237 dataset.

For FIT, please run the following command: 

```angular2html
python solve_EFOX.py 
```

