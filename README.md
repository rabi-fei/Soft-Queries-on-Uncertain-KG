# On Existential First Order Queries Inference on Knowledge Graphs

This repository is for implementation for the paper "On Existential First Order Queries Inference on Knowledge Graphs".

See the arXiv version [here](https://arxiv.org/abs/2304.07063).

## 1 Preparation

### 1.1 Data Preparation
Please download the data from [here](https) and put it in the `data` folder.


### 1.2 Matrix Creation

The matrix that has been used in the paper can be downloaded as 

## 2. Run the FIT code.

For the reproduction of the experiment in paper, run the following code:
 you can also run the solve_EFO1.v2   for the experiments on the NELL dataset, if you
```
## python solve_EFO1.py
```

In case you have problem with your gpu memory, for example, for the experiments on the NELL dataset, you can run the following code:
```
## python solve_EFO1.v2.py --ckpt 'sparse/NELL/torch_0.01_0.001.ckpt'
```

If you want to reproduce the ablation study of the influence of hyperparameter, you can run the following code.
For different c_norm:
```
## python solve_EFO1.py --c_norm Godel
```
For different max enumeration:
```
## python solve_EFO1.py --max 5
## python solve_EFO1.py --max 20
```

For different epsilon, delta:
```
## python solve_EFO1.py --ckpt 'sparse/237/torch_0.01_0.001.ckpt'
## python solve_EFO1.py --ckpt 'sparse/237/torch_0.01_0.ckpt'
```

## 3. Citing the paper

Please cite the paper if you found the resources in this repository useful.

```
@article{yin2023existential,
  title={On Existential First Order Queries Inference on Knowledge Graphs},
  author={Yin, Hang and Wang, Zihao and Song, Yangqiu},
  journal={arXiv preprint arXiv:2304.07063},
  year={2023}
}