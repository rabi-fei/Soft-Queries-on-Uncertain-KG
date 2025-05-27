#!/bin/bash

# 定义循环次数
iterations=11

# 循环执行命令
for ((i=0; i<=$iterations; i++))
do
    # 生成路径字符串
    path=$(printf "test_type%04d_soft_efo1_qaa.json" $i)

    # 执行命令，将路径作为参数传递给 Python 脚本
    python solve_soft_EFO1.py --cuda 0 --data_folder data/processed/onet20k --out_folder results/onet20k/main_box --query_path "$path" --ckpt checkpoints/onet20k/full_matrix_list_0.1_0.001.ckpt
done

#python construct_and_analyze.py --dataset ppi5k --mode test -- task hybrid_train
