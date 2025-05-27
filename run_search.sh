#!/bin/bash

# 指定存储参数的文件夹路径
folder_path="config/search"

# 遍历文件夹中的文件
for file in "$folder_path"/*; do
    # 提取文件名
    filename=$(basename "$file")
    filename="config/search/$filename"
    echo "$filename"
    
    # 调用执行函数，并将文件名作为参数传递，并将作业放入后台执行
    python QG_soft_train.py --config "$filename" &
done

# 等待所有后台作业完成
wait

# nohup python sample_hybrid_soft_queries.py --sample_formula_scope zero_soft_efo1 --mode valid --a_mode zero --b_mode equal > log_sample_onet20k_valid_zero_equal.txt
# nohup python sample_hybrid_soft_queries.py --sample_formula_scope zero_soft_efo1 --mode valid --a_mode zero --b_mode random > log_sample_onet20k_valid_zero_random.txt
# nohup python sample_hybrid_soft_queries.py --sample_formula_scope zero_soft_efo1 --mode test --a_mode zero --b_mode equal > log_sample_onet20k_test_zero_equal.txt
# nohup python sample_hybrid_soft_queries.py --sample_formula_scope zero_soft_efo1 --mode test --a_mode zero --b_mode random > log_sample_onet20k_testzero_random.txt