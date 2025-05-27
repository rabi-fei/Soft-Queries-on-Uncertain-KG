#!/bin/bash

target_folder="data/UKG/ppi5k/"

cp data/ppi5k/kgindex.json "$target_folder"
cp data/ppi5k/percentile_25_50_75.json "$target_folder"
cp data/ppi5k/test.txt "$target_folder"
cp data/ppi5k/train.txt "$target_folder"
cp data/ppi5k/valid.txt "$target_folder"
# cp data/processed/onet20k/train_qaa.json "$target_folder"

# iterations=11

# for ((i=0; i<=$iterations; i++))
# do
#     # path=$(printf "data/processed/cn15k/test_type%04d_soft_efo1_qaa.json" $i)
#     path=$(printf "data/UKG/onet20k/test_type%04d_soft_efo1_qaa.json" $i)

#     # cp  data/UKG/cn15k/ "$path"
#     cp   "$path" data/processed/onet20k

# done

