#!/usr/local/Cellar/bash/5.1.16/bin/bash

declare -A name_dict
name_dict['2998076']='bs2048_lr0.0001_sgm3_nss10'
name_dict['2998075']='bs2048_lr0.0001_sgm3_nss1'
name_dict['2998073']='bs2048_lr0.0001_sgm1_nss10'
name_dict['2998072']='bs2048_lr0.0001_sgm1_nss1'
name_dict['2998071']='bs2048_lr0.001_sgm3_nss10'
name_dict['2998069']='bs2048_lr0.001_sgm3_nss1'
name_dict['2998068']='bs2048_lr0.001_sgm1_nss10'
name_dict['2998067']='bs2048_lr0.001_sgm1_nss1'
name_dict['2998066']='bs256_lr0.0001_sgm3_nss10'
name_dict['2998063']='bs256_lr0.0001_sgm1_nss10'
name_dict['2998062']='bs256_lr0.0001_sgm1_nss1'
name_dict['2998060']='bs256_lr0.001_sgm3_nss10'
name_dict['2998056']='bs256_lr0.001_sgm1_nss1'

mkdir log/search

for jid in "${!name_dict[@]}";
do
    echo $jid
    echo ${name_dict[$jid]}
    ngc result download $jid \
            --file "**/output.log" \
            --dest log/search

done