import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict
from typing import List
from math import ceil

import torch


def compute_batch_score(rel, arg1, arg2, rank):
    rel_real, rel_img = rel[:, :, :rank], rel[:, :, rank:]
    arg1_real, arg1_img = arg1[:, :, :rank], arg1[:, :, rank:]
    arg2_real, arg2_img = arg2[:, :, :rank], arg2[:, :, rank:]

    # [B] Tensor
    score1 = torch.sum(rel_real * arg1_real * arg2_real, -1)
    score2 = torch.sum(rel_real * arg1_img * arg2_img, -1)
    score3 = torch.sum(rel_img * arg1_real * arg2_img, -1)
    score4 = torch.sum(rel_img * arg1_img * arg2_real, -1)
    res = score1 + score2 + score3 - score4
    del score1, score2, score3, score4, rel_real, rel_img, arg1_real, arg1_img, arg2_real, arg2_img

    return res


if __name__ == "__main__":
    device = torch.device('cuda:{}'.format(1))
    cqd_path = '/home/hyin/cqd/models/FB15k.ckpt'
    cqd_ckpt = torch.load(cqd_path)
    ent_emb = cqd_ckpt['embeddings.0.weight'].to(device)
    rel_emb = cqd_ckpt['embeddings.1.weight'].to(device)
    n_rel, n_ent = rel_emb.shape[0], ent_emb.shape[0]
    split_num = 269
    split_each_relation = int(n_rel / split_num)
    batch_head = 50
    head_total_batch = ceil(n_ent / batch_head)
    for split in range(201, split_num):
        all_matrix = torch.zeros((split_each_relation, n_ent, n_ent), requires_grad=False)
        for relation_id in range(split_each_relation):
            relation_total_id = relation_id + split * split_each_relation
            print('r_id', relation_total_id)
            for head_batch_id in range(head_total_batch):
                starting_h_id = int(head_batch_id * batch_head)
                batch_head_emb = ent_emb[starting_h_id: starting_h_id + batch_head, :]
                batch_head_emb = batch_head_emb.unsqueeze(-2)
                tail_emb = ent_emb.unsqueeze(0)
                this_rel_emb = rel_emb[relation_total_id].unsqueeze(0).unsqueeze(0)
                batch_score = compute_batch_score(this_rel_emb, batch_head_emb, tail_emb, 1000)
                all_matrix[relation_id, starting_h_id: starting_h_id + batch_head] = batch_score
                del tail_emb, batch_score, batch_head_emb, this_rel_emb
        torch.save(all_matrix, f'/home/hyin/cqd/models/FB15k_matrix/matrix_{split}.ckpt')
        print(f"split{split} saved")
