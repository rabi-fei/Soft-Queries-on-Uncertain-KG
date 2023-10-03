import argparse
import pickle
import os.path as osp
from math import ceil

import torch

#from create_matrix import create_matrix_from_ckpt
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default='checkpoints/ppi5k/params_numpy')
parser.add_argument("--ckpt_type", type=str, default='ukge', choices=['cqd', 'ukge'])
parser.add_argument("--data_folder", type=str, default='data/processed/ppi5k')
parser.add_argument("--cuda", type=int, default=1)
parser.add_argument("--batch", type=int, default=1000)
parser.add_argument("--output_folder", type=str, default='checkpoints/ppi5k')


def compute_batch_score_complex(rel, arg1, arg2, rank):
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


def create_matrix_from_ckpt_for_UKG(scoring_matrix, observed_kg: KnowledgeGraph, real_starting_r, threshold=0.01, epsilon=0.01):
    n_rel, n_entity = scoring_matrix.shape[0], scoring_matrix.shape[1]
    full_tail_prob = scoring_matrix
    sparse_matrix_list = []
    for rel_id in range(n_rel):
        for h_id in range(n_entity):
            tail_prob = full_tail_prob[rel_id][h_id]

            full_tail_prob[rel_id][h_id] = torch.where(full_tail_prob[rel_id][h_id] > threshold,
                                                       full_tail_prob[rel_id][h_id], torch.zeros(n_entity))
            full_tail_prob[rel_id][h_id] = full_tail_prob[rel_id][h_id].clamp(0, 1-epsilon)

        sparse_matrix_list.append(full_tail_prob[rel_id].to_sparse())
    return sparse_matrix_list

def compute_batch_score_transe(rel, h_emb, t_emb):
    difference = rel + h_emb - t_emb
    score = -(torch.linalg.norm(difference, dim=-1))
    return score


def compute_batch_score_distmult(rel, h_emb, t_emb):
    score = torch.sum(rel * h_emb * t_emb, dim=-1)
    return score

def compute_batch_score_ukge(rel, h_emb, t_emb, w, b):
    score = w * torch.sum(rel * h_emb * t_emb, dim=-1) + b
    return score


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda))
    data_folder = args.data_folder
    cqd_path = args.ckpt_path
    kgidx = KGIndex.load(osp.join(data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        quadruple_files=osp.join(data_folder, 'train.txt'),
        kgindex=kgidx)
    threshold, epsilon = 0.05, 0.001

    if args.ckpt_type == 'ukge':
        with open(cqd_path, "rb") as handle:
            params_dict = pickle.load(handle)
        ent_emb = torch.tensor(params_dict['entity_embedding'], device=device)
        rel_emb = torch.tensor(params_dict['relation_embedding'], device=device)
        w, b = params_dict["w"], params_dict["b"]
    else:
        cqd_ckpt = torch.load(cqd_path)
        model_param = cqd_ckpt['model'][0]
        ent_emb = model_param['_entity_embedder.embeddings.weight']
        rel_emb = model_param['_relation_embedder.embeddings.weight']
    n_rel, n_ent = rel_emb.shape[0], ent_emb.shape[0]
    split_num = n_rel
    split_each_relation = int(n_rel / n_rel) # deal 
    batch_head = args.batch
    head_total_batch = ceil(n_ent / batch_head)
    sparse_list, part_sparse_list = [], []
    for split in range(0, split_num):
        for relation_id in range(split_each_relation):
            all_matrix = torch.zeros((1, n_ent, n_ent), requires_grad=False)
            relation_total_id = relation_id + split * split_each_relation
            print('r_id', relation_total_id)
            for head_batch_id in range(head_total_batch):
                starting_h_id = int(head_batch_id * batch_head)
                batch_head_emb = ent_emb[starting_h_id: starting_h_id + batch_head, :]
                batch_head_emb = batch_head_emb.unsqueeze(-2)
                tail_emb = ent_emb.unsqueeze(0)
                this_rel_emb = rel_emb[relation_total_id].unsqueeze(0).unsqueeze(0)
                if args.ckpt_type == 'ukge':
                    batch_score = compute_batch_score_ukge(this_rel_emb, batch_head_emb, tail_emb, w, b)
                else:
                    batch_score = compute_batch_score_distmult(this_rel_emb, batch_head_emb, tail_emb)
                all_matrix[0, starting_h_id: starting_h_id + batch_head] = batch_score
                del tail_emb, batch_score, batch_head_emb, this_rel_emb
            sparse_one_list = create_matrix_from_ckpt_for_UKG(all_matrix, train_kg, relation_total_id, threshold,
                                                      epsilon)
            del all_matrix
            sparse_list.extend(sparse_one_list)
        if len(part_sparse_list) > 5:
            torch.save(part_sparse_list, osp.join(args.output_folder, f'split_{split}_matrix_{threshold}_{epsilon}.ckpt'))
        print(f"split{split} done")
    torch.save(sparse_list, osp.join(args.output_folder, f'full_matrix_list_{threshold}_{epsilon}.ckpt'))
    print("sucessfully saved")
    
