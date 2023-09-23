import torch
import os.path as osp


def merge_relation_matrix_list(n_rel, out, threshold, epsilon):

    relation_matrix_list = []

    for split in range(n_rel):
        single_path = osp.join(out, f'split_{split}_matrix_{threshold}_{epsilon}.ckpt')
        single_relation_matrix = torch.load(single_path)[0]
        single_relation_matrix = single_relation_matrix[0].to(torch.float16)
        relation_matrix_list.append(single_relation_matrix)
        del single_relation_matrix

    torch.save(relation_matrix_list, f'{out}_matrix_list_{threshold}_{epsilon}.ckpt')
    
merge_relation_matrix_list(409, "checkpoints/nl27k", 0.005, 0.001)

print("merge finished")