import torch


def lcwa_negative_sampling(phead_id_ten, ptail_id_ten, num_entities):

    random_entities = torch.randint(
        low=0, high=num_entities, size=phead_id_ten.shape)

    head_collapse = torch.randint(low=0,
                                  high=2,
                                  size=phead_id_ten.shape).bool()
    tail_collapse = head_collapse.logical_not()

    nhead = torch.where(head_collapse, random_entities, phead_id_ten)
    ntail = torch.where(tail_collapse, random_entities, ptail_id_ten)
    return nhead, ntail
