import torch


def triples_to_tensors(triples, device=None):
    H, R, T = [], [], []
    for h, r, t in triples:
        H.append(h)
        R.append(r)
        T.append(t)
    if device is None:
        device = 'cpu'
    return torch.tensor(H, device=device), torch.tensor(R, device=device), torch.tensor(T, device=device)

# def densify_ragged_matrix(values, count, fill, **kwargs):
#     num_rows = len(count)
#     num_cols = count.max()
#     limit = count.cumsum(dim=0)
#     base = torch.ones(size=(num_rows, num_cols), **kwargs) * fill
#     _row_id = torch.arange(len(values), **kwargs)
#     if _row_id == limit
