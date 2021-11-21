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
