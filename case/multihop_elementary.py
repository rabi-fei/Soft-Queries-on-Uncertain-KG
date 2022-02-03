"""
This file studies the random PU case of study.

Follow the paper by PU learning for matrix completion, we have three steps

Step 1. Underline matrix generation

    For the binary predicates, we generate the underlined matrix M

    For complex rule based predicates, we populate the underlined matrix N

Step 2. Binary Quantization

    Then we quantize the random matrics by bernoulli sampling into Y

Step 3. Observed training data

    For Y_ij = 1, we have some observed prob rho so that A_ij = 1 with prob rho
"""

import json
import os
import pickle
import random

import numpy as np


def get_bipred_matrix(size, scale=1):
    print("generating bipred matrix", size, scale)
    M = np.random.rand(size, size) * scale
    return M


def get_multihop_pred_matrix(*multi_hop_M):
    print("generating multi_hop matrix", len(multi_hop_M))
    assert len(multi_hop_M) > 1
    N = multi_hop_M[0].copy()
    size = N.shape[0]
    print('><'*10)
    for _M in multi_hop_M[1:]:
        print("N=", N)
        print("_M=", _M)
        N = N.reshape((size, size, 1))
        _M = _M.copy().reshape((1, size, size))
        conj = np.minimum(N, _M)
        print("conj=", conj)
        N = np.max(conj, axis=-2, keepdims=False)
    return N


def get_quantized_binary_matrix(M):
    M = M / max(1, np.max(M))  # normalization
    Y = np.random.binomial(n=1, p=M)
    assert Y.shape == M.shape
    return Y


def get_observed_matrix(M, rho):
    mask = np.random.rand(*M.shape) < rho
    return M * mask


def case_generation(size=100,
                    num_bipred=5,
                    bi_scale=1,
                    # num_chains[i] = j i's order
                    num_hop_chain=[1, 1, 1, 1, 1],
                    store_path=None
                    ):
    node_dict = {i: str(i) for i in range(size)}
    bipred_dict = {f"bi_pred:{i}": get_bipred_matrix(size, scale=bi_scale)
                   for i in range(num_bipred)}

    multipred_dict = {}
    for num_hops, num_chains in enumerate(num_hop_chain):
        for i in range(num_chains):
            key = f"multihop_pred:{num_hops}"
            bipred_keys = random.choices(list(bipred_dict.keys()), k=num_hops)
            key += "::" + "->".join(bipred_keys)
            if key in multipred_dict:
                continue
            N = get_multihop_pred_matrix(*[bipred_dict[k] for k in bipred_keys])
            multipred_dict[key] = N
    
    print("binary quantization")
    birel_dict = {
        k: get_quantized_binary_matrix(bipred_dict[k])
        for k in bipred_dict
    }

    print("multi-hop quantization")
    multirel_dict = {
        k: get_quantized_binary_matrix(multipred_dict[k])
        for k in multipred_dict
    }

    relmat_dict = {**birel_dict, **multirel_dict}

    # birel_owa_dict = {
    #     k: get_observed_matrix(birel_dict[k], rho=rho_bipred)
    #     for k in birel_dict
    # }

    # multirel_owa_dict = {
    #     k: get_observed_matrix(multirel_dict[k], rho_chains)
    #     for k in multirel_dict
    # }

    # rel_owa_dict = {**birel_owa_dict, **multirel_owa_dict}

    triples = []
    rel_dict = {}
    for i, key in enumerate(relmat_dict):
        rel_dict[i] = key
        head, tail = np.nonzero(relmat_dict[key])
        print("sparse ratio for the specific relation", key, len(head)/size**2)
        for h, t in zip(head, tail):
            triples.append((h, i, t))

    intermediate = {
        'bipred_dict': birel_dict,
        'multipred_dict': multirel_dict,
        'birel_dict': birel_dict,
        'multirel_dict': multirel_dict,
    }

    meta = {
        'num_bipred': num_bipred,
        'num_hop_chain': num_hop_chain,
    }
    for key in relmat_dict:
        meta[f'count:rel={key}'] = np.count_nonzero(relmat_dict[key])

    if store_path is not None:
        store_dataset(store_path, triples, node_dict, rel_dict, intermediate, meta)

    return triples, rel_dict, node_dict, intermediate, meta


def store_dataset(target_folder, triples, node_dict, rel_dict, intermediate, meta):
    if os.path.exists(target_folder):
        target_folder += '1'
    os.makedirs(target_folder, exist_ok=False)
    random.shuffle(triples)

    node_index_path = os.path.join(target_folder, 'map_entity_id_to_text.tsv')
    rel_index_path = os.path.join(target_folder, 'map_relation_id_to_text.tsv')
    intermediate_pickle_path = os.path.join(target_folder, 'intermediate.pickle')
    meta_json_path = os.path.join(target_folder, 'meta.json')
    train_edges_path = os.path.join(target_folder, 'edges_as_id_train.tsv')
    valid_edges_path = os.path.join(target_folder, 'edges_as_id_valid.tsv')
    test_edges_path = os.path.join(target_folder, 'edges_as_id_test.tsv')

    with open(node_index_path, 'wt') as f:
        for i, name in node_dict.items():
            f.write(f"{i}\t{name}\n")

    with open(rel_index_path, 'wt') as f:
        for i, name in rel_dict.items():
            f.write(f"{i}\t{name}\n")

    with open(intermediate_pickle_path, 'wb') as f:
        pickle.dump(intermediate, f)

    with open(meta_json_path, 'wt') as f:
        json.dump(meta, f, indent=2)

    size = len(triples)
    train_size = int(size * 0.8)
    valid_test_size = size - train_size
    valid_size = int(valid_test_size * 0.5)

    train_triples = triples[: train_size]
    valid_triples = triples[train_size: train_size + valid_size]
    test_triples = triples[train_size + valid_size :]

    print('num_train_triples', len(train_triples))
    with open(train_edges_path, 'wt') as f:
        for h, r, t in train_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    print('num_valid_triples', len(valid_triples))
    with open(valid_edges_path, 'wt') as f:
        for h, r, t in valid_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    print('num_test_triples', len(test_triples))
    with open(test_edges_path, 'wt') as f:
        for h, r, t in test_triples:
            f.write(f"{h}\t{r}\t{t}\n")


if __name__ == "__main__":
    case_generation(
        size=500,
        num_bipred=5,
        bi_scale=0.1,
        num_hop_chain=[0, 0, 2, 4, 8],
        store_path="data/multihop")
    # case_generation(
    #     size=2,
    #     num_bipred=2,
    #     bi_scale=0.1,
    #     num_hop_chain=[0, 0, 2, 4, 8],
    #     store_path=None)