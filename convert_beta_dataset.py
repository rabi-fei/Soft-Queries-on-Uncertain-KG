import os
import os.path as osp
from pprint import pprint
import pickle
from typing import Dict

from src.language import fol
from src.language import parse_lstr
from src.structure.knowledge_graph_index import KGIndex
from src.structure.knowledge_graph import KnowledgeGraph

beta_types_list = [
    ('e', ('r',)),
    ('e', ('r', 'r')),
    ('e', ('r', 'r', 'r')),
    (('e', ('r',)), ('e', ('r',))),
    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
    ((('e', ('r',)), ('e', ('r',))), ('r',)),
    (('e', ('r', 'r')), ('e', ('r',))),
    (('e', ('r',)), ('e', ('r', 'n'))),
    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))),
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)),
    (('e', ('r', 'r')), ('e', ('r', 'n'))),
    (('e', ('r', 'r', 'n')), ('e', ('r',))),
    (('e', ('r',)), ('e', ('r',)), ('u',)),
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)),
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)),
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r'))]

labeled_beta_types_list = [
    ('s1', ('r1',)),
    ('s1', ('r1', 'r2')),
    ('s1', ('r1', 'r2', 'r3')),
    (('s1', ('r1',)), ('s2', ('r2',))),
    (('s1', ('r1',)), ('s2', ('r2',)), ('s3', ('r3',))),
    ((('s1', ('r1',)), ('s2', ('r2',))), ('r3',)),
    (('s1', ('r1', 'r2')), ('s2', ('r3',))),
    (('s1', ('r1',)), ('s2', ('r2', 'n'))),
    (('s1', ('r1',)), ('s2', ('r2',)), ('s3', ('r3', 'n'))),
    ((('s1', ('r1',)), ('s2', ('r2', 'n'))), ('r3',)),
    (('s1', ('r1', 'r2')), ('s2', ('r3', 'n'))),
    (('s1', ('r1', 'r2', 'n')), ('s2', ('r3',))),
    (('s1', ('r1',)), ('s2', ('r2',)), ('u',)),
    ((('s1', ('r1',)), ('s2', ('r2',)), ('u',)), ('r3',)),
    ((('s1', ('r1', 'n')), ('s2', ('r2', 'n'))), ('n',)),
    ((('s1', ('r1', 'n')), ('s2', ('r2', 'n'))), ('n', 'r3'))
    ]

beta_lstr_list = [
    "r1(s1,f)",  # 1p
    "r1(s1,e1)&r2(e1,f)",  # 2p
    "r1(s1,e1)&r2(e1,e2)&r3(e2,f)",  # 3p
    "r1(s1,f)&r2(s2,f)",  # 2i
    "r1(s1,f)&r2(s2,f)&r3(s3,f)",  # 3i
    "r1(s1,e1)&r2(s2,e1)&r3(e1,f)",
    "r1(s1,e1)&r2(e1,f)&r3(s2,f)",
    "r1(s1,f)&!r2(s2,f)",
    "r1(s1,f)&r2(s2,f)&!r3(s3,f)",
    "r1(s1,e1)&!r2(s2,e1)&r3(e1,f)",
    "r1(s1,e1)&r2(e1,f)&!r3(s2,f)",
    "r1(s1,e1)&!r2(e1,f)&r3(s2,f)",
    "r1(s1,f)|r2(s2,f)",
    "r1(s1,e1)|r2(s2,e1))&r3(e1,f)",
    "!(!r1(s1,f)&!r2(s2,f))",
    "!(!r1(s1,e1)|r2(s2,e1))&r3(e1,f)",
]

def beta_type_to_ldict():
    pass

def align_entities_relations(labeled_beta_type, beta_sample) -> Dict:
    d = {}
    def _align(labeled_beta_type, beta_sample):
        for sub_type, sub_sample in zip(labeled_beta_type, beta_sample):
            if not isinstance(sub_type, str):
                _align(sub_type, sub_sample)
            else:
                if sub_type[0] in 'sr':
                    d[sub_type] = sub_sample
    _align(labeled_beta_type, beta_sample)
    return d

def convert_beta_folder(beta_folder, output_folder):
    """
    Convert the folder of beta dataset into the output data
    the structure of the beta folder
        indices
        - ent2id.pkl
        - id2ent.pkl
        - id2rel.pkl
        - rel2id.pkl

        knowledge graphs and queries
        {test/valid/train}.txt

        train_queries
        train-queries.pkl
        train-answers.pkl

        evaluation queries
        {test/valid}-queries.pkl
        {test/valid}-easy_answers.pkl
        {test/valid}-hard_answers.pkl

    the structure of output folder
        kgindex.json, aggregrates the indices files into one


    """

    # build knowledge graph indices
    os.makedirs(output_folder, exist_ok=True)

    kgidx = KGIndex()
    with open(osp.join(beta_folder, 'ent2id.pkl'), 'rb') as f:
        entity_to_id = pickle.load(f)
    eids = []
    for name, eid in entity_to_id.items():
        kgidx.register_entity(name, eid)
        eids.append(eid)
    assert max(eids) - min(eids) + 1 == len(kgidx.map_entity_name_to_id)

    with open(osp.join(beta_folder, 'rel2id.pkl'), 'rb') as f:
        relation_to_id = pickle.load(f)

    rids = []
    for name, rid in relation_to_id.items():
        kgidx.register_relation(name, rid)
        rids.append(rid)
    assert max(rids) - min(rids) + 1 == len(kgidx.map_relation_name_to_id)

    kgidx.dump(osp.join(output_folder, 'kgindex.json'))
    kgidx = KGIndex.load(osp.join(output_folder, 'kgindex.json'))

    # train knowledge graphs
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(beta_folder, 'train.txt'),
        kgindex=kgidx)

    train_kg.dump(osp.join(output_folder, 'train_kg.tsv'))

    valid_kg = KnowledgeGraph.create(
        triple_files=[osp.join(beta_folder, 'train.txt'),
                      osp.join(beta_folder, 'valid.txt')],
        kgindex=kgidx)

    valid_kg.dump(osp.join(output_folder, 'valid_kg.tsv'))

    test_kg = KnowledgeGraph.create(
        triple_files=[osp.join(beta_folder, 'train.txt'),
                      osp.join(beta_folder, 'valid.txt'),
                      osp.join(beta_folder, 'test.txt')],
        kgindex=kgidx)

    test_kg.dump(osp.join(output_folder, 'test_kg.tsv'))

    # knowledge graph queries

    # train queries
    with open(osp.join(beta_folder, "train-queries.pkl"), 'rb') as f:
        train_query = pickle.load(f)

    for beta_type, labeled_beta_type, lstr in zip(
        beta_types_list, labeled_beta_types_list, beta_lstr_list):

        samples = list(train_query[beta_type])[:3]
        lformula = parse_lstr(lstr)
        folf = fol.FirstOrderFormula(lformula)
        print(folf.formula.to_lstr())
        for sample in samples:
            d = align_entities_relations(labeled_beta_type, sample)
            folf.append_relation_and_symbols(d)
        print(folf.formula)

if __name__ == "__main__":
    beta_folder = "/Users/zihao/Project/FirstOrderQueryEstimation/data/FB15k-237-betae"
    output_folder = "./data/FB15k-237-betae"

    convert_beta_folder(beta_folder, output_folder)