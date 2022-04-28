import os.path as osp
from pprint import pprint
import pickle

from src.language import fol
from src.language import parse_lstr

beta_types = [
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

beta_types_labeled = [
    ('l1', ('r1',)),
    ('l1', ('r1', 'r2')),
    ('l1', ('r1', 'r2', 'r3')),
    (('l1', ('r1',)), ('l2', ('r2',))),
    (('l1', ('r1',)), ('l2', ('r2',)), ('l3', ('r3',))),
    ((('l1', ('r1',)), ('l2', ('r2',))), ('r3',)),
    (('l1', ('r1', 'r2')), ('l2', ('r3',))),
    (('l1', ('r1',)), ('l2', ('r2', 'n'))),
    (('l1', ('r1',)), ('l2', ('r2',)), ('l3', ('r3', 'n'))),
    ((('l1', ('r1',)), ('l2', ('r2', 'n'))), ('r3',)),
    (('l1', ('r1', 'r2')), ('l2', ('r3', 'n'))),
    (('l1', ('r1', 'r2', 'n')), ('l2', ('r3',))),
    (('l1', ('r1',)), ('l2', ('r2',)), ('u',)),
    ((('l1', ('r1',)), ('l2', ('r2',)), ('u',)), ('r3',)),
    ((('l1', ('r1', 'n')), ('l2', ('r2', 'n'))), ('n',)),
    ((('l1', ('r1', 'n')), ('l2', ('r2', 'n'))), ('n', 'r3'))
    ]

beta_lstr = [
    "r1(l1,f)",  # 1p
    "r1(l1,e1)&r2(e1,f)",  # 2p
    "r1(l1,e1)&r2(e1,e2)&r3(e2,f)",  # 3p
    "r1(l1,f)&r2(l2,f)",  # 2i
    "r1(l1,f)&r2(l2,f)&r3(l3,f)",  # 3i
    "r1(l1,e1)&r2(l2,e1)&r3(e1,f)",
    "r1(l1,e1)&r2(e1,f)&r3(l2,f)",
    "r1(l1,f)&!r2(l2,f)",
    "r1(l1,f)&r2(l2,f)&!r3(l3,f)",
    "r1(l1,e1)&!r2(l2,e1)&r3(e1,f)",
    "r1(l1,e1)&r2(e1,f)&!r3(l2,f)",
    "r1(l1,e1)&!r2(e1,f)&r3(l2,f)",
    "r1(l1,f)|r2(l2,f)",
    "r1(l1,e1)|r2(l2,e1))&r3(e1,f)",
    "!(!r1(l1,f)&!r2(l2,f))",
    "!(!r1(l1,e1)|r2(l2,e1))&r3(e1,f)",
]

def beta_type_to_ldict():
    pass

def convert_beta_folder(beta_folder, output_folder):
    """
    Convert the folder of beta dataset into the output data
    """
    pass

if __name__ == "__main__":
    beta_folder = "/Users/zihao/Project/FirstOrderQueryEstimation/data/FB15k-237-betae"
    with open(osp.join(beta_folder, "test-queries.pkl"), 'rb') as f:
        test_query = pickle.load(f)
    for beta_type in beta_types:
        print(beta_type)
        pprint(list(test_query[beta_type])[:3])

    for labeled_data_type, lstr in zip(beta_types_labeled, beta_lstr):
        print(labeled_data_type)
        print(lstr)
        lobject = parse_lstr(lstr)
        print(lobject)
