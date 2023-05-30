
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from src.language.tnorm import GodelTNorm, ProductTNorm, Tnorm
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph, kg2matrix
from src.structure.knowledge_graph_index import KGIndex
from src.utils.data_util import RaggedBatch
from lifted_embedding_estimation_with_truth_value import name2lstr, newlstr2name, index2newlstr, index2EFOX_minimal
from src.language.grammar import parse_lstr_to_disjunctive_formula
from src.language.fof import Disjunction, ConjunctiveFormula, DisjunctiveFormula
from src.utils.data import (QueryAnsweringMixDataLoader, QueryAnsweringSeqDataLoader,
                            QueryAnsweringSeqDataLoader_v2,
                            TrainRandomSentencePairDataLoader)



train_queries = list(name2lstr.values())
query_2in = 'r1(s1,f)&!r2(s2,f)'
query_2i = 'r1(s1,f)&r2(s2,f)'
parser = argparse.ArgumentParser()
#parser.add_argument("--output_name", type=str, default='new-qaa')
parser.add_argument("--double_check", type=float, default=-1)
parser.add_argument("--output_folder", type=str, default='data/NELL-EFOX')
parser.add_argument("--data_folder", type=str, default='data/NELL-EFOX')
parser.add_argument("--num_positive", type=int, default=1000)
parser.add_argument("--num_negative", type=int, default=500)
parser.add_argument('--mode', choices=['train', 'valid', 'test'], default='test')
parser.add_argument("--meaningful_negation", type=bool, default=False)
parser.add_argument("--negation_tolerance", type=int, default=1)
parser.add_argument("--ncpus", type=int, default=10)
parser.add_argument("--skip_exist", type=bool, default=True)
parser.add_argument("--sample_formula_scope", type=str, default='EFOX', choices=['real_EFO1', 'EFOX_minimal', 'EFOX'])
parser.add_argument("--sample_formula_list", type=list, default=list(range(0, 1)))
parser.add_argument("--start_index", type=int, default=555)
parser.add_argument("--end_index", type=int, default=555)
parser.add_argument("--max_ans", type=int, default=100)
