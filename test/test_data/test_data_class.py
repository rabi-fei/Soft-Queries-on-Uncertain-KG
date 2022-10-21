import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from src.language.tnorm import GodelTNorm, ProductTNorm, Tnorm
from src.pipeline.reasoning_machine import (DeepsetEFOReasoner,
                                            GradientEFOReasoner, Reasoner,
                                            GNNEFOReasonerComplEx,
                                            RelationalDeepSet,
                                            VanillaGNNLayerComplEx,
                                            LogicalGNNLayerComplEx)
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.utils.data import (QueryAnsweringMixDataLoader,
                            QueryAnsweringSeqDataLoader, RaggedBatch,
                            TrainRandomSentencePairDataLoader)
from lifted_embedding_estimation_with_truth_value import name2lstr
from src.language.grammar import parse_lstr_to_lformula, parse_lstr_to_lformula_v2, DNF_Transformation, concate_iu_chains
from src.language.fof import Disjunction, ConjunctiveFormula, DisjunctiveFormula

data_folder = 'data/FB15k-237-betae'
train_queries = list(name2lstr.values())
batch_size = 10

for name, lstr in name2lstr.items():
    formula = parse_lstr_to_lformula(lstr)
    formula_v2 = parse_lstr_to_lformula_v2(lstr)
    DNF_formula = DNF_Transformation(formula_v2)
    concate_formula = concate_iu_chains(formula_v2)
    print(formula.lstr(), formula_v2.lstr(), DNF_formula.lstr(), concate_formula.lstr())
    formula_v2_check = parse_lstr_to_lformula_v2(formula_v2.lstr())
    assert formula_v2_check.lstr() == formula_v2.lstr()


lstr2name = {}
for name, lstr in name2lstr.items():
    formula = parse_lstr_to_lformula_v2(lstr)
    formula = concate_iu_chains(formula)
    if isinstance(formula, Disjunction):
        formula_list = formula.formulas
    else:
        formula_list = [formula]
    conjunctive_formulas_list = [ConjunctiveFormula(formula) for formula in formula_list]
    fof = DisjunctiveFormula(conjunctive_formulas_list)
    print(fof.lstr)
    lstr2name[fof.lstr] = name
print(lstr2name)

'''
train_dataloader = QueryAnsweringSeqDataLoader(
    osp.join(data_folder, 'valid-qaa.json'),
    # size_limit=args.batch_size * 1,
    target_lstr=train_queries,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
fof_list = train_dataloader.get_fof_list()
t = tqdm.tqdm(enumerate(fof_list), total=len(fof_list))
for ifof, fof in t:
    print(fof)
'''