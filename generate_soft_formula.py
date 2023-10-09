import sys
import argparse
import json
import logging
import os
import os.path as osp
import random
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from src.language.grammar import parse_lstr_to_disjunctive_formula
random.seed(42)

level2per = {"zero": 0, "low" : 25, "normal": 50}

def recursion_update_b(formula):
    if formula.op == "pred":
        beta = float(f"{math.ceil(random.random() * 10) / 10:.1f}")
        while beta == formula.beta:
            p = random.random()
            beta = float(f"{math.ceil(p * 10) / 10:.1f}")
        formula.beta = beta
        return None
    sub_formula_list = [formula.formula] if formula.op == "neg" else formula.formulas
    for sub_formula in sub_formula_list:
        recursion_update_b(sub_formula)

def recursion_update_a(formula, level):

    if formula.op == "pred":
        formula.alpha = f"{level2per[level]}%"
        return None
    sub_formula_list = [formula.formula] if formula.op == "neg" else formula.formulas
    for sub_formula in sub_formula_list:
        recursion_update_a(sub_formula, level)

require_level = "normal"
new_rows = []
formula_scope = pd.read_csv(osp.join('data', 'DNF_train_soft_EFO1.csv'))
for i, row in formula_scope.iterrows():
    given_lstr = row.formula
    p = random.random()
    fof = parse_lstr_to_disjunctive_formula(given_lstr)
#    new_rows.append(row)
    for conj_formula in fof.formula_list:
        if conj_formula.formula.op == 'pred':
            recursion_update_a(conj_formula.formula, require_level)
        else:
            recursion_update_a(conj_formula.formula, require_level)

    part_lstr = fof.lstr
    row.formula = part_lstr
    new_rows.append(row)
#    for conj_formula in fof.formula_list:
#        recursion_update_b(conj_formula.formula)
#    full_lstr = fof.lstr
#    row.formula = full_lstr
#    new_rows.append(row)

df_new = pd.DataFrame(new_rows, columns=formula_scope.columns)
df_new = df_new.reset_index(drop=True)
df_new.to_csv(f"data/DNF_train_{require_level}_soft_EFO1.csv")