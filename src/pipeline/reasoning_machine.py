"""
A file maintains reasoning machine
"""
from tkinter import W
from typing import Dict, List

import torch

from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.language.fof import FirstOrderFormula, get_head_embed_from_formula, get_tail_embed_from_formula

def gather_formula(formula_list) -> Dict:
    pass


class GradientReasoningMachine:
    def __init__(self,
                 reasoning_rate,
                 reasoning_steps,
                 reasoning_optimizer,
                 nbp: NeuralBinaryPredicate,
                 tnorm):
        self.reasoning_rate = reasoning_rate
        self.reasoning_steps = reasoning_steps
        self.reasoinng_optimizer = reasoning_optimizer
        self.nbp = nbp
        self.tnorm = tnorm

    def _reason_single_formula(self, formula: FirstOrderFormula):
        """
        reasoning the first order formula
        Input args:
            formula: a first order formula with batched instantiation
        Output args:
            truth_values: a tensor for all optimized truth value
        """
        # first initialize all variables
        self.initialize_variable_embeddings(formula)
        evar_local_emb = list(filter(
            lambda x: x is not None,
            [formula.get_var_local_embedding(k)
             for k in formula.existential_variable_dict]))

        fvar_local_emb = list(filter(
            lambda x: x is not None,
            [formula.get_var_local_embedding(k)
            for k in formula.free_variable_dict]))
        efvar_local_emb = evar_local_emb + fvar_local_emb

        uvar_local_emb = list(filter(
            lambda x: x is not None,
            [formula.get_var_local_embedding(k)
             for k in formula.universal_variable_dict]))

        optimizer_class = getattr(torch.optim, self.reasoinng_optimizer)

        if efvar_local_emb:
            EF_opt = optimizer_class(efvar_local_emb, self.reasoning_rate)

        if uvar_local_emb:
            U_opt = optimizer_class(uvar_local_emb, self.reasoning_rate)
        eflosses = []
        ulosses = []
        for i in range(self.reasoning_steps):
            # maximize the truth value with repect to evars and fvars
            if efvar_local_emb:
                EF_opt.zero_grad()
                ntv = - formula.evaluate_truth_values(
                    self.tnorm, self.nbp, self.nbp.margin).mean()
                reg = sum([torch.sum(torch.abs(emb) ** 3, -1)
                               for emb in efvar_local_emb]).mean() * 0.05
                efloss = ntv + reg
                eflosses.append((ntv.item(), reg.item(), efloss.item()))
                efloss.backward()
                EF_opt.step()
            else:
                efloss = None

            if uvar_local_emb:
                # minimize the truth value with respect to uvars
                U_opt.zero_grad()
                uloss = formula.evaluate_truth_values(
                    self.tnorm, self.nbp, self.nbp.margin).mean()
                ulosses.append(ulosses.item())
                uloss.backward()
                U_opt.step()
            else:
                uloss = None

        truth_values = formula.evaluate_truth_values(
            self.tnorm, self.nbp, self.nbp.margin)
        fvar_local_emb_dict = {
                k: formula.get_var_local_embedding(k) for k in formula.free_variable_dict}
        return {'tv': truth_values,
                'fvar_local_emb_dict': fvar_local_emb_dict}


    def initialize_variable_embeddings(self, formula: FirstOrderFormula):
        """
        Input args:
        Return args:
            evars: list of existential variables
            uvars: list of universal variables
            fvars: list of free variables
        """
        def check_all_var_initialized():
            return all(formula.term_initialized(term_name)
                       for term_name in formula.term_dict)

        while not check_all_var_initialized():
            for rel_name, pred in formula.predicate_dict.items():
                head_name, tail_name = pred.head.name, pred.tail.name

                if formula.term_initialized(head_name) and not formula.term_initialized(tail_name):
                    head_emb = get_head_embed_from_formula(
                        self.nbp, formula, head_name
                    )
                    rel_emb = self.nbp.get_relation_emb(
                        formula.pred_grounded_relation_id_dict[rel_name]
                    )

                    tail_emb = self.nbp.estimate_tail_emb(head_emb, rel_emb)
                    # formula.set_var_local_embedding(tail_name, tail_emb)
                    formula.init_var_local_embedding(tail_name, tail_emb)

                elif not formula.term_initialized(head_name) and formula.term_initialized(tail_name):
                    tail_emb = get_tail_embed_from_formula(
                        self.nbp, formula, tail_name
                    )
                    rel_emb = self.nbp.get_relation_emb(
                        formula.pred_grounded_relation_id_dict[rel_name]
                    )

                    head_emb = self.nbp.estimate_head_emb(tail_emb, rel_emb)
                    # formula.set_var_local_embedding(head_name, head_emb)

                    formula.init_var_local_embedding(head_name, head_emb)

                else:
                    continue
        return

    def reasoning(self, fof_list: List[FirstOrderFormula]):
        # then it comes into a batched formula list
        return [self._reason_single_formula(fof) for fof in fof_list]
