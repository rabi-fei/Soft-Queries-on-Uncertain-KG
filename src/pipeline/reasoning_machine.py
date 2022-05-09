"""
A file maintains reasoning machine
"""
from typing import Dict, List

import torch
from src.language.tnorm import ProductTNorm

from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.language.fol import FirstOrderFormula, Term

def gather_formula(formula_list) -> Dict:
    pass


class GradientReasoningMachine:
    def __init__(self,
                 reasoning_rate,
                 reasoning_steps,
                 reasoning_optimizer,
                 nbp: NeuralBinaryPredicate):
        self.reasoning_rate = reasoning_rate
        self.reasoning_steps = reasoning_steps
        self.reasoinng_optimizer = reasoning_optimizer
        self.nbp = nbp

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
        evar_local_emb = [formula.get_var_local_embedding(k)
            for k in formula.existential_variable_dict]
        fvar_local_emb = [formula.get_var_local_embedding(k)
            for k in formula.free_variable_dict]
        uvar_local_emb = [formula.get_var_local_embedding(k)
            for k in formula.universal_variable_dict]

        optimizer_class = getattr(torch.optim, self.reasoinng_optimizer)
        EF_opt = optimizer_class(evar_local_emb + fvar_local_emb)
        if uvar_local_emb:
            U_opt = optimizer_class(uvar_local_emb)

        for i in range(self.reasoning_steps):
            # maximize the truth value with repect to evars and fvars
            EF_opt.zero_grad()
            efloss = - formula.evaluate_truth_values(ProductTNorm, self.nbp, self.nbp.margin).mean()
            efloss.backward()
            EF_opt.step()


            if uvar_local_emb:
                # minimize the truth value with respect to uvars
                U_opt.zero_grad()
                uloss = formula.evaluate_truth_values().mean()
                uloss.backward()
                U_opt.step()

        truth_values = formula.evaluate_truth_values(
            ProductTNorm, self.nbp, self.nbp.margin)
        return truth_values, fvar_local_emb

    def _train_single_formula(self, formula: FirstOrderFormula):
        self.initialize_variable_embeddings(formula)
        evar_local_emb = [formula.get_var_local_embedding(k)
            for k in formula.existential_variable_dict]
        fvar_local_emb = [formula.get_var_local_embedding(k)
            for k in formula.free_variable_dict]
        uvar_local_emb = [formula.get_var_local_embedding(k)
            for k in formula.universal_variable_dict]



    def initialize_variable_embeddings(self, formula: FirstOrderFormula):
        """
        Input args:
        Return args:
            evars: list of existential variables
            uvars: list of universal variables
            fvars: list of free variables
        """
        def check_all_var_initialized():
            for emb in formula.variable_local_embedding_dict.values():
                if emb is None:
                    return False
            return True

        while not check_all_var_initialized():
            for k, pred in formula.predicate_dict.items():
                term1, term2 = pred.term1, pred.term2

                if term1.is_symbol or term2.is_symbol:
                    if term1.is_symbol and not term2.is_symbol:
                        formula.set_var_local_embedding(
                            term2.name,
                            self.nbp.estimate_tail_emb(
                                self.nbp.get_entity_emb(term1.entity_id_list),
                                self.nbp.get_relation_emb(pred.relation_id_list)
                            )
                        )
                    elif not term1.is_symbol and term2.is_symbol:
                        formula.set_var_local_embedding(
                            term1.name,
                            self.nbp.estimate_head_emb(
                                self.nbp.get_entity_emb(term2.entity_id_list),
                                self.nbp.get_relation_emb(pred.relation_id_list)
                            )
                        )
                    else:
                        break
                else:
                    tn1, tn2 = term1.name, term2.name
                    if (formula.has_var_local_embedding(tn1)
                        and not formula.has_var_local_embedding(tn2)):
                        formula.set_var_local_embedding(
                            tn2,
                            self.nbp.estimate_tail_emb(
                                formula.get_var_local_embedding(tn1),
                                self.nbp.get_relation_emb(pred.relation_id_list)
                            )
                        )
                    elif (not formula.has_var_local_embedding(tn1)
                          and formula.has_var_local_embedding(tn2)):
                        formula.set_var_local_embedding(
                            tn1,
                            self.nbp.estimate_head_emb(
                                formula.get_var_local_embedding(tn2),
                                self.nbp.get_relation_emb(pred.relation_id_list)
                            )
                        )
                    else:
                        break
        return

    def inference_with_reasoning(self, fof_list: List[FirstOrderFormula]):
        # then it comes into a batched formula list
        for formula in fof_list:
            # for each batch formula,
            truth_values, fvar_local_emb = self._reason_single_formula(formula)

        return {'tv': truth_values, 'fvar_local_emb': fvar_local_emb}

    def train_forward_with_reasoning(self, fof_list: List[FirstOrderFormula]):
        for formula in fof_list:
            self._train_single_formula(formula)

        return {'pos_mean_tv': None,
                'neg_mean_tv': None}
