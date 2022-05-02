"""
A file maintains reasoning machine
"""
from typing import Dict, List

import torch

from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.language.fol import FirstOrderFormula

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

    def _single_formula_reasoning(self, formula: FirstOrderFormula):
        """
        reasoning the first order formula
        Input args:
            formula: a first order formula with batched instantiation
        Output args:
            truth_values: a tensor for all optimized truth value
        """
        # first initialize all variables
        evars, uvars, fvars = self.initialize_variable_embeddings(formula)
        optimizer_class = getattr(torch.optim, self.reasoinng_optimizer)
        EF_opt = optimizer_class(evars + fvars)
        U_opt = optimizer_class(uvars)

        for i in range(self.reasoning_steps):
            # maximize the truth value with repect to evars and fvars
            EF_opt.zero_grad()
            efloss = - formula.evaluate_truth_values().mean()
            efloss.backward()
            EF_opt.step()

            if uvars:
                # minimize the truth value with respect to uvars
                U_opt.zero_grad()
                uloss = formula.evaluate_truth_values().mean()
                uloss.backward()
                U_opt.step()

        truth_values = formula.evaluate_truth_values()
        return truth_values, fvars

    def initialize_variable_embeddings(self, formula: FirstOrderFormula):
        """
        Input args:
        Return args:
            evars: list of existential variables
            uvars: list of universal variables
            fvars: list of free variables
        """
        def check_all_var_initialized():
            for k, v in formula.existential_variable_dict.items():
                if v.not_initialized_at_all:
                    return False
            for k, v in formula.universal_variable_dict.items():
                if v.not_initialized_at_all:
                    return False
            for k, v in formula.free_variable_dict.items():
                if v.not_initialized_at_all:
                    return False
            return True

        while check_all_var_initialized():
            for k, pred in formula.predicate_dict.items():
                if (pred.term1.not_initialized_at_all
                    and not pred.term2.not_initialized_at_all):
                    # initialize the term2
                    if not pred.term2.has_embedding:
                        assert pred.term2.has_proposal
                        pred.term2.update_embeddingby_proposals()
                    # estimate term 1
                    term1_proposal = pred.predict_term1()
                    pred.term1.append_proposal(term1_proposal)

                elif (not pred.term1.not_initialized_at_all
                      and pred.term2.not_initialized_at_all):
                    # initialize the term1
                    if not pred.term1.has_embedding:
                        assert pred.term1.has_proposal
                        pred.term1.update_embeddingby_proposals()

                    term2_proposal = pred.predict_term2()
                    pred.term2.append_proposal(term2_proposal)

        return (list(formula.existential_variable_dict.values()),
                list(formula.universal_variable_dict.values()),
                list(formula.free_variable_dict.values()))


    def reasoning(self, fof_list: List[FirstOrderFormula]):
        # then it comes into a batched formula list
        for formula in fof_list:
            # for each batch formula,
            truth_values, fvars = self._single_formula_reasoning(formula)

    def inference(self):
        pass
