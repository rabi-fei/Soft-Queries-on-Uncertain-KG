"""
A file maintains reasoning machine
"""
from typing import Dict
from language.fol import FirstOrderFormula

import torch

def gather_formula(formula_list) -> Dict:
    pass


class GradientReasoningMachine:
    def __init__(self,
                 reasoning_rate,
                 reasoning_steps,
                 reasoning_optimizer,
                 learning_rate,
                 learning_steps,
                 learning_optimizer):
        self.reasoning_rate = reasoning_rate
        self.reasoning_steps = reasoning_steps
        self.reasoinng_optimizer = reasoning_optimizer
        self.learning_rate = learning_rate
        self.learning_steps = learning_steps
        self.learning_optimizer = learning_optimizer

    def _single_formula_reasoning(self, formula: FirstOrderFormula):
        """
        reasoning the first order formula
        Input args:
            formula: a first order formula with batched instantiation
        Output args:
            truth_values: a tensor for all optimized truth value
        """
        # first initialize all variables
        evars, uvars, fvars = formula.initialize_variable_embeddings()
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

    def reasoning(self, formula_list):
        # gather the formulas as their string representation
        formula_dict = gather_formula(formula_list)

        # then it comes into a batched formula list
        for k, formula in formula_dict.items():
            # for each batch formula,
            truth_values, fvars = self._single_formula_reasoning(formula)

    def inference(self):
        pass
