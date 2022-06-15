"""
A file maintains reasoning machine
"""
from typing import Dict, List
import math

import torch

from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.language.fof import FirstOrderFormula

def gather_formula(formula_list) -> Dict:
    pass

class GradientReasoningMachineEFO:
    def __init__(self,
                 reasoning_rate,
                 reasoning_steps,
                 reasoning_optimizer,
                 nbp: NeuralBinaryPredicate,
                 tnorm,
                 sigma=1):
        self.reasoning_rate = reasoning_rate
        self.reasoning_steps = reasoning_steps
        self.reasoinng_optimizer = reasoning_optimizer
        self.nbp = nbp
        self.tnorm = tnorm
        self.sigma = sigma

    def _reason_single_formula(self, formula: FirstOrderFormula, free_var_treatment, fole=False):
        """
        reasoning the first order formula
        Input args:
            formula: a first order formula with batched instantiation
            free_var_treatment:
                - existential: treat the free variable as the existential variable
                - ground1random: ground the free variable into one random answers
                - ground1noisy: ground the free variable into single ngative variables
                - groundfull: ground the answers to a full answer set
            fole: boolean, whether consider free variables as boolean function
        Output args:
            truth_values:
                a tensor for all optimized truth value [batch_size]
                if infer_free, [all_candidate, batch_size]
        """

        assert len(formula.universal_variable_dict) == 0
        # first initialize all variables
        self.initialize_variable_embeddings(formula)
        evar_local_emb = list(filter(
            lambda x: x is not None,
            [formula.get_var_local_embedding(k)
             for k in formula.existential_variable_dict]))

        if free_var_treatment.lower() == 'existential':
            fvar_local_emb = list(filter(
                lambda x: x is not None,
                [formula.get_var_local_embedding(k)
                for k in formula.free_variable_dict]))
            efvar_local_emb = evar_local_emb + fvar_local_emb
        else:
            efvar_local_emb = evar_local_emb

        if len(efvar_local_emb) == 0:
            # this only applies for the cases where no existential variables
            return formula.evaluate_truth_values(self.tnorm, self.nbp, free_var_treatment)


        optimizer_class = getattr(torch.optim, self.reasoinng_optimizer)
        opt = optimizer_class(efvar_local_emb, self.reasoning_rate)
        eflosses = [(-1, -1, -1)]

        if enable_target:
            fvar_target_emb_dict = {}
            for k in formula.free_variable_dict:
                answer_barycenters = []
                for easy_answer in formula.easy_answer_list:
                    ans_bry_emb = torch.mean(self.nbp.get_head_emb(easy_answer[k]),
                                            dim=0, keepdim=True)
                    answer_barycenters.append(ans_bry_emb)
                fvar_target_emb_dict[k] = torch.cat(answer_barycenters, dim=0).detach().clone()

        for i in range(self.reasoning_steps):
            # maximize the truth value with repect to evars and fvars
            if efvar_local_emb:
                tv = formula.evaluate_truth_values(
                    self.tnorm, self.nbp,
                    free_var_treatment=False)

                if enable_target:
                    for k in formula.free_variable_dict:
                        loc_emb = formula.get_var_local_embedding(k)
                        tar_emb = fvar_target_emb_dict[k]
                        dist = torch.sum((loc_emb - tar_emb)**2, -1)
                        eqtv = torch.exp(- dist / self.sigma ** 2)
                        tv = self.tnorm.conjunction(tv, eqtv)

                ntv = - tv.mean()
                reg = 0.05 * sum([
                    torch.mean(
                        torch.sum(
                            torch.abs(self.nbp.regularization(emb)) ** 3, -1)
                            )
                    for emb in efvar_local_emb])

                efloss = ntv + reg
                eflosses.append((ntv.item(), reg.item(), efloss.item()))
                opt.zero_grad()
                efloss.backward()
                opt.step()
            else:
                efloss = None

            if math.fabs(eflosses[-1][-1] - eflosses[-2][-1]) < 1e-9:
                break

        # print(f"continuous search for {formula.lstr()} breaks at step {i}")

        truth_values = formula.evaluate_truth_values(
            self.tnorm, self.nbp,
            free_var_treatment=not infer_free)
        fvar_local_emb_dict = {
                k: formula.get_var_local_embedding(k) for k in formula.free_variable_dict}
        assert truth_values is not None
        return {'tv': truth_values,
                'fvar_local_emb_dict': fvar_local_emb_dict,
                'eflosses': eflosses}


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
                    head_emb = formula.get_tail_embedding(
                        self.nbp, head_name
                    )
                    rel_emb = self.nbp.get_relation_emb(
                        formula.pred_grounded_relation_id_dict[rel_name]
                    )

                    tail_emb = self.nbp.estimate_tail_emb(head_emb, rel_emb)
                    # formula.set_var_local_embedding(tail_name, tail_emb)
                    formula.set_var_local_embedding(tail_name, tail_emb)

                elif not formula.term_initialized(head_name) and formula.term_initialized(tail_name):
                    tail_emb = formula.get_tail_embedding(
                        self.nbp, tail_name
                    )
                    rel_emb = self.nbp.get_relation_emb(
                        formula.pred_grounded_relation_id_dict[rel_name]
                    )

                    head_emb = self.nbp.estimate_head_emb(tail_emb, rel_emb)
                    # formula.set_var_local_embedding(head_name, head_emb)

                    formula.set_var_local_embedding(head_name, head_emb)

                else:
                    continue
        return

        # new implementation, random initialize
        # for term_name in formula.term_dict:
        #     if not formula.term_initialized(term_name):
        #         emb = self.nbp.get_random_entity_embed(formula.num_instances)
        #         formula.init_var_local_embedding(term_name, emb)

    def reasoning(self, fofs: List[FirstOrderFormula], free_var_treatment, fole=False):
        """
        fof_list: list of FirstOrderFormula, also, it can be just a FirstOrderFormula
        free_var_treatment:
            - existential: treat the free variable as the existential variable
            - ground1random: ground the free variable into one random answers
            - ground1noisy: ground the free variable into single ngative variables
            - groundfull: ground the answers to a full answer set
        fole: boolean, whether consider free variables as boolean function
        """
        # then it comes into a batched formula list
        if isinstance(fofs, list):
            return [self._reason_single_formula(fof, free_var_treatment, fole) for fof in fofs]
        else:
            return self._reason_single_formula(fofs, free_var_treatment, fole)
