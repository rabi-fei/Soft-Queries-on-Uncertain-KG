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


class GradientReasoningMachine:
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

    def _reason_single_formula(self, formula: FirstOrderFormula, infer_free, enable_target=False):
        """
        reasoning the first order formula
        Input args:
            formula: a first order formula with batched instantiation
            infer_free: whether to output the free embedding`
                if true, output the embeddings of free variables
                otherwise, output the None
        Output args:
            truth_values:
                a tensor for all optimized truth value [batch_size]
                if infer_free, [all_candidate, batch_size]
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

        if enable_target:
            target_emb_dict = {}
            for f in formula.free_variable_dict:
                target_emb_dict[f] = torch.cat(
                    [torch.mean(self.nbp.get_head_emb(easy_answer[f]), dim=0, keepdim=True)
                     for easy_answer in formula.easy_answer_list]
                ).detach()

        optimizer_class = getattr(torch.optim, self.reasoinng_optimizer)

        if efvar_local_emb:
            EF_opt = optimizer_class(efvar_local_emb, self.reasoning_rate)

        if uvar_local_emb:
            U_opt = optimizer_class(uvar_local_emb, self.reasoning_rate)
        eflosses = [(-1, -1, -1)]
        ulosses = []

        for i in range(self.reasoning_steps):
            # maximize the truth value with repect to evars and fvars
            if efvar_local_emb:
                EF_opt.zero_grad()
                tv = formula.evaluate_truth_values(
                    self.tnorm, self.nbp, self.nbp.margin,
                    all_candidates=False)

                if enable_target:
                    for f in target_emb_dict:
                        _f_emb = formula.get_var_local_embedding(f)
                        _t_emb = target_emb_dict[f]
                        dist = torch.norm(_f_emb - _t_emb, p=2, dim=-1)
                        equality_tv = torch.exp(-dist / self.sigma ** 2)
                        tv = self.tnorm.conjunction(tv, equality_tv)

                ntv = - tv.mean()
                reg = 0.05 * sum([
                    torch.mean(
                        torch.sum(
                            torch.abs(self.nbp.regularization(emb)) ** 3, -1)
                            )
                    for emb in efvar_local_emb])
                efloss = ntv + reg
                eflosses.append((ntv.item(), reg.item(), efloss.item()))
                efloss.backward()
                EF_opt.step()
            else:
                efloss = None

            if uvar_local_emb:
                # TODO update the universal part according to the existential part
                # minimize the truth value with respect to uvars
                U_opt.zero_grad()
                uloss = formula.evaluate_truth_values(
                    self.tnorm, self.nbp, self.nbp.margin,
                    all_candidates=False).mean()
                ulosses.append(ulosses.item())
                uloss.backward()
                U_opt.step()
            else:
                uloss = None

            if math.fabs(eflosses[-1][-1] - eflosses[-2][-1]) < 1e-9:
                break

        # print(f"continuous search for {formula.lstr()} breaks at step {i}")

        truth_values = formula.evaluate_truth_values(
            self.tnorm, self.nbp, self.nbp.margin,
            all_candidates=not infer_free)
        fvar_local_emb_dict = {
                k: formula.get_var_local_embedding(k) for k in formula.free_variable_dict}
        assert truth_values is not None
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
                    head_emb = formula.get_tail_embed_from_formula(
                        self.nbp, head_name
                    )
                    rel_emb = self.nbp.get_relation_emb(
                        formula.pred_grounded_relation_id_dict[rel_name]
                    )

                    tail_emb = self.nbp.estimate_tail_emb(head_emb, rel_emb)
                    # formula.set_var_local_embedding(tail_name, tail_emb)
                    formula.set_var_local_embedding(tail_name, tail_emb)

                elif not formula.term_initialized(head_name) and formula.term_initialized(tail_name):
                    tail_emb = formula.get_tail_embed_from_formula(
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

    def reasoning(self, fof_list: List[FirstOrderFormula], infer_free, enable_target=False):
        # then it comes into a batched formula list
        if isinstance(fof_list, list):
            return [self._reason_single_formula(fof, infer_free, enable_target) for fof in fof_list]
        else:
            return self._reason_single_formula(fof_list, infer_free, enable_target)
