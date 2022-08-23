"""
A file maintains reasoning machine
"""
import math
from typing import Dict, List
from random import sample

import torch
from torch import nn
from src.language.fof import (BinaryPredicate, Conjunction, Disjunction,
                              FirstOrderFormula, Negation, Term)
from src.language.tnorm import Tnorm
from src.structure.neural_binary_predicate import NeuralBinaryPredicate


class GradientEFOReasoner:
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 tnorm: Tnorm,
                 reasoning_rate,
                 reasoning_steps,
                 reasoning_optimizer,
                 sigma):
        self.reasoning_rate = reasoning_rate
        self.reasoning_steps = reasoning_steps
        self.reasoinng_optimizer = reasoning_optimizer
        self.nbp = nbp
        self.tnorm: Tnorm = tnorm
        self.sigma = sigma

        # determined during the optimization
        self.formula: FirstOrderFormula = None
        self.term_local_emb_dict = {}
        self._last_ground_free_var_emb = {}

    @classmethod
    def create(cls,
               nbp: NeuralBinaryPredicate,
               tnorm: Tnorm,
               reasoning_rate,
               reasoning_steps,
               reasoning_optimizer,
               sigma=1):
        rm = cls(nbp,
                 tnorm,
                 reasoning_rate,
                 reasoning_steps,
                 reasoning_optimizer,
                 sigma)
        return rm

    def initialize_with_formula(self, formula: FirstOrderFormula):
        self.formula = formula
        self.term_local_emb_dict = {
            term_name: None
            for term_name in self.formula.term_dict
        }

        self._last_ground_free_var_emb = {}

    def set_local_embedding(self, key, tensor):
        self.term_local_emb_dict[key] = tensor.detach().clone()
        self.term_local_emb_dict[key].requires_grad = True

    def term_initialized(self, term_name):
        return self.formula.has_term_grounded_entity_id_list(term_name) \
                    or self.term_local_emb_dict[term_name] is not None

    def initialize_variable_embeddings(self):
        """
        Input args:
        Return args:
            evars: list of existential variables
            uvars: list of universal variables
            fvars: list of free variables
        """

        def check_all_var_initialized():
            return all(self.term_initialized(term_name)
                       for term_name in self.formula.term_dict)

        while not check_all_var_initialized():
            for rel_name, pred in self.formula.predicate_dict.items():
                head_name, tail_name = pred.head.name, pred.tail.name

                if self.term_initialized(head_name) and not self.term_initialized(tail_name):
                    head_emb = self.get_embedding(
                        head_name, free_var_treatment='existential'
                    )
                    rel_emb = self.nbp.get_relation_emb(
                        self.formula.pred_grounded_relation_id_dict[rel_name]
                    )

                    tail_emb = self.nbp.estimate_tail_emb(head_emb, rel_emb)
                    self.set_local_embedding(tail_name, tail_emb)

                elif not self.term_initialized(head_name) and self.term_initialized(tail_name):
                    tail_emb = self.get_embedding(
                        head_name, free_var_treatment='existential'
                    )
                    rel_emb = self.nbp.get_relation_emb(
                        self.formula.pred_grounded_relation_id_dict[rel_name]
                    )

                    head_emb = self.nbp.estimate_head_emb(tail_emb, rel_emb)
                    # formula.set_var_local_embedding(head_name, head_emb)
                    self.set_local_embedding(head_name, head_emb)

                else:
                    continue
        return


    def initialize_variable_embeddings_v2(self):
        # normal initialization
        for symb_name in self.formula.symbol_dict:
            symb_emb = self.get_embedding(symb_name)

        for term_name in self.formula.existential_variable_dict:
            init_vec = torch.normal(0, 1e-3, symb_emb.shape, device=symb_emb.device)
            self.set_local_embedding(term_name, init_vec)

        for term_name in self.formula.free_variable_dict:
            init_vec = torch.normal(0, 1e-3, symb_emb.shape, device=symb_emb.device)
            self.set_local_embedding(term_name, init_vec)


    def get_embedding(self,
                      term_name,
                      begin_index=None,
                      end_index=None,
                      free_var_treatment='lift'):
        """
            free_var_treatment:
                (implemented)
                - all: evaluate across all candidates
                - lift: lift the free variable as the existential variable
                - groundans:{k}: ground the free variable into k random answers
                - groundnoisy:{k}: ground the free variable into k noisy variables
                (to implement)
                - groundansfull: ground the answers to a full answer set
                - groundbarycenterfull: ground the barycenter of the answer set
                - groundbarycenter: ground the barycenter of the answer set
                - groundbarycenter: ground the barycenter of the answer set
        """
        if self.formula.term_dict[term_name].state == Term.FREE:
            # when it comes to the treatment of free variables, we dont consider batch
            if free_var_treatment.lower() == 'all':
                return self.nbp.entity_embedding.unsqueeze(-2)
            elif free_var_treatment.lower() == 'lift':
                return self.term_local_emb_dict[term_name]
            elif 'groundans' in free_var_treatment.lower():
                k = int(free_var_treatment.split(':')[-1])
                entity_id_list = [sample(eans[term_name], k=k)[0]
                                  for eans in self.formula.easy_answer_list[begin_index: end_index]]
                entity_id_tensor = torch.tensor(entity_id_list).T
                emb = self.nbp.get_entity_emb(entity_id_tensor)
                # emb = emb.unsqueeze(-2)
                self._last_ground_free_var_emb[term_name] = emb
                return emb
            elif 'groundnoisy' in free_var_treatment.lower():
                k = int(free_var_treatment.split(':')[-1])
                entity_id_list = torch.randint(low=0,
                                               high=self.nbp.num_entities,
                                               size=(k, end_index - begin_index))
                emb = self.nbp.get_entity_emb(entity_id_list)
                self._last_ground_free_var_emb[term_name] = emb
                return emb
            else:
                raise NotImplementedError
        else:
            if self.formula.has_term_grounded_entity_id_list(term_name):
                emb = self.nbp.get_entity_emb(
                    self.formula.get_term_grounded_entity_id_list(term_name))
            elif self.term_local_emb_dict[term_name] is not None:
                emb = self.term_local_emb_dict[term_name]
            else:
                raise KeyError("Embedding does not found")
            # when it is not free variable, we consider the batch
            if begin_index is not None and end_index is not None:
                emb = emb[begin_index: end_index]
            return emb

    def evaluate_truth_values(self, free_var_treatment, batch_size_eval=None):
        """
        Input args:

        Return args:
        """
        def run_in_batch(batch_size):
            begin_idx = 0
            end_idx = begin_idx + batch_size
            end_idx = min(self.formula.num_instances, end_idx)
            collect = []
            while begin_idx < self.formula.num_instances:
                ret = self.batch_evaluate_truth_values(
                    self.formula.formula,
                    begin_idx, end_idx, free_var_treatment)
                collect.append(ret)

                begin_idx = end_idx
                end_idx = begin_idx + batch_size
                end_idx = min(self.formula.num_instances, end_idx)

            return torch.cat(collect, dim=-1)

        if batch_size_eval:
            return run_in_batch(batch_size=batch_size_eval)
        else:
            return run_in_batch(batch_size=self.formula.num_instances)

    def batch_evaluate_truth_values(self,
                                    formula,
                                    begin_index,
                                    end_index,
                                    free_var_treatment):
        """
        Recursive evaluation of the formula functions
        Input args:
            formula: the formula at this time
        Return args:
            - truth values in shape either:
                 [num candidate answers, batch size]
                 or
                 [batch size]

        """
        if isinstance(formula, Conjunction):
            return self.tnorm.conjunction(
                self.batch_evaluate_truth_values(
                    formula.formulas[0], begin_index, end_index, free_var_treatment),
                self.batch_evaluate_truth_values(
                    formula.formulas[1], begin_index, end_index, free_var_treatment)
            )

        elif isinstance(formula, Disjunction):
            return self.tnorm.disjunction(
                self.batch_evaluate_truth_values(
                    formula.formulas[0], begin_index, end_index, free_var_treatment),
                self.batch_evaluate_truth_values(
                    formula.formulas[1], begin_index, end_index, free_var_treatment)
            )

        elif isinstance(formula, Negation):
            return self.tnorm.negation(
                self.batch_evaluate_truth_values(
                    formula.formula, begin_index, end_index, free_var_treatment)
            )

        elif isinstance(formula, BinaryPredicate):
            head_name = formula.head.name
            tail_name = formula.tail.name
            head_emb = self.get_embedding(
                head_name, begin_index, end_index, free_var_treatment)
            tail_emb = self.get_embedding(
                tail_name, begin_index, end_index, free_var_treatment)

            rel_emb = self.nbp.get_relation_emb(
                formula.relation_id_list[begin_index: end_index])
            batch_score = self.nbp.embedding_score(head_emb, rel_emb, tail_emb)
            batch_truth_value = self.nbp.score2truth_value(batch_score)
            # batch_truth_value = batch_score  # CQD's trick for 2i, 3i
            return batch_truth_value

    def optimize_term_local_embedding(self, free_var_treatment, equality):
        if equality: assert 'ground' in free_var_treatment

        self.initialize_variable_embeddings()
        evar_local_emb = [
            self.get_embedding(term_name)
            for term_name in self.formula.existential_variable_dict
            if self.term_local_emb_dict[term_name] is not None]

        if free_var_treatment.lower() == "lift" or equality:
            for term_name in self.formula.free_variable_dict:
                assert self.term_local_emb_dict[term_name] is not None
                emb = self.get_embedding(term_name, free_var_treatment='lift')
                evar_local_emb.append(emb)

        # TODO logic needs to be optimized
        if len(evar_local_emb) == 0 or (len(evar_local_emb) == 1 and free_var_treatment.lower() == 'lift'):
            return []

        OptimizerClass = getattr(torch.optim, self.reasoinng_optimizer)
        optim: torch.optim.Optimizer = OptimizerClass(
            evar_local_emb, self.reasoning_rate)

        traj = [(-1, -1, -1)]

        for i in range(self.reasoning_steps):
            if equality:
                tv = self.evaluate_truth_values(free_var_treatment='lift')
                # conjunction when equality
                for term_name in self.formula.free_variable_dict:
                    free_var_local_emb = self.get_embedding(
                        term_name, free_var_treatment='lift')
                    free_var_ground_emb = self.get_embedding(
                        term_name, free_var_treatment=free_var_treatment)
                    free_var_dist = torch.sum(
                        (free_var_local_emb - free_var_ground_emb) ** 2, dim=-1)
                    dist_tv = torch.exp(
                         free_var_dist / self.sigma ** 2
                    )
                    tv = self.tnorm.conjunction(tv, dist_tv)
            else:
                tv = self.evaluate_truth_values(free_var_treatment=free_var_treatment)

            ntv = -tv.mean()
            efvar_local_emb_mat = torch.stack(evar_local_emb)
            reg = self.nbp.regularization(efvar_local_emb_mat).mean()

            loss = ntv + reg * 0.05
            traj.append((ntv.item(), reg.item(), loss.item()))
            optim.zero_grad()
            loss.backward()
            optim.step()

            if math.fabs(traj[-1][-1] - traj[-2][-1]) < 1e-9:
                break

        return traj


class RelationalGNNLayer(nn.Module):
    def __init__(self, input_dim, rel_dim, output_dim):
        super(RelationalGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.rel_dim = rel_dim
        self.hidden_dim = output_dim

        self.h2t_linear = nn.Linear(input_dim + rel_dim, output_dim)
        # self.t2h_linear = nn.Linear(input_dim + rel_dim, output_dim)

    def forward(self, ent_rel_list):
        encoded_entity_list = []
        for ent, rel, rel_order in ent_rel_list:
            # if rel_order == +1, we estimate the tail by head and rel
            # if rel_order == -1, we estimate the head by rel and tail
            entrel = torch.cat([ent, rel], dim=-1)
            if rel_order > 0:
                enc_entrel = self.h2t_linear(entrel)
            else:
                enc_entrel = self.t2h_linear(entrel)

            encoded_entity_list.append([enc_entrel, rel, rel_order])
        return encoded_entity_list


class RelationalDeepSet(nn.Module):
    def __init__(self, ent_dim, rel_dim, num_layer=1) -> None:
        super(RelationalDeepSet, self).__init__()
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.hidden_dim = ent_dim
        self.num_layer = num_layer

        self.init_rlinear = RelationalGNNLayer(self.ent_dim, self.rel_dim, self.hidden_dim)
        # for i in range(self.num_layer-1):
        #     # encode the inputs into the hidden dim.
        #     setattr(self,
        #         f'rlinear-{i}',
        #         RelationalGNNLayer(self.hidden_dim, self.rel_dim, self.hidden_dim))

        # self.clf = nn.Sequential(
        #     # nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.ent_dim)
        #     )


    def forward(self, ent_rel_list):
        # element wise entity transformation

        def apply_relu(ent_rel_list):
            ent_rel_list = [(torch.relu(e), r, o)
                            for e, r, o in ent_rel_list]
            return ent_rel_list

        def add(ent_rel_list1, ent_rel_list2):
            ent_rel_list = [
                (e1+e2, r, o)
                for (e1, r, o), (e2, *_)
                in zip(ent_rel_list1, ent_rel_list2)]
            return ent_rel_list

        hidden_ent_rel_list = self.init_rlinear(ent_rel_list)
        # for i in range(self.num_layer-1):
        #     apply_relu(hidden_ent_rel_list)
        #     hidden_ent_rel_list = getattr(self,
        #                                   f'rlinear-{i}')(hidden_ent_rel_list)
        #     hidden_ent_rel_list = add(hidden_ent_rel_list, ent_rel_list)

        agg = 0
        for e, *_ in hidden_ent_rel_list:
            agg += e
        # out = self.clf(agg)
        return agg


class GNNEFOReasoner:
    """
    In this class, we estimate the lifted embeddings of existential variables
    by GNN.

    """
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 tnorm: Tnorm,
                 relational_deepset: RelationalDeepSet):
        self.nbp = nbp
        self.tnorm: Tnorm = tnorm
        self.relational_deepset = relational_deepset

        # formula dependent
        self.formula: FirstOrderFormula = None
        self.term_local_emb_dict = {}
        self._last_ground_free_var_emb = {}
        self.visited_set = set()

    @classmethod
    def create(cls,
               formula: FirstOrderFormula,
               nbp: NeuralBinaryPredicate,
               tnorm: Tnorm,
               reasoning_rate,
               reasoning_steps,
               reasoning_optimizer,
               sigma=1):
        rm = cls(formula,
                 nbp,
                 tnorm,
                 reasoning_rate,
                 reasoning_steps,
                 reasoning_optimizer,
                 sigma)
        return rm

    def initialize_with_formula(self, formula):
        self.formula: FirstOrderFormula = formula
        self.term_local_emb_dict = {term_name: None
                                    for term_name in self.formula.term_dict}
        self._last_ground_free_var_emb = {}
        self.visited_set = set()

    def set_local_embedding(self, key, tensor):
        self.term_local_emb_dict[key] = tensor

    def term_initialized(self, term_name):
        return self.formula.has_term_grounded_entity_id_list(term_name) \
                    or self.term_local_emb_dict[term_name] is not None

    def get_embedding(self,
                      term_name,
                      begin_index=None,
                      end_index=None):
        """
            If the embedding is known, then return the original ones
            Otherwise, return the lifted embedding estimated by DeepSet
        """
        self.visited_set.add(term_name)

        if self.term_local_emb_dict[term_name] is not None:
            return self.term_local_emb_dict[term_name]

        if self.formula.has_term_grounded_entity_id_list(term_name):
            entity_id = self.formula.get_term_grounded_entity_id_list(term_name)
            if begin_index is not None and end_index is not None:
                entity_id = entity_id[begin_index: end_index]
            emb = self.nbp.get_entity_emb(entity_id)
            self.set_local_embedding(term_name, emb)
        else:
            related_predicate_list = self.formula.term_name2predicate_name_dict[
                term_name]
            ent_rel_ord = []
            for pred_name in related_predicate_list:
                head, tail = self.formula.predicate_dict[pred_name].get_terms()
                rel_id = self.formula.get_pred_grounded_relation_id_list(pred_name)
                rel = self.nbp.get_relation_emb(rel_id)
                if head.name == term_name:
                    if tail.name in self.visited_set:
                        continue
                    ord = -1
                    ent = self.get_embedding(tail.name)
                elif tail.name == term_name:
                    if head.name in self.visited_set:
                        continue
                    ord = 1
                    ent = self.get_embedding(head.name)
                else:
                    raise ValueError()
                ent_rel_ord.append([ent, rel, ord])
            emb = self.relational_deepset(ent_rel_ord)
            self.set_local_embedding(term_name, emb)
        print("return emb of ", term_name)
        return emb

    def evaluate_truth_values(self, batch_size_eval=None):
        """
        Input args:

        Return args:
        """
        def run_in_batch(batch_size):
            begin_idx = 0
            end_idx = begin_idx + batch_size
            collect = []
            while begin_idx < self.formula.num_instances:
                ret = self.batch_evaluate_truth_values(
                    self.formula.formula,
                    begin_idx, end_idx)
                collect.append(ret)

                begin_idx = end_idx
                end_idx = begin_idx + batch_size
                end_idx = min(self.formula.num_instances, end_idx)

            return torch.cat(collect, dim=-1)

        if batch_size_eval:
            return run_in_batch(batch_size=batch_size_eval)
        else:
            return run_in_batch(batch_size=self.formula.num_instances)

    def batch_evaluate_truth_values(self,
                                    formula,
                                    begin_index,
                                    end_index,
                                  ):
        """
        Recursive evaluation of the formula functions
        Input args:
            formula: the formula at this time
        Return args:
            - truth values in shape either:
                 [num candidate answers, batch size]
                 or
                 [batch size]

        """
        if isinstance(formula, Conjunction):
            return self.tnorm.conjunction(
                self.batch_evaluate_truth_values(
                    formula.formulas[0], begin_index, end_index),
                self.batch_evaluate_truth_values(
                    formula.formulas[1], begin_index, end_index)
            )

        elif isinstance(formula, Disjunction):
            return self.tnorm.disjunction(
                self.batch_evaluate_truth_values(
                    formula.formulas[0], begin_index, end_index),
                self.batch_evaluate_truth_values(
                    formula.formulas[1], begin_index, end_index)
            )

        elif isinstance(formula, Negation):
            return self.tnorm.negation(
                self.batch_evaluate_truth_values(
                    formula.formula, begin_index, end_index)
            )

        elif isinstance(formula, BinaryPredicate):
            head_name = formula.head.name
            tail_name = formula.tail.name
            head_emb = self.get_embedding(
                head_name, begin_index, end_index)
            tail_emb = self.get_embedding(
                tail_name, begin_index, end_index)

            rel_emb = self.nbp.get_relation_emb(
                formula.relation_id_list[begin_index: end_index])
            batch_score = self.nbp.embedding_score(head_emb, rel_emb, tail_emb)
            batch_truth_value = self.nbp.score2truth_value(batch_score)
            # batch_truth_value = batch_score  # CQD's trick for 2i, 3i
            return batch_truth_value
