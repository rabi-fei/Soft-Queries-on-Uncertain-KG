from typing import List
import random
from unicodedata import bidirectional
import torch

from torch.utils.data.dataloader import DataLoader
from learner.utils import lcwa_negative_sampling
from model.abstract_models import KG, Triple, NeuralBinaryPredicate, triples_to_tensors


class EFG:
    """
    A class for Ehrenfeucht–Fraı̈sśe Game.
    EFL is based on EFG to optimize the neural (binary predicate) model
    so that it can be more elementary equivalent to the finite (knowledge graph)
    model.
    """

    def __init__(self,
                 finite_model: KG,
                 neural_model: NeuralBinaryPredicate,
                 round=5):
        self.finite_model = finite_model
        self.neural_model = neural_model
        self.round = round

    def __call__(self, begin_entity_id_list):
        pos_triples = []
        for begin_entity_id in begin_entity_id_list:
            pos_triples.extend(self.play_efg(begin_entity_id=begin_entity_id))
        phead, prel, ptail = triples_to_tensors(pos_triples)
        return phead, prel, ptail

    def play_efg(self,
                 begin_entity_id=None,
                 round=None,
                 spoiler_mode=None,
                 **kwargs) -> List[Triple]:
        """
        Play the EF game and get the game position
        Optimize the score over the sub-graph
        """
        # prepare rounds
        if round is None:
            round = self.round

        # prepare initial entities
        if begin_entity_id is None:
            entity_list = [self.get_random_entity()]
        else:
            entity_list = [begin_entity_id]

        # prepare spolier argument
        if spoiler_mode is None:
            spoiler_mode = 'random'
            spolier_args = {'threshold': 0.5}

        # inside the game
        for _ in range(round):
            new_entity_id = self._spoiler_step(
                entity_list, mode=spoiler_mode, **spolier_args)
            if new_entity_id is None:
                break
            entity_list.append(new_entity_id)

        # get sub_graph from self.finite_model
        pos_triples, neg_triples = self.finite_model.get_sub_graph(entity_list)
        return pos_triples, neg_triples

    def get_random_entity(self):
        return self.finite_model.get_random_entity()

    def get_random_relation(self):
        return self.finite_model.get_random_relation()

    def _spoiler_step(self, entity_list, mode, **spolier_args):
        if mode == 'random':
            return self._spoiler_random_step(entity_list, **spolier_args)
        else:
            raise NotImplementedError(
                f"spoiler step mode {mode} is not implemented")

    def _spoiler_random_step(self, entity_list, threshold=0.5, **kwargs):
        rand = random.random()
        if rand < threshold:
            new_entity_id = self._spolier_act_on_finite_model(
                entity_list, **kwargs)
        else:
            new_entity_id = self._spoiler_act_on_neural_model(
                entity_list, **kwargs)
        return new_entity_id

    def _spolier_act_on_finite_model(self, known_entity_list):
        # get all possible triples (neighbering_graph) from self.finite_model
        neighbering_graph_triples = self.finite_model.get_neighbor_graph(
            known_entity_list)

        # batch evaluation by self.neural_model
        sorted_triples = self.neural_model.sort_triples_by_scores(
            neighbering_graph_triples, assending=True)

        # pick the worst triple (or other choices define by the mode)
        for h, _, t in sorted_triples:
            if h not in known_entity_list:
                return h
            if t not in known_entity_list:
                return t

    def _spoiler_act_on_neural_model(self, known_entity_list, random_search_size=3):
        neighbors_triples = set(self.finite_model.get_neighbor_graph(
            known_entity_list))

        non_local_triples = []

        def register_non_local_triples(h, r, t):
            if (h, r, t) not in neighbors_triples:
                non_local_triples.append((h, r, t))

        for _ in range(random_search_size):
            _e = self.get_random_entity()
            _r = self.get_random_relation()
            for e in known_entity_list:
                register_non_local_triples(_e, _r, e)
                register_non_local_triples(e, _r, _e)

        sorted_triples = self.neural_model.sort_triples_by_scores(
            non_local_triples, assending=False)

        for h, _, t in sorted_triples:
            if h not in known_entity_list:
                return h
            if t not in known_entity_list:
                return t


class TensorizedEFG:
    def __init__(self,
                 finite_model: KG,
                 neural_model: NeuralBinaryPredicate,
                 round=5):
        self.finite_model = finite_model
        self.neural_model = neural_model
        assert self.finite_model.device == self.neural_model.device
        self.device = self.finite_model.device
        self.round = round

    def play(self, begin_entity_id_list: List[int], spoiler_mode=None):

        # prepare initial entities
        if isinstance(begin_entity_id_list, list):
            batch_entities = torch.tensor(
                begin_entity_id_list,
                device=self.device).view(-1, 1)
        elif isinstance(begin_entity_id_list, torch.Tensor):
            batch_entities = begin_entity_id_list.to(self.device).view(-1, 1)
        else:
            raise NotImplementedError

        round_mask = torch.ones(size=(len(begin_entity_id_list), self.round),
                                dtype=torch.long,
                                device=self.device)

        # prepare spolier argument
        if spoiler_mode is None:
            spoiler_mode = 'random'
            spoiler_args = {'threshold': 5}

        # inside the game
        for i in range(1, self.round):
            new_batch_entity, _round_mask = self._spoiler_step(
                batch_entities, round_mask[:, i-1].detach().clone(), mode=spoiler_mode, **spoiler_args)
            batch_entities = torch.cat([batch_entities, new_batch_entity],
                                       dim=-1)
            round_mask[:, i] = _round_mask

        # get sub_graph from self.finite_model
        pos_triples, neg_triples = self.finite_model.get_sub_graph(
            batch_entities)
        return pos_triples, neg_triples

    def _spoiler_step(self, batch_entities, round_mask, mode, **kwargs):
        if mode == 'random':
            return self._spoiler_random_step(batch_entities, round_mask, **kwargs)
        else:
            raise NotImplementedError(
                f"spoiler step mode {mode} is not implemented")

    def _spoiler_random_step(self, batch_entities, round_mask, threshold=0.5, **kwargs):
        rand = random.random()
        if rand < threshold:
            new_entity_id = self._spoiler_act_on_finite_model(
                batch_entities, round_mask, *kwargs)
        else:
            new_entity_id = self._spoiler_act_on_neural_model(
                batch_entities, round_mask, **kwargs)
        return new_entity_id

    def _spoiler_act_on_finite_model(self, batch_entities, round_mask):
        head_triples, counts = self.finite_model.get_neighbor_new_head(
            batch_entities)
        head_limits = counts.cumsum(dim=0)
        head_scores = self.neural_model.batch_pred_score(
            head_triples[:, 0], head_triples[:, 1], head_triples[:, 2])

        tail_triples, counts = self.finite_model.get_neighbor_new_tail(
            batch_entities)
        tail_limits = counts.cumsum(dim=0)
        tail_scores = self.neural_model.batch_pred_score(
            tail_triples[:, 0], tail_triples[:, 1], tail_triples[:, 2])

        head_begin_idx, tail_begin_idx = 0, 0
        batch_new_entity = batch_entities[:, -1].detach().clone().view(-1, 1)

        # adhoc may be improved by ragged tensor if one uses TF
        for i in range(len(counts)):
            if round_mask[i] == 0:
                continue

            head_end_idx = head_limits[i]
            tail_end_idx = tail_limits[i]

            case_head_scores = head_scores[head_begin_idx: head_end_idx]
            case_tail_scores = tail_scores[tail_begin_idx: tail_end_idx]

            if case_head_scores.numel() > 0:
                min_head_triple_id = case_head_scores.argmin()
                min_head_triple_score = case_head_scores[min_head_triple_id]
            else:
                min_head_triple_id = None
                min_head_triple_score = float('inf')

            if case_tail_scores.numel() > 0:
                min_tail_triple_id = case_tail_scores.argmin()
                min_tail_triple_score = case_tail_scores[min_tail_triple_id]
            else:
                min_tail_triple_id = None
                min_tail_triple_score = float('inf')

            if min_tail_triple_id is None and min_head_triple_id is None:
                round_mask[i] = 0
                continue

            if min_head_triple_score < min_tail_triple_score:
                batch_new_entity[i] = \
                    head_triples[head_begin_idx + min_head_triple_id, 0]
            else:
                batch_new_entity[i] = \
                    tail_triples[tail_begin_idx + min_tail_triple_id, 2]

            head_begin_idx = head_end_idx
            tail_begin_idx = tail_end_idx

        return batch_new_entity, round_mask


class EFL:

    def __init__(self,
                 finite_model: KG,
                 neural_model: NeuralBinaryPredicate,
                 round,
                 **kwargs):
        self.finite_model = finite_model
        self.neural_model = neural_model
        self.device = neural_model.device
        self.round = round
        self.num_epoch = 0
        self.kwargs = kwargs
        self.node_iter = self.get_train_node_efg_iterator()

        self.efg = TensorizedEFG(
            self.finite_model, self.neural_model, self.round)

    def get_train_node_efg_iterator(self):
        entity_list = list(self.finite_model.entity_set)
        if self.kwargs['shuffle']:
            random.shuffle(entity_list)

        pos_triple_buffer = []
        neg_triple_buffer = []
        for e in entity_list:
            pos_triples, neg_triples = self.efg.play_efg(begin_entity_id=e)
            pos_triple_buffer.extend(pos_triples)
            neg_triple_buffer.extend(neg_triples)
            if len(pos_triple_buffer) > self.kwargs['batch_size']:
                phead, prel, ptail = triples_to_tensors(
                    pos_triple_buffer, device=self.device)
                nhead, nrel, ntail = triples_to_tensors(
                    neg_triple_buffer, device=self.device)

                pos_triple_buffer = []
                neg_triple_buffer = []
                yield ((phead, prel, ptail), (nhead, nrel, ntail))

    def random_training_triple(self):
        entity_list = list(self.finite_model.entity_set)
        elist = random.sample(
            entity_list, k=self.kwargs['batch_size']//self.round)

        pos_triples_ten, _ = self.efg.play(begin_entity_id_list=elist)

        phead, prel, ptail = torch.split(
            pos_triples_ten, dim=1, split_size_or_sections=1)

        nhead, ntail = lcwa_negative_sampling(phead, ptail, self.finite_model.num_entities)


        return ((phead, prel, ptail), (nhead, prel, ntail))

    def get_next_batch_of_triples(self, epoch=False):
        if epoch:
            try:
                batch = next(self.node_iter)
            except StopIteration:
                self.num_epoch += 1
                print("train epoch", self.num_epoch)
                self.node_iter = self.get_train_node_efg_iterator()
                batch = next(self.node_iter)
        else:
            batch = self.random_training_triple()
        return batch

    def learning_step(self, optimizer, log=True):
        if log:
            log_dict = {}

        optimizer.zero_grad()

        pos_triple_ten, neg_triple_ten = self.get_next_batch_of_triples()

        loss = self.neural_model.compute_triple_loss(
            pos_triples=pos_triple_ten,
            neg_triples=neg_triple_ten)

        loss.backward()
        optimizer.step()

        if log:
            log_dict['loss'] = loss.item()
            return log_dict
