from typing import List
import random
from unicodedata import bidirectional
import torch

from torch.utils.data.dataloader import DataLoader
from learner.utils import lcwa_negative_sampling
from torch.nn.utils.rnn import pad_sequence
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
                 efg_round=5,
                 efg_rand_thr=0.5,
                 k_neural=5,
                 k_subgraph=5,
                 **kwargs):
        self.finite_model = finite_model
        self.neural_model = neural_model
        assert self.finite_model.device == self.neural_model.device
        self.device = self.finite_model.device
        self.round = efg_round
        self.k_neural = k_neural
        self.k_subgraph = k_subgraph
        self.efg_rand_thr = efg_rand_thr

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

        round_mask = torch.ones(size=(len(begin_entity_id_list), self.round+1),
                                dtype=torch.long,
                                device=self.device)

        # prepare spolier argument
        if spoiler_mode is None:
            spoiler_mode = 'random'
            spoiler_args = {'threshold': self.efg_rand_thr}

        # inside the game
        for i in range(1, self.round + 1):
            new_batch_entity, _round_mask = self._spoiler_step(
                batch_entities, round_mask[:, i-1].detach().clone(), mode=spoiler_mode, **spoiler_args)
            batch_entities = torch.cat([batch_entities, new_batch_entity],
                                       dim=-1)
            round_mask[:, i] = _round_mask

        # get sub_graph from self.finite_model
        outputs = self.finite_model.get_sub_graph(batch_entities)
        return outputs

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
        batch_size = batch_entities.size(0)
        first_index = torch.arange(batch_size, device=self.device)

        head_triples, counts = self.finite_model.get_neighbor_triples(
            batch_entities, reverse=True)
        head_scores = self.neural_model.batch_pred_score(
            head_triples[:, 0], head_triples[:, 1], head_triples[:, 2])

        head_triple_list = torch.split(
            head_triples, split_size_or_sections=counts)
        batch_head_triples = pad_sequence(
            head_triple_list, batch_first=True, padding_value=-1)
        head_score_list = torch.split(
            head_scores, split_size_or_sections=counts)
        batch_head_scores = pad_sequence(
            head_score_list, batch_first=True, padding_value=torch.inf)

        head_min_index = torch.argmin(batch_head_scores, dim=-1)
        head_min_score = batch_head_scores[first_index, head_min_index]
        head_min_entity = batch_head_triples[first_index, head_min_index, 0]

        tail_triples, counts = self.finite_model.get_neighbor_triples(
            batch_entities, reverse=False)
        tail_scores = self.neural_model.batch_pred_score(
            tail_triples[:, 0], tail_triples[:, 1], tail_triples[:, 2])

        tail_triple_list = torch.split(
            tail_triples, split_size_or_sections=counts)
        batch_tail_triples = pad_sequence(
            tail_triple_list, batch_first=True, padding_value=-1)
        tail_score_list = torch.split(
            tail_scores, split_size_or_sections=counts)
        batch_tail_scores = pad_sequence(
            tail_score_list, batch_first=True, padding_value=torch.inf)

        tail_min_index = torch.argmin(batch_tail_scores, dim=-1)
        tail_min_score = batch_tail_scores[first_index, tail_min_index]
        tail_min_entity = batch_tail_triples[first_index, tail_min_index, -1]

        batch_new_entity = torch.where(
            head_min_score < tail_min_score, head_min_entity, tail_min_entity)

        return batch_new_entity, batch_new_entity >= 0

    def _spoiler_act_on_neural_model(self, batch_entities, round_mask):

        Thead, Trel, Ttail = self.finite_model.get_non_neightbor_triple(
            batch_entities, k=self.k_neural, reverse=False)
        Tscores = self.neural_model.batch_pred_score(
            Thead, Trel, Ttail).squeeze()

        Hhead, Hrel, Htail = self.finite_model.get_non_neightbor_triple(
            batch_entities, k=self.k_neural, reverse=True)
        Hscores = self.neural_model.batch_pred_score(
            Hhead, Hrel, Htail).squeeze()

        batch_new_entity = batch_entities[:, -1].detach().clone().view(-1, 1)

        # adhoc may be improved by ragged tensor if one uses TF
        Tmax_index = Tscores.argmax(-1)
        Hmax_index = Hscores.argmax(-1)

        first_indices = torch.arange(
            batch_entities.size(0), device=self.device)

        Tmax_scores = Tscores[first_indices, Tmax_index]
        Hmax_scores = Tscores[first_indices, Hmax_index]

        batch_new_entity = torch.where(
            Tmax_scores > Hmax_scores, Tmax_index, Hmax_index)

        return batch_new_entity.view(-1, 1), torch.ones_like(batch_new_entity)


class EFL:

    def __init__(self,
                 finite_model: KG,
                 neural_model: NeuralBinaryPredicate,
                 batch_size,
                 k_nce,
                 margin,
                 **kwargs):
        self.finite_model = finite_model
        self.neural_model = neural_model
        self.batch_size = batch_size
        self.device = neural_model.device
        self.round = kwargs.get('round', 5)
        self.k_nce = k_nce
        self.margin = margin
        self.num_epoch = 0
        self.efg_kwargs = kwargs
        self.node_iter = self.get_train_node_efg_iterator()

        self.efg = TensorizedEFG(
            self.finite_model, self.neural_model, **self.efg_kwargs)

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
            entity_list, k=self.batch_size // self.round)

        output = self.efg.play(begin_entity_id_list=elist)

        return output

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

        output = self.get_next_batch_of_triples()

        loss = self.neural_model.compute_efg_nce_loss(
            **output, k_nce=self.k_nce, margin=self.margin)
        # loss = self.neural_model.compute_efg_pair_loss(**output)

        loss.backward()
        optimizer.step()

        if log:
            log_dict['loss'] = loss.item()
            log_dict['num_triples'] = len(output['subgraph_flat_triples'])
            return log_dict
