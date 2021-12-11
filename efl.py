from typing import List
import random
from model.abstract_models import KG, Triple, NeuralBinaryPredicate


class EFL:
    """
    A class for Ehrenfeucht–Fraı̈sśe Learning (EFL).
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

    def play_efg(self,
                 begin_entity_id=None,
                 round=None,
                 spolier_mode=None,
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
        if spolier_mode is None:
            spolier_mode = 'random'
            spolier_args = {'threshold': 0.5}

        # inside the game
        for _ in range(round):
            new_entity_id = self._spoiler_step(
                entity_list, mode=spolier_mode, **spolier_args)
            if new_entity_id is None:
                break
            entity_list.append(new_entity_id)

        # get sub_graph from self.finite_model
        pos_triples, neg_triples = self.finite_model.get_sub_graph(entity_list)
        assert len(pos_triples) == len(neg_triples)
        if len(pos_triples) > 0:
            return pos_triples, neg_triples
        else:
            return self.play_efg()

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

    def learning_step(self, batch_size, optimizer, log=True):
        if log:
            log_dict = {}

        optimizer.zero_grad()

        pos_triples, neg_triples = [], []
        for _ in range(batch_size):
            pt, nt = self.play_efg()
            pos_triples += pt
            neg_triples += nt

        loss = self.neural_model.compute_triple_loss(
            pos_triples, neg_triples)
        loss.backward()
        optimizer.step()

        if log:
            log_dict['num_pos_triples'] = len(pos_triples)
            log_dict['num_neg_triples'] = len(neg_triples)
            log_dict['loss'] = loss
            return log_dict
