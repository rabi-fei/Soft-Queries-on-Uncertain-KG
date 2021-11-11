from model.kg import KG
from model.predicate import NeuralBinaryPredicate

class EFL:
    """
    A class for Ehrenfeucht–Fraı̈sśe Learning (EFL).
    EFL is based on play of EFG to optimize the neural (binary predicate) model
    so that it can be more elementary equivalent to the finite (knowledge graph)
    model.
    """

    def __init__(self,
                 finite_model: KG,
                 neural_model: NeuralBinaryPredicate,
                 round=2):
        self.finite_model = finite_model
        self.neural_model = neural_model
        self.round = round

    def play_efg(self, begin_entity_id=None, round=None):
        """
        Play the EF game and get the game position
        Optimize the score over the sub-graph
        """
        if round is None:
            round = self.round
        pass
        if begin_entity_id is None:
            entity_list = []
        else:
            entity_list = [begin_entity_id]
            round -= 1

        for _ in range(round):
            new_entity_id = self._pick_from_finite_model(entity_list)
            entity_list.append(new_entity_id)

        # get sub_graph from self.finite_model
        sub_graph_triples = self.finite_model.get_sub_graph(entity_list)
        return sub_graph_triples


    def _pick_from_finite_model(self, known_entity_list, mode='random'):
        # get all possible triples (neighbering_graph) from self.finite_model
        neighbering_graph_triples = self.finite_model.get_neighbor_graph(
            known_entity_list)

        # batch evaluation by self.neural_model
        self.neural_model.find_low_score_true_triples(neighbering_graph_triples, k=1)
        self.neural_model.find_high_score_false_triples(neighbering_graph_triples, k=1)

        # pick the worst triple (or other choices define by the mode)
        pass

    def learning_step(self, batch_size, optimizer):
        triples = self.play_efg()
