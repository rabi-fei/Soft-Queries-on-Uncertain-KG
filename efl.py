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

        pass


    def _pick_from_finite_model(self, known_entity_list, mode='random'):
        # get all possible triples (neighbering_graph) from self.finite_model

        # batch evaluation by self.neural_model

        # pick the worst triple (or other choices define by the mode)
        pass

    def learning_with_efg(self, batch_size, optimizer):
        triples = self.play_efg()
