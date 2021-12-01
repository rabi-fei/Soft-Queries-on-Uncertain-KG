from model.kg import KG, Triple
from model.predicate import NeuralBinaryPredicate


class LPL:
    """
    A class for Link Prediction Learning
    """

    def __init__(self,
                 finite_model: KG,
                 neural_model: NeuralBinaryPredicate):
        self.finite_model = finite_model
        self.neural_model = neural_model

    def learning_step(self,
                      optimizer=None,
                      batch_size=None,
                      positive_triples=None,
                      negative_triples=None,
                      log=True):
        assert optimizer is not None

        if log:
            log_dict = {}

        optimizer.zero_grad()

        if positive_triples is None:
            assert batch_size is not None
            # random positive samples according to the batch size

        if negative_triples is None:
            negative_triples = self.finite_model.lcwa_negative_sampling(
                positive_triples)

        loss = self.neural_model.compute_triple_loss(
            pos_triples=positive_triples,
            neg_triples=negative_triples)

        loss.backward()
        optimizer.step()

        if log:
            log_dict['loss'] = loss
            return log_dict
