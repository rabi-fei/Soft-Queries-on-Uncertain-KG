from model.kg import KG, Triple
from model.predicate import NeuralBinaryPredicate


class LPL:
    """
    A class for Link Prediction Learning
    """

    def __init__(self,
                 finite_model: KG,
                 triple_loader,
                 neural_model: NeuralBinaryPredicate):
        self.triple_loader = triple_loader
        self.triple_iter = iter(self.triple_loader)
        self.finite_model = finite_model
        self.neural_model = neural_model
        self.num_epoch = 0

    def get_next_batch_of_triples(self):
        try:
            batch = next(self.triple_iter)
            return batch
        except StopIteration:
            self.num_epoch += 1
            self.triple_iter = iter(self.triple_loader)
            batch = next(self.triple_iter)
            return batch

    def learning_step(self,
                      optimizer=None,
                      log=True):
        assert optimizer is not None

        if log:
            log_dict = {}

        optimizer.zero_grad()

        positive_triples = self.get_next_batch_of_triples()

        negative_triples = self.finite_model.lcwa_negative_sampling(
            positive_triples, negative_sample_scope='graph')

        loss = self.neural_model.compute_triple_loss(
            pos_triples=positive_triples,
            neg_triples=negative_triples)

        loss.backward()
        optimizer.step()

        if log:
            log_dict['loss'] = loss
            return log_dict
