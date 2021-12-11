from model.abstract_models import KG, Triple, NeuralBinaryPredicate


class LPL:
    """
    A class for Link Prediction Learning
    """

    def __init__(self,
                 finite_model: KG,
                 neural_model: NeuralBinaryPredicate,
                 **kwargs):
        """
        kwargs is intend for the iterator parameters
        """
        self.finite_model = finite_model
        self.neural_model = neural_model
        self.num_epoch = 0
        self.kwargs = kwargs
        self.triple_iter = self.finite_model.get_train_triple_ns_iterator(
            **self.kwargs)

    def get_next_batch_of_triples(self):
        try:
            batch = next(self.triple_iter)
            return batch
        except StopIteration:
            self.num_epoch += 1
            print("epoch", self.num_epoch)
            self.triple_iter = self.finite_model.get_train_triple_ns_iterator(
                **self.kwargs)
            batch = next(self.triple_iter)
            return batch

    def learning_step(self,
                      optimizer=None,
                      log=True):
        assert optimizer is not None

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
