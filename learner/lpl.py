from torch.utils.data import DataLoader
from model.abstract_models import KG, NeuralBinaryPredicate
from learner.utils import lcwa_negative_sampling


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
        self.device = neural_model.device
        self.num_epoch = 0
        self.kwargs = kwargs
        self.triple_iter = self.get_train_triple_ns_iterator()

    def get_train_triple_ns_iterator(self):
        dataloader = DataLoader(self.finite_model.triples, **self.kwargs)
        for phead, prel, ptail in dataloader:
            nhead, ntail = lcwa_negative_sampling(
                phead, ptail, self.finite_model.num_entities)
            yield ((phead.to(self.device),
                    prel.to(self.device),
                    ptail.to(self.device)),
                   (nhead.to(self.device),
                    prel.to(self.device),
                    ntail.to(self.device)))

    def get_next_batch_of_triples(self):
        try:
            batch = next(self.triple_iter)
        except StopIteration:
            self.num_epoch += 1
            print("train epoch", self.num_epoch)
            self.triple_iter = self.get_train_triple_ns_iterator()
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
