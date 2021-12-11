from torch.utils.data import DataLoader
import torch
from model.abstract_models import KG, NeuralBinaryPredicate


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
        self.triple_iter = self.get_train_triple_ns_iterator()

    def get_train_triple_ns_iterator(self):
        dataloader = DataLoader(self.finite_model.triples, **self.kwargs)
        for phead, prel, ptail in dataloader:
            random_entities = torch.randint(
                low=0, high=self.finite_model.num_entities, size=phead.shape)

            head_collapse = torch.randint(
                low=0, high=2, size=phead.shape).bool()
            tail_collapse = head_collapse.logical_not()

            nhead = torch.where(head_collapse, random_entities, phead)
            ntail = torch.where(tail_collapse, random_entities, ptail)

            yield (phead, prel, ptail), (nhead, prel, ntail)

    def get_next_batch_of_triples(self):
        try:
            batch = next(self.triple_iter)
            return batch
        except StopIteration:
            self.num_epoch += 1
            print("epoch", self.num_epoch)
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
