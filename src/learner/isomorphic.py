from torch.utils.data import DataLoader

from src.learner.abstract import Learner, LearnerForwardOutput
from src.learner.sampler import lcwa_negative_sampling, rel_negative_sampling
from src.structure.abstract_models import KnowledgeGraph, NeuralBinaryPredicate


class IsomorphicLearner(Learner):
    def __init__(self,
                 kg: KnowledgeGraph,
                 nbp: NeuralBinaryPredicate):
        self.kg = kg
        self.nbp = nbp

    def forward(self, batch_input, num_negative_samples=1, strategy='lcwa'):
        """
        In this case we assume the batch input is a list of 3 tensors
        """
        phead, prel, ptail = batch_input

        assert 'lcwa' in strategy
        nhead, ntail = lcwa_negative_sampling(
            phead_id_ten=phead,
            ptail_id_ten=ptail,
            num_entities=self.kg.num_entities)

        pos_scores = self.nbp.batch_predicate_score([phead, prel, ptail]).squeeze()
        neg_scores = self.nbp.batch_predicate_score([nhead, prel, ntail]).squeeze()

        if 'rel' in strategy:
            pass

        output = LearnerForwardOutput(
            pos_score=pos_scores,
            pos_prob=self.nbp.score2prob(pos_scores),
            neg_score=neg_scores,
            neg_prob=self.nbp.score2prob(neg_scores),
        )

        return output


class __IsomorphicLearner:
    """
    A class for Link Prediction Learning
    """

    def __init__(self,
                 finite_model: KnowledgeGraph,
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

        loss = self.neural_model.compute_triple_pair_loss(
            pos_triples=pos_triple_ten,
            neg_triples=neg_triple_ten)

        loss.backward()
        optimizer.step()

        if log:
            log_dict['loss'] = loss.item()
            return log_dict
