from abc import ABC, abstractmethod

class LearnerForwardOutput:
    def __init__(self, pos_score=0, pos_prob=0, neg_score=0, neg_prob=0):
        self.pos_score = pos_score
        self.pos_prob = pos_prob
        self.neg_score = neg_score
        self.neg_prob = neg_prob

class Learner(ABC):

    @classmethod
    @abstractmethod
    def get_data_iterator(datalist):
        """
        We pack the datalist into the iterator as we wish
        """
        pass

    @abstractmethod
    def forward(self, batch_input, num_negative_samples) -> LearnerForwardOutput:
        """
        the batch input is forward passed to dict of outputs
        """
        pass

