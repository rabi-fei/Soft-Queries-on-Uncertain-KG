from abc import ABC, abstractmethod

class Learner(ABC):

    @classmethod
    @abstractmethod
    def get_data_iterator(datalist):
        """
        We pack the datalist into the iterator as we wish
        """
        pass

    @abstractmethod
    def forward(self, batch_input, num_negative_samples):
        """
        the batch input is forward passed to dict of outputs
        """
        pass