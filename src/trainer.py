import torch
from src.structure import KnowledgeGraph, NeuralBinaryPredicate
from src.learner import Learner
from src.utils.recorder import Recorder


class Trainer:
    def __init__(self,
                 kg: KnowledgeGraph,
                 nbp: NeuralBinaryPredicate,
                 learner: Learner,
                 optimizer: torch.optim.Optimizer,
                 objective='nce',
                 margin=10,
                 k_nce=1,
                 num_negative_samples=1,
                 ns_strategy='lcwa',
                 batch_size=256,
                 **kwargs):
        # important objects
        self.kg = kg
        self.nbp = nbp
        self.learner = learner
        self.optimzier = optimizer
        # parameters
        self.objective = objective
        self.margin = margin
        self.num_negative_samples = num_negative_samples
        self.k_nce = k_nce
        self.ns_strategy = ns_strategy
        self.batch_size = batch_size
        # internal fields
        self._iterator = self.learner.get_data_iterator()
        self.epoch = 0
        self.step = 0

    @classmethod
    def create(cls, ecc):
        ecc.show_config()

        # create the KnowledgeGraph
        kg = KnowledgeGraph.from_config(
            ecc.knowledge_graph_config)

        # create the neural
        nbp = ecc.neural_binary_predicate_config.instantiate(
            kg)

        # create learner
        learner = ecc.learner_config.instantiate(kg, nbp)

        # create the optimizer
        optimizer = ecc.optimizer_config.instantiate(nbp.parameters())

        # create trainer
        trainer = cls(kg=kg,
                      nbp=nbp,
                      learner=learner,
                      optimizer=optimizer,
                      **ecc.trainer_config.to_dict())
        return trainer

    def get_next_batch_input(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            print("train epoch", self.epoch)
            self._iterator = self.learner.get_data_iterator()
            batch = next(self._iterator)
        return batch

    def _compute_nce_loss(self, batch_output):
        pass

    def _compute_pairwise_loss(self, batch_output):
        pass

    def train_step(self):
        log = {}

        self.optimizer.zero_grad()

        batch_input = self.get_next_batch_input()
        batch_output = self.learner.forward(
            batch_input, self.num_negative_samples)

        if self.objective == 'nce':
            loss = self._compute_nce_loss(batch_output)

        elif self.objective == 'pairwise':
            loss = self._compute_pairwise_loss(batch_output)

        else:
            raise NotImplementedError(
                f"Unknown loss function {self.objective}")

        loss.backward()
        self.optimizer.step()
        self.step += 1

        log['loss'] = loss.item()
        log['epoch'] = self.epoch
        log['step'] = self.step

        return log
