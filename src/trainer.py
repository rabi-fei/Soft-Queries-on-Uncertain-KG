import logging
import os

import torch

from .evaluator import Evaluator
from .structure import KnowledgeGraph, NeuralBinaryPredicate
from .learner import Learner, LearnerForwardOutput
from .utils.recorder import TrainRecorder
from .utils.config import ExperimentConfigCollection


class Trainer:
    """
    Basic interface for training and evaluate the model
    basic objects
        - kg
        - nbp
        - learner
        - optimizer
    """
    def __init__(self,
                 kg: KnowledgeGraph,
                 nbp: NeuralBinaryPredicate,
                 learner: Learner,
                 optimizer: torch.optim.Optimizer,
                 evaluator: Evaluator,
                 recorder: TrainRecorder,
                 objective='nce',
                 margin=10,
                 k_nce=1,
                 num_neg_samples=1,
                 ns_strategy='lcwa',
                 batch_size=256,
                 num_steps=10000,
                 **kwargs):
        # important objects
        self.kg = kg
        self.nbp = nbp
        self.learner = learner
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.recorder = recorder
        # parameters
        self.objective = objective
        self.margin = margin
        self.num_neg_samples = num_neg_samples
        self.k_nce = k_nce
        self.ns_strategy = ns_strategy
        self.batch_size = batch_size
        self.num_steps = num_steps
        # internal fields
        self._iterator = None
        self.epoch = -1
        self.step = 0

    @classmethod
    def create(cls, ecc: ExperimentConfigCollection):
        ecc.show_config()


        # create the KnowledgeGraph
        logging.info(f"create the (observed) knowledge graph")
        logging.info(f"\t {ecc.knowledge_graph_config.to_dict()}")
        kg = KnowledgeGraph.from_config(ecc.knowledge_graph_config)
        logging.info(f"kg created")

        # create the neural
        logging.info(f"create the neural binary predicate")
        logging.info(f"\t {ecc.neural_binary_predicate_config.to_dict()}")
        nbp = ecc.neural_binary_predicate_config.instantiate(kg)
        logging.info(f"nbp created")

        # create learner
        logging.info(f"create the learner")
        logging.info(f"\t {ecc.learner_config.to_dict()}")
        learner = ecc.learner_config.instantiate(kg, nbp)
        logging.info(f"learner created")

        # create the optimizer
        logging.info(f"create the optimizer")
        logging.info(f"\t {ecc.optimizer_config.to_dict()}")
        optimizer = ecc.optimizer_config.instantiate(nbp.parameters())
        logging.info(f"optimizer created")

        # create the evaluator
        evaluator = Evaluator.create(ecc.evaluation_config, ecc.logdir, kg)

        # create the train recorder
        recorder = TrainRecorder(ecc.logdir)

        # create trainer
        trainer = cls(kg=kg,
                      nbp=nbp,
                      learner=learner,
                      optimizer=optimizer,
                      evaluator=evaluator,
                      recorder=recorder,
                      **ecc.trainer_config.to_dict())

        return trainer

    def get_next_batch_input(self):
        try:
            if self._iterator is None:
                raise StopIteration
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            print("train epoch", self.epoch)
            self._iterator = self.learner.get_data_iterator(
                batch_size=self.batch_size,
                shuffle=True)
            batch = next(self._iterator)
        return batch

    def _compute_nce_loss(self, batch_output: LearnerForwardOutput):
        loss = 0
        loss -= torch.log(batch_output.pos_prob.mean(-1))
        loss -= torch.log(1 - batch_output.neg_prob.mean(-1))
        return loss.mean()

    def _compute_pairwise_loss(self, batch_output):
        loss = self.margin
        loss += batch_output.neg_score.mean(-1) 
        loss -= batch_output.pos_score.mean(-1)
        loss = torch.relu(loss).mean()
        return loss

    def train_step(self):
        log = {}

        self.optimizer.zero_grad()

        batch_input = self.get_next_batch_input()
        batch_output = self.learner.forward(
            batch_input, self.num_neg_samples, self.margin)

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

    def run(self):
        self.evaluator.evaluate_nbp(self.nbp, self.step)
        while self.step < self.num_steps:
            log = self.train_step()
            self.recorder.write(log)
            if (self.step + 1) % self.evaluator.eval_every == 0:
                self.evaluator.evaluate_nbp(self.nbp, self.step)    
            
