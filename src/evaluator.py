import logging

from . import task
from .structure import KnowledgeGraph, NeuralBinaryPredicate
from .utils.config import EvaluationConfig
from .utils.recorder import EvalRecorder


class Evaluator:
    def __init__(self, eval_every, logdir, task_dict, observed_kg, **kwargs) -> None:
        self.eval_every = eval_every
        self.task = {}
        self.task_recorder = {}
        for k, v in task_dict.items():
            logging.info(f"initialize task {k}")

            name = v['name']
            params = v['params']
            logging.info(f"\ttask type {name}: {params}")
            self.task[k] = task.get(name)(observed_kg=observed_kg, **params)
            self.task_recorder[k] = EvalRecorder(logdir, k)
            logging.info(f"task {k} initialized")

    @classmethod
    # TODO: make the logging system
    def create(cls, eval_config: EvaluationConfig, logdir, observed_kg: KnowledgeGraph):
        logging.info("initalize evaluator")
        logging.info(eval_config.to_dict())

        return cls(observed_kg=observed_kg, 
                   logdir=logdir, 
                   **eval_config.to_dict())

    def evaluate_nbp(self, nbp: NeuralBinaryPredicate, global_step):
        for k in task:
            metric = self.task[k].evaluate_nbp(nbp)
            metric['global_step'] = global_step
            self.task_recorder[k].write(metric)

        
