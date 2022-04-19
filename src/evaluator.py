import logging

from . import task
from .structure import KnowledgeGraph, NeuralBinaryPredicate
from .utils.config import EvaluationConfig
from .utils.recorder import EvalRecorder


class Evaluator:
    def __init__(self,
                 eval_every_step,
                 eval_every_epoch,
                 logdir,
                 task_dict,
                 device,
                 observed_kg,
                 **kwargs) -> None:
        self.eval_every_step = eval_every_step
        self.eval_every_epoch = eval_every_epoch

        self.task = {}
        self.task_recorder = {}

        for k, v in task_dict.items():
            logging.info(f"initialize task {k}")
            name = v['name']
            params = v['params']

            logging.info(f"\ttask type {name}: {params}")
            self.task[k] = task.get(name).create(
                observed_kg=observed_kg,
                device=device,
                **params)
            self.task_recorder[k] = EvalRecorder(logdir, k)
            logging.info(f"task {k} initialized")

        self.dev_task = kwargs.get('dev_task', None)
        self.dev_key = kwargs.get('dev_key', None)

    @classmethod
    def create(cls, eval_config: EvaluationConfig, logdir, observed_kg: KnowledgeGraph):
        logging.info("initalize evaluator")
        logging.info(eval_config.to_dict())

        return cls(observed_kg=observed_kg,
                   logdir=logdir,
                   **eval_config.to_dict())

    def evaluate_nbp(self, nbp: NeuralBinaryPredicate, global_step, global_epoch):
        for k in self.task:
            metric = self.task[k].evaluate_nbp(nbp, prefix=k)
            metric['global_step'] = global_step
            metric['global_epoch'] = global_epoch
            self.task_recorder[k].write(metric)

            if k == self.dev_task:
                return metric[self.dev_key]
