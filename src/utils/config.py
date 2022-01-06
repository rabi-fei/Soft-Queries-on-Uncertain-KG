import os
from abc import abstractmethod

import torch
import yaml

from datetime import datetime

from src import learner, structure


class Config:
    default_kv = {}

    def __init__(self, config_dict) -> None:
        self.params = {}

        for k in self.default_kv:
            v = config_dict.pop(k, self.default_kv[k])
            setattr(self, k, v)

        # use self.params to absorb the non-named kvs
        if config_dict is not None:
            self.params.update(config_dict)

    def to_dict(self):
        return vars(self)


class KnowledgeGraphConfig(Config):
    default_kv = {'filelist': 
    ['datasets-knowledge-embedding/COUNTRIES-S1/edges_as_id_train.tsv']}

    def __init__(self, config_dict={}) -> None:
        self.filelist = []
        super().__init__(config_dict)


class TrainerConfig(Config):
    default_kv = {'objective': 'nce',
                  'margin': 10,
                  'k_nce': 1,
                  'num_neg_samples': 1,
                  'ns_strategy': 'lcwa',
                  'batch_size': 256}

    def __init__(self, config_dict={}) -> None:
        self.objective = ""
        self.margin = -1
        self.k_nce = -1
        self.num_neg_samples = -1
        self.ns_strategy = ""
        self.batch_size = -1
        super().__init__(config_dict)


class EvaluationConfig(Config):
    default_kv = {'eval_every': 200,
                  'task_dict': {
                      'dev': {"name": "LinkPrediction",
                              "params": {"filelist": []}},
                      'test': {"name": "LinkPrediction",
                               "params": {"filelist": []}},
                  }}

    def __init__(self, config_dict={}) -> None:
        self.eval_every = 9999999
        self.task_dict = {}
        super().__init__(config_dict)


class ConfigWithChoice(Config):
    @abstractmethod
    def instantiate(self):
        pass


class NeuralBinaryPredicateConfig(ConfigWithChoice):
    default_kv = {'name': 'TransE',
                  'params': {'embedding_dim': 600}}

    def __init__(self, config_dict={}) -> None:
        self.name = ""
        self.params = {}
        super().__init__(config_dict)

    def instantiate(self, knowledge_graph):
        return structure.get(self.name)(
            num_entities=knowledge_graph.num_entities,
            num_relations=knowledge_graph.num_relations,
            device=self.device,
            **self.params)


class OptimizerConfig(ConfigWithChoice):
    default_kv = {'name': 'Adam',
                  'params': {"lr": 1e-2}}

    def __init__(self, config_dict={}) -> None:
        self.name = ""
        self.params = {}
        super().__init__(config_dict)

    def instantiate(self, parameters):
        return getattr(torch.optim, self.name)(parameters, **self.params)


class LearnerConfig(ConfigWithChoice):
    default_kv = {'name': 'I',
                  'params': {
                      'efg_round': 5,
                      'efg_mode': 'random',
                      'efg_rand_thr': 0.5,
                      'neural_act_search_size': 10
                  }
                  }

    def __init__(self, config_dict={}) -> None:
        self.name = ""
        self.params = {}
        super().__init__(config_dict)

    def instantiate(self, kg, nbp):
        return learner.get(self.name)(kg, nbp, **self.params)


class ExperimentConfigCollection:
    components = {'knowledge_graph': KnowledgeGraphConfig,
                  'neural_binary_predicate': NeuralBinaryPredicateConfig,
                  'trainer': TrainerConfig,
                  'optimizer': OptimizerConfig,
                  'learner': LearnerConfig,
                  'evaluation': EvaluationConfig}

    def __init__(self, config_collection):
        self.knowledge_graph_config = KnowledgeGraphConfig()
        self.neural_binary_predicate_config = NeuralBinaryPredicateConfig()
        self.trainer_config = TrainerConfig()
        self.optimizer_config = OptimizerConfig()
        self.learner_config = LearnerConfig()
        self.evaluation_config = EvaluationConfig()

        self.logdir = "_".join(
            [config_collection.pop('logdir'),
             datetime.strftime(
                datetime.now(),
                "%Y-%m-%d_%H:%M:%S")])

        self.cuda = config_collection.pop('cuda', -1)
        if torch.cuda.is_available() and self.cuda >= 0:
            self.device = f'cuda:{self.cuda}'
        else:
            self.device = 'cpu'

        for comp in self.components:
            config_instance = self.components[comp](
                config_dict=config_collection.pop(comp, {}))
            setattr(self, comp+'_config', config_instance)
            setattr(
                getattr(self, comp+'_config'),
                'device',
                self.device
            )

    @classmethod
    def from_yaml_file(cls, filename):
        with open(filename, 'rt') as f:
            config_collection = yaml.full_load(f)
        print(config_collection)
        return cls(config_collection=config_collection)

    def show_config(self):
        for comp in self.components:
            print('-' * 10)
            print(comp)
            print(getattr(self, comp + '_config').to_dict())
