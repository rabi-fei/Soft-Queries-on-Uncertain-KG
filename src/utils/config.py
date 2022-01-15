import argparse
import json
import os
from abc import abstractmethod

import torch
import yaml

from datetime import datetime


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
                  ['datasets-knowledge-embedding/COUNTRIES-S1/edges_as_id_train.tsv'],
                  'tensorize': True}

    def __init__(self, config_dict={}) -> None:
        self.filelist = []
        super().__init__(config_dict)


class TrainerConfig(Config):
    default_kv = {'objective': 'nce',
                  'margin': 10,
                  'k_nce': 1,
                  'num_neg_samples': 1,
                  'ns_strategy': 'lcwa',
                  'batch_size': 256,
                  'num_steps': 10000,
                  'num_epochs': 1000}

    def __init__(self, config_dict={}) -> None:
        self.objective = ""
        self.margin = -1
        self.k_nce = -1
        self.num_neg_samples = -1
        self.ns_strategy = ""
        self.batch_size = -1
        self.num_epochs = -1
        super().__init__(config_dict)


class EvaluationConfig(Config):
    default_kv = {'eval_every_step': 200,
                  'eval_every_epoch': 5,
                  'task_dict': {
                      'dev': {"name": "LinkPrediction",
                              "params": {"filelist": []}},
                      'test': {"name": "LinkPrediction",
                               "params": {"filelist": []}},
                  }}

    def __init__(self, config_dict={}) -> None:
        self.eval_every_step = 9999999
        self.eval_every_epoch = 9999999
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
        from src import structure
        return structure.get(self.name).create(
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
        from src import learner
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

        os.makedirs(self.logdir, exist_ok=True)
        with open(os.path.join(self.logdir, 'config.json'), 'wt') as f:
            json.dump(config_collection, f, indent=2)

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
    def from_args(cls, args):
        override_dict = vars(args)
        filename = override_dict.pop('config')
        with open(filename, 'rt') as f:
            config_collection = yaml.full_load(f)
        for k, v in override_dict.items():
            if v:
                *key_chain, final_key = k.split('.')
                pointer = config_collection
                for _k in key_chain:
                    pointer = pointer[_k]
                pointer[final_key] = v
        return cls(config_collection=config_collection)

    @classmethod
    def create_argument_parser(cls):
        parser = argparse.ArgumentParser()
        for comp_name in cls.components:
            comp_cls = cls.components[comp_name]

            linear_dict = dict()

            def _linearize(_d, prefix=None):
                for _k, _v in _d.items():
                    if prefix is not None:
                        key = prefix + '.' + _k
                    else:
                        key = _k

                    if isinstance(_v, dict):
                        _linearize(_v, key)
                    else:
                        linear_dict[key] = _v

            _linearize(comp_cls.default_kv)

            for k, v in linear_dict.items():
                if isinstance(v, list):
                    parser.add_argument(f"--{comp_name}.{k}",
                                        action='append', required=False)
                else:
                    parser.add_argument(f"--{comp_name}.{k}",
                                        type=type(v), required=False)

        parser.add_argument('--cuda', type=int)
        parser.add_argument('--logdir', type=str)
        parser.add_argument(
            '--config', default='config/default_config.yaml', type=str)
        return parser

    def show_config(self):
        for comp in self.components:
            print('-' * 10)
            print(comp)
            print(getattr(self, comp + '_config').to_dict())
