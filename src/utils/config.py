import torch
import yaml


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
        attrs = dir(self)
        ret = {k: attrs[k] for k in attrs if not k.startwith('__')}
        return ret


class KnowledgeGraphConfig(Config):
    default_kv = {'filelist': ['data/family-loss-0.05/train.tsv']}


class NeuralBinaryPredicateConfig(Config):
    default_kv = {'name': 'transe'}


class TrainerConfig(Config):
    default_kv = {'objective': 'nce',
                  'margin': 10,
                  'k_nce': 1,
                  'num_negative_samples': 1,
                  'ns_strategy': 'lcwa',
                  'batch_size': 256}

class OptimizerConfig(Config):
    default_kv = {'name': 'Adam'}

class LearnerConfig(Config):
    default_kv = {'name': 'I'}

class EvaluationConfig(Config):
    default_kv = {'eval_every': 200,
                  'dev_task_file': "",
                  'test_task_file': ""}

class ExperimentConfigCollection:
    components = {'knowledge_graph': KnowledgeGraphConfig,
                  'neural_binary_predicate': NeuralBinaryPredicateConfig,
                  'trainer': TrainerConfig,
                  'optimizer': OptimizerConfig,
                  'learner': LearnerConfig,
                  'evaluation': EvaluationConfig}

    def __init__(self, config_collection):
        self.logdir = config_collection.pop('logdir')

        self.cuda = config_collection.pop('cuda', -1)
        if torch.cuda.is_available() and self.cuda >= 0:
            self.device = f'cuda:{self.cuda}'
        else:
            self.device = 'cpu'

        for comp in self.components:
            config_instance = self.components[comp](
                config_dict=config_collection.pop('comp', {}))
            setattr(self, comp + '_config', config_instance)



def dump_to_yaml(config):
    pass


def load_config(path):
    Config.load_from_yaml
