import yaml


class Config(object):
    def __init__(self):
        pass

    def load_from_yaml(self, file):
        config_dict = yaml.full_load(file)
        for k, v in config_dict.items():
            setattr(self, k, v)


class ConfigKG(Config):
    def __init__(self):


def dump_to_yaml(config):
    pass
