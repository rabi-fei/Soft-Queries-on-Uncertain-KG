"""
A script to evaluate the models
# TODO
"""
import argparse

import torch
import yaml
from torch.utils.data import dataloader

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint')
parser.add_argument('--nbp_config') # TODO: one may detach the nbp config from the entire config
parser.add_argument('--evaluate_dataset_folder')

if __name__ == "__main__":
    print()