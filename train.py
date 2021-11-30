import argparse
import logging
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from efl import EFL
from model import KG, NeuralBinaryPredicate, TransE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='data/family-loss-0.1')
parser.add_argument('--learning_method', default='efl')
parser.add_argument('--log_dir', default='log')
parser.add_argument('--num_steps', default=10000, type=int)


def run_efl(finite_model, neural_model, optimizer, num_steps):
    efl = EFL(finite_model, neural_model)
    for i in trange(num_steps):
        log = efl.learning_step(1, optimizer)
        logging.info(f'EFL Step {i+1}|'
                     + '|'.join(f"{k}:{v}" for k, v in log.items())
                     + '\n')
        tb_writer.add_scalar(f'train/loss', log['loss'], global_step=i+1)


def train_period(finite_model,
                 neural_model,
                 optimizer,
                 num_steps=1000,
                 learning_method='efl'):
    if learning_method == 'efl':
        run_efl(finite_model, neural_model, optimizer, num_steps=num_steps)


if __name__ == "__main__":
    args = parser.parse_args()
    # log folder
    os.makedirs(args.log_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=args.log_dir)
    log_file = os.path.join(args.log_dir, 'exp.log')
    logging.basicConfig(filename=log_file,
                        level=logging.INFO)

    # create the KG
    finite_model_train = KG.create(
        os.path.join(args.dataset_dir, 'train.tsv'))

    # create the neural log_dir
    neural_model = TransE.create(finite_model_train)

    # create the optimizer
    optimizer = torch.optim.Adam(neural_model.parameters())

    train_period(finite_model=finite_model_train,
                 neural_model=neural_model,
                 optimizer=optimizer,
                 learning_method=args.learning_method,
                 num_steps=args.num_steps)
