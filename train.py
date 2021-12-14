from abc import abstractclassmethod
import argparse
import logging
import os
from random import shuffle

import torch
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from learner.efl import EFL
from learner.lpl import LPL
from model.abstract_models import KG, NeuralBinaryPredicate
from model.transe import TransE

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', default='data/family-loss-0.05/train.tsv')
parser.add_argument('--dev_data', default='data/family-loss-0.05/dev.tsv')
parser.add_argument('--auto_index', default=True, type=bool)

parser.add_argument('--log_dir', default='log')

parser.add_argument('--learning_method', default='efl')
parser.add_argument('--efl_round', default=5, type=int)
parser.add_argument('--num_steps', default=50000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--num_workers', default=1, type=int)

parser.add_argument('--cuda', default=-1, type=int)
parser.add_argument('--eval_every', default=1000, type=int)


def run_efl(finite_model_train: KG,
            finite_model_dev: KG,
            neural_model: NeuralBinaryPredicate,
            optimizer,
            efl_round,
            num_steps,
            batch_size,
            eval_every,
            num_workers,
            **kwargs):
    print("running EFL")
    efl = EFL(finite_model_train,
              neural_model,
              round=efl_round,
              batch_size=batch_size,
              num_workers=num_workers,
              shuffle=True)
    with trange(num_steps) as t:
        for i in t:
            log = efl.learning_step(optimizer)
            logging.info(f'EFL Step {i+1}|'
                         + '|'.join(f"{k}:{v}" for k, v in log.items()))
            tb_writer.add_scalar(f'train/loss', log['loss'], global_step=i+1)
            t.set_postfix(log)

            if (i+1) % eval_every == 0:
                metric = neural_model.evaluate_kg(
                    finite_model_dev)
                logging.info(f'EFL Eval {i+1}|'
                             + '|'.join(f"{k}:{v}" for k, v in metric.items()))
                for k in metric:
                    tb_writer.add_scalar(
                        f"dev/{k}", metric[k], global_step=(i+1))


def run_lpl(finite_model_train: KG,
            finite_model_dev: KG,
            neural_model: NeuralBinaryPredicate,
            optimizer,
            num_steps,
            batch_size,
            eval_every,
            ** kwargs):
    print("running LPL")
    print("triple loader get")
    lpl = LPL(finite_model_train, neural_model,
              batch_size=batch_size, shuffle=True)
    with trange(num_steps) as t:
        for i in t:
            log = lpl.learning_step(optimizer)
            logging.info(f'LPL Step {i+1}|'
                         + '|'.join(f"{k}:{v}" for k, v in log.items()))
            tb_writer.add_scalar(f'train/loss', log['loss'], global_step=i+1)

            t.set_postfix(log)

            if (i+1) % eval_every == 0:
                metric = neural_model.evaluate_kg(finite_model_dev)
                logging.info(f'LPL Eval {i+1}|'
                             + '|'.join(f"{k}:{v}" for k, v in metric.items()))
                for k in metric:
                    tb_writer.add_scalar(
                        f"dev/{k}", metric[k], global_step=(i+1))


def train_period(finite_model_train,
                 finite_model_dev,
                 neural_model,
                 optimizer,
                 learning_method='efl',
                 **kwargs):
    if learning_method == 'efl':
        print(kwargs)
        run_efl(finite_model_train,
                finite_model_dev,
                neural_model,
                optimizer,
                **kwargs)
    if learning_method == 'lpl':
        run_lpl(finite_model_train,
                finite_model_dev,
                neural_model,
                optimizer,
                **kwargs)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    args = parser.parse_args()
    # log folder
    os.makedirs(args.log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    log_file = os.path.join(args.log_dir, 'exp.log')
    logging.basicConfig(filename=log_file,
                        level=logging.INFO)

    if torch.cuda.is_available() and args.cuda >= 0:
        device = f'cuda:{args.cuda}'
    else:
        device = 'cpu'

    # create the KG
    finite_model_train = KG.create(
        args.train_data, auto_index=args.auto_index, device=device)

    finite_model_dev = KG.create(
        args.dev_data, auto_index=args.auto_index, device=device)

    # create the neural
    neural_model = TransE.create(finite_model_train,
                                 embedding_dim=600,
                                 device=device)

    # create the optimizer
    optimizer = torch.optim.Adam(neural_model.parameters(), lr=args.lr)

    train_period(finite_model_train=finite_model_train,
                 finite_model_dev=finite_model_dev,
                 neural_model=neural_model,
                 optimizer=optimizer,
                 learning_method=args.learning_method,
                 num_steps=args.num_steps,
                 batch_size=args.batch_size,
                 eval_every=args.eval_every,
                 efl_round=args.efl_round,
                 num_workers=args.num_workers)
