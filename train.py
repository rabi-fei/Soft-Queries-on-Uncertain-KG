import argparse
import logging
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from efl import EFL
from model import KG, NeuralBinaryPredicate, TransE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='data/family-loss-0.05')
parser.add_argument('--log_dir', default='log')

parser.add_argument('--learning_method', default='efl')
parser.add_argument('--num_steps', default=10000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-4, type=float)

parser.add_argument('--cuda', default=-1, type=int)
parser.add_argument('--eval_every', default=1000, type=int)


def run_efl(finite_model_train: KG,
            finite_model_dev: KG,
            neural_model: NeuralBinaryPredicate,
            optimizer,
            num_steps,
            batch_size,
            eval_every,
            **kwargs):
    efl = EFL(finite_model_train, neural_model)
    for i in trange(num_steps):
        log = efl.learning_step(batch_size, optimizer)
        logging.info(f'EFL Step {i+1}|'
                     + '|'.join(f"{k}:{v}" for k, v in log.items())
                     + '\n')
        tb_writer.add_scalar(f'train/loss', log['loss'], global_step=i+1)

        if (i+1) % eval_every == 0:
            metric = neural_model.evaluate_triples(finite_model_dev.triples)
            for k in metric:
                tb_writer.add_scalar(f"dev/{k}", metric[k], global_step=(i+1))


def train_period(finite_model_train,
                 finite_model_dev,
                 neural_model,
                 optimizer,
                 learning_method='efl',
                 **kwargs):
    if learning_method == 'efl':
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

    finite_model_dev = KG.create(
        os.path.join(args.dataset_dir, 'dev.tsv'))

    # create the neural
    if torch.cuda.is_available() and args.cuda >= 0:
        device = f'cuda:{args.cuda}'
    else:
        device = 'cpu'

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
                 eval_every=args.eval_every)
