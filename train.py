from abc import abstractclassmethod
import argparse
import logging
import os
from random import shuffle

import torch
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from learner.elementary import ElementaryLearner
from learner.isomorphic import IsomorphicLearner
from src.model.abstract_models import KG, NeuralBinaryPredicate
from src.model.transe import TransE

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', default='data/family-loss-0.05/train.tsv')
parser.add_argument('--dev_data', default='data/family-loss-0.05/dev.tsv')
parser.add_argument('--test_data', default='data/family-loss-0.05/test.tsv')
parser.add_argument('--auto_index', default=False, action='store_true')

parser.add_argument('--log_dir', default='log')

parser.add_argument('--learning_method', default='efl')
# efl parameter
parser.add_argument('--efg_round', default=5, type=int)
parser.add_argument('--efg_rand_thr', default=0.5, type=float,
                    help="thr = 1 all on finite side, thr = 0 all on neural side")
parser.add_argument('--k_neural', default=5, type=int,
                    help="search size when play efg on the neural side")
parser.add_argument('--k_subgraph', default=5, type=int,
                    help="number of negative samples for each batch")
parser.add_argument('--k_nce', default=1, type=int)
parser.add_argument('--margin', default=10, type=float)

parser.add_argument('--num_steps', default=50000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--num_workers', default=1, type=int)

parser.add_argument('--eval_every', default=200, type=int)
parser.add_argument('--cuda', default=-1, type=int)


def run_efl(finite_model_train: KG,
            finite_model_dev: KG,
            finite_model_test: KG,
            neural_model: NeuralBinaryPredicate,
            optimizer,
            num_steps,
            batch_size,
            eval_every,
            efg_round,
            efg_rand_thr,
            k_neural,
            k_subgraph,
            k_nce,
            margin,
            **kwargs):
    print("running EFL")
    efl = ElementaryLearner(finite_model_train,
              neural_model,
              batch_size=batch_size,
              shuffle=True,
              efg_round=efg_round,
              efg_rand_thr=efg_rand_thr,
              k_neural=k_neural,
              k_subgraph=k_subgraph,
              k_nce=k_nce,
              margin=margin)
    with trange(num_steps) as t:
        for i in t:
            log = efl.learning_step(optimizer)
            logging.info(f'EFL Step {i+1}|'
                         + '|'.join(f"{k}:{v}" for k, v in log.items()))
            tb_writer.add_scalar(f'train/loss', log['loss'], global_step=i+1)
            t.set_postfix(log)

            if (i+1) % eval_every == 0:
                metric = neural_model.evaluate_kg(
                    finite_model_train)
                logging.info(f'EFL Eval Train {i+1}|'
                             + '|'.join(f"{k}:{v}" for k, v in metric.items()))
                for k in metric:
                    tb_writer.add_scalar(
                        f"train/{k}", metric[k], global_step=(i+1))

                metric = neural_model.evaluate_kg(
                    finite_model_dev)
                logging.info(f'EFL Eval Dev {i+1}|'
                             + '|'.join(f"{k}:{v}" for k, v in metric.items()))
                for k in metric:
                    tb_writer.add_scalar(
                        f"dev/{k}", metric[k], global_step=(i+1))

                metric = neural_model.evaluate_kg(
                    finite_model_test)
                logging.info(f'EFL Eval Test {i+1}|'
                             + '|'.join(f"{k}:{v}" for k, v in metric.items()))
                for k in metric:
                    tb_writer.add_scalar(
                        f"test/{k}", metric[k], global_step=(i+1))


def run_lpl(finite_model_train: KG,
            finite_model_dev: KG,
            finite_model_test: KG,
            neural_model: NeuralBinaryPredicate,
            optimizer,
            num_steps,
            batch_size,
            eval_every,
            **kwargs):
    print("running LPL")
    print("triple loader get")
    lpl = IsomorphicLearner(finite_model_train, neural_model,
              batch_size=batch_size, shuffle=True)
    with trange(num_steps) as t:
        for i in t:
            log = lpl.learning_step(optimizer)
            logging.info(f'LPL Step {i+1}|'
                         + '|'.join(f"{k}:{v}" for k, v in log.items()))
            tb_writer.add_scalar(f'train/loss', log['loss'], global_step=i+1)

            t.set_postfix(log)

            if (i+1) % eval_every == 0:
                metric = neural_model.evaluate_kg(finite_model_train)
                logging.info(f'LPL Eval Train {i+1}|'
                             + '|'.join(f"{k}:{v}" for k, v in metric.items()))
                for k in metric:
                    tb_writer.add_scalar(
                        f"train/{k}", metric[k], global_step=(i+1))

                metric = neural_model.evaluate_kg(finite_model_dev)
                logging.info(f'LPL Eval Dev {i+1}|'
                             + '|'.join(f"{k}:{v}" for k, v in metric.items()))
                for k in metric:
                    tb_writer.add_scalar(
                        f"dev/{k}", metric[k], global_step=(i+1))

                metric = neural_model.evaluate_kg(finite_model_test)
                logging.info(f'LPL Eval Test {i+1}|'
                             + '|'.join(f"{k}:{v}" for k, v in metric.items()))
                for k in metric:
                    tb_writer.add_scalar(
                        f"test/{k}", metric[k], global_step=(i+1))


def train_period(finite_model_train,
                 finite_model_dev,
                 finite_model_test,
                 neural_model,
                 optimizer,
                 learning_method='efl',
                 **kwargs):
    if learning_method == 'efl':
        print(kwargs)
        run_efl(finite_model_train,
                finite_model_dev,
                finite_model_test,
                neural_model,
                optimizer,
                **kwargs)
    if learning_method == 'lpl':
        run_lpl(finite_model_train,
                finite_model_dev,
                finite_model_test,
                neural_model,
                optimizer,
                **kwargs)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
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
        args.dev_data,
        auto_index=args.auto_index,
        num_entities=finite_model_train.num_entities,
        num_relations=finite_model_train.num_relations,
        device=device)

    finite_model_test = KG.create(
        args.test_data,
        auto_index=args.auto_index,
        num_entities=finite_model_train.num_entities,
        num_relations=finite_model_train.num_relations,
        device=device)

    # create the neural
    neural_model = TransE.create(finite_model_train,
                                 embedding_dim=600,
                                 device=device)

    # create the optimizer
    optimizer = torch.optim.Adam(neural_model.parameters(), lr=args.lr)

    train_period(finite_model_train=finite_model_train,
                 finite_model_dev=finite_model_dev,
                 finite_model_test=finite_model_test,
                 neural_model=neural_model,
                 optimizer=optimizer,
                 learning_method=args.learning_method,
                 num_steps=args.num_steps,
                 batch_size=args.batch_size,
                 eval_every=args.eval_every,
                 efg_round=args.efg_round,
                 efg_rand_thr=args.efg_rand_thr,
                 k_neural=args.k_neural,
                 k_subgraph=args.k_subgraph,
                 k_nce=args.k_nce,
                 margin=args.margin
                 )
