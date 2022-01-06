from abc import abstractclassmethod
import argparse
import logging
import os

import torch
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from src.learner import Learner
from src.structure import KnowledgeGraph, NeuralBinaryPredicate
from src.utils.config import ExperimentConfigCollection
from src.trainer import Trainer
from src.evaluator import Evaluator

parser = argparse.ArgumentParser()

# saved config override
parser.add_argument('--config_file', default='config/default_config.yaml')

# model argument
parser.add_argument('--cuda', type=int, required=False)
# finite model
parser.add_argument('--observed_finite_model_data_list',
                    default='data/family-loss-0.05/train.tsv', action='append')
# neural model argument
parser.add_argument('--neural_model', default='transe', type=str)

# learner arguments
parser.add_argument('--learner', default='I', choices=['E', 'I'])
parser.add_argument('--loss_function_type',
                    default='nce',
                    choices=['nce', 'pairwise'])
parser.add_argument('--k_nce', default=1, type=int)
parser.add_argument('--margin', default=10, type=float)
# elementary arguments
parser.add_argument('--efg_round', default=5, type=int)
parser.add_argument('--efg_rand_thr',
                    default=0.5, type=float,
                    help="thr=1 all on finite side, thr = 0 all on neural side")
parser.add_argument('--spoiler_neural_play_search_size',
                    default=5, type=int,
                    help="search size when play efg on the neural side")
parser.add_argument('--negative_subgraph_sampling', default=5, type=int,
                    help="number of negative samples for each batch")

# optimization arguments
parser.add_argument('--num_steps', default=50000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-2, type=float)

# output arguments
parser.add_argument('--log_dir', default='log/default')
parser.add_argument('--eval_every', default=200, type=int)

# evaluate tasks
parser.add_argument('--dev_task',
                    default='data/family-loss-0.05/dev.tsv', action='append')
parser.add_argument('--test_task',
                    default='data/family-loss-0.05/test.tsv', action='append')


def run_efl(finite_model_train: KnowledgeGraph,
            finite_model_dev: KnowledgeGraph,
            finite_model_test: KnowledgeGraph,
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


def run_lpl(finite_model_train: KnowledgeGraph,
            finite_model_dev: KnowledgeGraph,
            finite_model_test: KnowledgeGraph,
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

    ecc = ExperimentConfigCollection.from_yaml_file(args.config_file)
    ecc.show_config()

    # log folder
    os.makedirs(ecc.logdir, exist_ok=True)

    log_file = os.path.join(ecc.logdir, 'exp.log')
    logging.basicConfig(filename=log_file,
                        level=logging.INFO)

    # create trainer 
    trainer = Trainer.create(ecc)

    trainer.run()
