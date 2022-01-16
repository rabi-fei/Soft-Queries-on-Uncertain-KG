import logging
import os

from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from src.utils.config import ExperimentConfigCollection
from src.trainer import Trainer


if __name__ == "__main__":
    parser = ExperimentConfigCollection.create_argument_parser()
    args = parser.parse_args()
    ecc = ExperimentConfigCollection.from_args(args)
    ecc.show_config()
    # exit()

    # log folder
    os.makedirs(ecc.logdir, exist_ok=True)

    log_file = os.path.join(ecc.logdir, 'exp.log')
    logging.basicConfig(filename=log_file,
                        level=logging.INFO)

    # create trainer 
    trainer = Trainer.create(ecc)

    trainer.run()
