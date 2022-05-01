from .abstract_task import AbstractTask
import json

from torch.utils.data import DataLoader

from ..utils.data import collate_qaa_into_first_order_formula

class QueryAnsweringTV(AbstractTask):
    def __init__(self, qaafile, **dataloader_kwargs) -> None:
        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)

        self.lstr_dataloader = {}
        self.dataloader_kwargs = dataloader_kwargs

    def construct_dataloaders(self):
        for lstr, qaa in self.lstr_qaa.items():
            self.lstr_dataloader[lstr] = DataLoader(qaa,
                collate_fn=lambda batch: collate_qaa_into_first_order_formula(lstr=lstr, batch=batch),
                **self.dataloader_kwargs)

    def evaluate_nbp(self, nbp) -> Dict:
        return super().evaluate_nbp(nbp)