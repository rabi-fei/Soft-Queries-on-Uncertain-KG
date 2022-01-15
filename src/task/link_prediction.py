from collections import defaultdict
import gc
import sys

from tqdm import tqdm
import torch
import numpy as np

from .abstract_task import AbstractTask

from ..structure.abstract_models import KnowledgeGraph, NeuralBinaryPredicate


def show_objects():
    to_print = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                s = '\t'.join(
                    [str(id(obj)), str(sys.getrefcount(obj)), str(type(obj)), str(obj.size())])
                print(">>>>", s)
                to_print.append(s)
        except:
            pass
    to_print = sorted(to_print)
    for s in to_print:
        print(s)


class LinkPrediction(AbstractTask):
    def __init__(self, kg: KnowledgeGraph, observed_kg: KnowledgeGraph):
        self.kg = kg
        self.observed_kg = observed_kg
        self.device = self.observed_kg.device

    @classmethod
    def create(cls, filelist, observed_kg, device):
        kg = KnowledgeGraph.create(filelist, tensorize=False, device=device)
        return cls(kg, observed_kg)

    def evaluate_nbp(self, nbp: NeuralBinaryPredicate, init_batch_size=1000, prefix=""):
        # return self._evaluate_nbp(nbp, init_batch_size, prefix)
        if init_batch_size == 0:
            raise RuntimeError("zero batch size")

        oom = False
        try:
            return self._evaluate_nbp(nbp, init_batch_size, prefix)
        except RuntimeError as error:
            print(error)
            oom = True
        if oom:
            next_batch_size = init_batch_size//2
            return self.evaluate_nbp(nbp, next_batch_size, prefix)

    def _evaluate_nbp(self, nbp: NeuralBinaryPredicate, batch_size, prefix):
        # nbp.eval()
        filtered_rec = defaultdict(list)
        raw_rec = defaultdict(list)

        cand_id_ten = torch.arange(
            0,
            end=self.kg.num_entities,
            step=1,
            device=nbp.device)
        # raise RuntimeError
        cand_id_ten = torch.reshape(cand_id_ten, (1, -1))

        def cfn(batch):
            hl, rl, tl = [], [], []
            ot_coo_index, oh_coo_index = [[], []], [[], []]

            for i, (h, r, t) in enumerate(batch):
                hl.append(h)
                rl.append(r)
                tl.append(t)

                ot_list = self.observed_kg.hr2t[(h, r)]
                ot_coo_index[0] += [i] * len(ot_list)
                ot_coo_index[1] += ot_list

                oh_list = self.observed_kg.tr2h[(t, r)]
                oh_coo_index[0] += [i] * len(oh_list)
                oh_coo_index[1] += oh_list

            return [torch.tensor(l, device=nbp.device).view((-1, 1))
                    for l in [hl, rl, tl]] + [ot_coo_index, oh_coo_index]

        def link_pred_metric(rank, record, prefix):
            record[prefix + 'hit1'].extend((rank < 1).tolist())
            record[prefix + 'hit3'].extend((rank < 3).tolist())
            record[prefix + 'hit10'].extend((rank < 10).tolist())
            record[prefix + 'mrr'].extend((1/(1+rank)).tolist())
            record[prefix + 'mr'].extend((rank).tolist())

        with tqdm(self.kg.get_triple_dataloader(batch_size=batch_size,
                                           collate_fn=cfn),
                  desc=f"{prefix} Link Prediction Evaluation") as t:
            for head_id_ten, rel_id_ten, tail_id_ten, ot_idx, oh_idx in t:
                # predict head
                head_cand_score_tensor = nbp.batch_predicate_score(
                    [cand_id_ten, rel_id_ten, tail_id_ten])  # [num_cases, num_candidates]
                head_cand_score_tensor[oh_idx[0], oh_idx[1]] = - torch.inf

                head_score = torch.take_along_dim(input=head_cand_score_tensor,
                                                  indices=head_id_ten,
                                                  dim=1)

                head_rank = torch.sum(head_cand_score_tensor >
                                      head_score, -1).cpu().numpy()

                link_pred_metric(head_rank, filtered_rec, "")
                link_pred_metric(head_rank, filtered_rec, "head:")


                # [num_cases, num_candidates]
                # head_cand_sorted = torch.argsort(head_cand_score_tensor,
                #  dim=-1, descending=True)

                # assert (head_cand_sorted[torch.arange(
                #     len(head_rank)), head_rank] == _head_id_ten).all()
                # predict tail
                tail_cand_score_tensor = nbp.batch_predicate_score(
                    [head_id_ten, rel_id_ten, cand_id_ten])  # [num_cases, num_candidates]

                tail_cand_score_tensor[ot_idx[0], ot_idx[1]] = - torch.inf

                tail_score = torch.take_along_dim(input=tail_cand_score_tensor,
                                                  indices=tail_id_ten,
                                                  dim=1)

                tail_rank = torch.sum(tail_cand_score_tensor >
                                      tail_score, -1).cpu().numpy()

                filtered_rec['tail_hit1'].extend((tail_rank < 1).tolist())
                filtered_rec['tail_hit3'].extend((tail_rank < 3).tolist())
                filtered_rec['tail_hit10'].extend((tail_rank < 10).tolist())
                filtered_rec['tail_mrr'].extend((1/(1+tail_rank)).tolist())
                metric = {}
                for k in record:
                    metric[k] = np.mean(record[k])

                t.set_postfix(metric)

        return metric
