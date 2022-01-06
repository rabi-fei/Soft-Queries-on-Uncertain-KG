from collections import defaultdict

from tqdm import tqdm
import torch
from torch.nn.functional import one_hot
import numpy as np

from .abstract_task import AbstractTask

from ..structure.abstract_models import KnowledgeGraph, NeuralBinaryPredicate


class LinkPrediction(AbstractTask):
    def __init__(self, kg: KnowledgeGraph, observed_kg: KnowledgeGraph):
        self.kg = kg
        self.observed_kg = observed_kg
        self.device = self.observed_kg.device

    @classmethod
    def create(cls, filelist, observed_kg):
        kg = KnowledgeGraph.create(filelist)
        return cls(kg, observed_kg)

    def evaluate_nbp(self, nbp: NeuralBinaryPredicate, init_batch_size=1000):
        try:
            return self._evaluate_nbp(nbp, init_batch_size)
        except:
            return self.evaluate_nbp(nbp, init_batch_size//2)

    def _evaluate_nbp(self, nbp: NeuralBinaryPredicate, batch_size):
        record = defaultdict(list)
        _cand_id_ten = torch.arange(
            0,
            end=nbp.entity_embedding.weight.shape[0],
            step=1,
            device=nbp.device)
        cand_id_ten = torch.reshape(_cand_id_ten, (1, -1))

        def cfn(batch):
            hl, rl, tl, otml, ohml = [], [], [], [], []
            for h, r, t in batch:
                hl.append(h)
                rl.append(r)
                tl.append(t)

                ot_list = self.observed_kg.hr2t[(h, r)]
                ot_id_ten = torch.tensor(ot_list)
                otm = one_hot(
                    ot_id_ten,
                    num_classes=self.observed_kg.num_entities,
                    device=self.device).reshape(1, -1)
                otml.append(otm)

                oh_list = self.observed_kg.tr2h[(t, r)]
                oh_id_ten = torch.tensor(oh_list)
                ohm = one_hot(
                    oh_id_ten,
                    num_classes=self.observed_kg.num_entities,
                    device=self.device).reshape(1, -1)
                ohml.append(ohm)

            return [torch.tensor(l, device=self.device) for l in [hl, rl, tl]] + \
                   [torch.stack(otml), torch.stack(ohml)]

        with tqdm(self.kg.get_triple_dataloader(batch_size=batch_size,
                                                collate_fn=cfn),
                  desc="Link Prediction Evaluation") as t:
            for _head_id_ten, _rel_id_ten, _tail_id_ten, ot_mask, oh_mask in t:
                # oh_mask: observed head mask
                # ot_mask: observed tail mask
                num_cases = len(_rel_id_ten)

                head_id_ten = torch.reshape(_head_id_ten, (num_cases, 1))
                rel_id_ten = torch.reshape(_rel_id_ten, (num_cases, 1))
                tail_id_ten = torch.reshape(_tail_id_ten, (num_cases, 1))

                # predict head
                head_cand_score_tensor = nbp.batch_pred_score(
                    [cand_id_ten, rel_id_ten, tail_id_ten])  # [num_cases, num_candidates]

                head_score = torch.take_along_dim(input=head_cand_score_tensor,
                                                  indices=head_id_ten,
                                                  dim=1)

                first_idx = torch.arange(num_cases, device=self.device)
                head_score[first_idx, oh_mask] = head_score - 1

                head_rank = torch.sum(head_cand_score_tensor >
                                      head_score, -1).cpu().numpy()

                record['head_hit1'].extend((head_rank < 1).tolist())
                record['head_hit3'].extend((head_rank < 3).tolist())
                record['head_hit10'].extend((head_rank < 10).tolist())
                record['head_mrr'].extend((1/(1+head_rank)).tolist())

                # [num_cases, num_candidates]
                # head_cand_sorted = torch.argsort(head_cand_score_tensor,
                #  dim=-1, descending=True)

                # assert (head_cand_sorted[torch.arange(
                #     len(head_rank)), head_rank] == _head_id_ten).all()
                # predict tail
                tail_cand_score_tensor = self.batch_pred_score(
                    head_id_ten, rel_id_ten, cand_id_ten)  # [num_cases, num_candidates]

                tail_score = torch.take_along_dim(input=tail_cand_score_tensor,
                                                  indices=tail_id_ten,
                                                  dim=1)

                tail_rank = torch.sum(tail_cand_score_tensor >
                                      tail_score, -1).cpu().numpy()

                record['tail_hit1'].extend((tail_rank < 1).tolist())
                record['tail_hit3'].extend((tail_rank < 3).tolist())
                record['tail_hit10'].extend((tail_rank < 10).tolist())
                record['tail_mrr'].extend((1/(1+tail_rank)).tolist())
                metric = {}
                for k in record:
                    metric[k] = np.mean(record[k])

                t.set_postfix(metric)

        return metric
