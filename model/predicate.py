from abc import abstractmethod
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from model.kg import KG
from model.model_utils import triples_to_tensors


class NeuralBinaryPredicate:
    def __init__(self):
        pass

    @abstractmethod
    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        This method computes the score for the triple given the head, tail and
        relation embedding. The higher score means more likely to be a predicate.
        Inputs:
            Three embeddings are in the shape [batch_size|1, embed_dim]
        Returns:
            The tensor of scores in the shape [batch_size|1]
        """
        pass

    def batch_pred_score(self,
                         head_id_ten: torch.Tensor,
                         rel_id_ten: torch.Tensor,
                         tail_id_ten: torch.Tensor) -> torch.Tensor:
        """
        This method computes the scores for the triple. Three tensors are in the
        same shape, i.e. [batch_size]
        It returns the same size of predicate scores.
        """
        head_emb = self.entity_embedding(head_id_ten)
        rel_emb = self.relation_embedding(rel_id_ten)
        tail_emb = self.entity_embedding(tail_id_ten)
        return self.embedding_score(head_emb, rel_emb, tail_emb)

    def sort_triples_by_scores(self, triples, assending=True):
        """
        One and only one tensor of the three input tensors is None.
        This function returns the id with the least score.
        It can be interpreted as a sentence.
        """
        h, r, t = triples_to_tensors(triples, self.device)
        triple_score_tensor = self.batch_pred_score(h, r, t)
        triple_scores = triple_score_tensor.detach().cpu().numpy().tolist()
        triple_with_score = sorted(zip(triples, triple_scores),
                                   reverse=not assending,
                                   key=lambda x: x[1])
        ret = []
        for triple, score in triple_with_score:
            ret.append(triple)
        return ret

    @abstractmethod
    def compute_triple_loss(self, pos_triples, neg_triples, **kwargs):
        pass

    def evaluate_triples(self, triples, **kwargs):
        record = defaultdict(list)

        def record_rank(r, title):
            if r == 0:
                record[f'{title}.hit1'].append(1)
            else:
                record[f'{title}.hit1'].append(0)

            if r < 3:
                record[f'{title}.hit3'].append(1)
            else:
                record[f'{title}.hit3'].append(0)

            if r < 10:
                record[f'{title}.hit10'].append(1)
            else:
                record[f'{title}.hit10'].append(0)

            record[f'{title}.mrr'].append(1/(1+r))

        for h, r, t in tqdm(triples):
            # eval default head
            _eval_triples = [(e, r, t) for e in range(self.num_entities)]
            sorted_eval_triples = self.sort_triples_by_scores(
                _eval_triples, assending=False)
            rank = None
            for i, (_h, _r, _t) in enumerate(sorted_eval_triples):
                if _h == h:
                    rank = i
                    record_rank(rank, 'head')
                    break

            # eval default rel
            _eval_triples = [(h, _r, t) for _r in range(self.num_relations)]
            sorted_eval_triples = self.sort_triples_by_scores(
                _eval_triples, assending=False)
            rank = None
            for i, (_h, _r, _t) in enumerate(sorted_eval_triples):
                if _r == r:
                    rank = i
                    record_rank(rank, 'rel')
                    break

            # eval default tail
            _eval_triples = [(h, r, e) for e in range(self.num_entities)]
            sorted_eval_triples = self.sort_triples_by_scores(
                _eval_triples, assending=False)
            rank = None
            for i, (_h, _r, _t) in enumerate(sorted_eval_triples):
                if _t == t:
                    rank = i
                    record_rank(rank, 'tail')
                    break
        metric = {}
        for k in record:
            metric[k] = np.mean(record[k])
        return metric
