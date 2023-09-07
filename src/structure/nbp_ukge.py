import torch
import numpy as np
from torch import nn

from .neural_binary_predicate import NeuralBinaryPredicate

class TransE(nn.Module, NeuralBinaryPredicate):
    def __init__(self, num_entities, num_relations, embedding_dim, p, margin, scale, device, **kwargs):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.scale = margin
        self.scale = scale
        self.p = p
        self._entity_embedding = nn.Embedding(num_entities, embedding_dim, max_norm=1)
        nn.init.xavier_uniform_(self._entity_embedding.weight)
        self._relation_embedding = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self._relation_embedding.weight)

    @property
    def entity_embedding(self):
        return self._entity_embedding.weight

    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        board castable for the last dimension
        """
        return - torch.norm(head_emb + rel_emb - tail_emb, p=self.p, dim=-1)

    def score2truth_value(self, score):
        return torch.sigmoid(self.scale + score * self.scale)

    def estimate_tail_emb(self, head_emb, rel_emb):
        return head_emb + rel_emb

    def estimate_head_emb(self, tail_emb, rel_emb):
        return tail_emb - rel_emb

    def estiamte_rel_emb(self, head_emb, tail_emb):
        return tail_emb - head_emb

    def get_relation_emb(self, relation_id_or_tensor):
        rel_id = torch.tensor(relation_id_or_tensor, device=self.device)
        return self._relation_embedding(rel_id)

    def get_entity_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)

    def get_tail_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)

    def get_all_entity_rankings(self, batch_embedding_input):
        batch_embedding_input = batch_embedding_input.unsqueeze(-2)
        # batch_size, all_candidates
        # ranking score should be the higher the better
        # ranking_score[entity_id] = the score of {entity_id}
        ranking_score = - torch.norm(batch_embedding_input - self.entity_embedding, p=self.p, dim=-1)
        # ranked_entity_ids[ranking] = {entity_id} at the {rankings}-th place
        ranked_entity_ids = torch.argsort(ranking_score, dim=-1, descending=True)
        # entity_rankings[entity_id] = {rankings} of the entity
        entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
        return entity_rankings

class UKGE_rect_numpy(object):
    #Just for inference
    def __init__(self, emb_dim, entity_num, relation_num):
        self.entity_embedding = None
        self.relation_embedding = None
        self.w = None
        self.b = None

    def load_params(self, params_dict):
        self.entity_embedding = params_dict["entity_embedding"]
        self.relation_embedding = params_dict["relation_embedding"]
        self.w = params_dict["w"]
        self.b = params_dict["b"]

    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    def get_score_predict(self, h_batch, r_batch, t_batch): #indices
        h_emb = self.entity_embedding[h_batch, :]
        t_emb = self.entity_embedding[t_batch, :]
        r_emb = self.relation_embedding[r_batch, :]

        scores = self.w * (h_emb * t_emb * r_emb).sum(1) + self.b

        return scores

    def get_mse(self, test_triples, epoch=0):
        N = test_triples.shape[0]

        # existing triples
        # (score - w)^2
        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]
        scores = self.get_score_predict(h_batch, r_batch, t_batch)
        scores = self.bound_score(scores)
        mse = np.sum(np.square(scores - w_batch))

        mse = mse / N

        return mse