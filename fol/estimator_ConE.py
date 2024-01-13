from typing import List
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .appfoq import (AppFOQEstimator, IntList, find_optimal_batch,
                     inclusion_sampling)



pi = 3.14159265358979323846


def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y


def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y


class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale


class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ConeProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)

        self.layer_alpha = nn.Linear(self.entity_dim, self.entity_dim)
        self.layer_beta = nn.Linear(self.entity_dim, self.entity_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

        nn.init.xavier_uniform_(self.layer_alpha.weight)
        nn.init.xavier_uniform_(self.layer_beta.weight)

    def forward(self, source_embedding_axis, source_embedding_arg, r_embedding_axis, 
    r_embedding_arg, a_embedding_axis, a_embedding_arg, b_embedding_axis, b_embedding_arg):

        ab_emb_axis = self.layer_alpha(a_embedding_axis) + self.layer_beta(b_embedding_axis)
        ab_emb_args = self.layer_alpha(a_embedding_arg) + self.layer_beta(b_embedding_arg)
        x = torch.cat(
            [source_embedding_axis + r_embedding_axis + ab_emb_axis,
             source_embedding_arg + r_embedding_arg + ab_emb_args],
        dim=-1)
#        x = torch.cat(
#            [source_embedding_axis + r_embedding_axis,
#             source_embedding_arg + r_embedding_arg],
#        dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg = torch.chunk(x, 2, dim=-1)
        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)
        return axis_embeddings, arg_embeddings


class ConeIntersection(nn.Module):
    def __init__(self, dim, drop):
        super(ConeIntersection, self).__init__()
        self.dim = dim
        self.layer_axis1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_arg1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_axis2 = nn.Linear(self.dim, self.dim)
        self.layer_arg2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_axis1.weight)
        nn.init.xavier_uniform_(self.layer_arg1.weight)
        nn.init.xavier_uniform_(self.layer_axis2.weight)
        nn.init.xavier_uniform_(self.layer_arg2.weight)

        self.drop = nn.Dropout(p=drop)

    def forward(self, axis_embeddings, arg_embeddings, **kwargs):
        logits = torch.cat([axis_embeddings - arg_embeddings, axis_embeddings + arg_embeddings], dim=-1)
        axis_layer1_act = F.relu(self.layer_axis1(logits))

        axis_attention = F.softmax(self.layer_axis2(axis_layer1_act), dim=0)

        x_embeddings = torch.cos(axis_embeddings)
        y_embeddings = torch.sin(axis_embeddings)
        x_embeddings = torch.sum(axis_attention * x_embeddings, dim=0)
        y_embeddings = torch.sum(axis_attention * y_embeddings, dim=0)

        # when x_embeddings are very closed to zero, the tangent may be nan
        # no need to consider the sign of x_embeddings
        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        axis_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        axis_embeddings[indicator_two] += pi
        axis_embeddings[indicator_three] -= pi

        # DeepSets
        arg_layer1_act = F.relu(self.layer_arg1(logits))
        arg_layer1_mean = torch.mean(arg_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arg2(arg_layer1_mean))

        arg_embeddings = self.drop(arg_embeddings)
        arg_embeddings, _ = torch.min(arg_embeddings, dim=0)
        arg_embeddings *= gate

        return axis_embeddings, arg_embeddings


class ConeNegation(nn.Module):
    def __init__(self):
        super(ConeNegation, self).__init__()

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0
        new_axis_embedding = torch.zeros_like(axis_embedding)
        new_axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        new_axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi

        arg_embedding = pi - arg_embedding

        return new_axis_embedding, arg_embedding


class ConEstimator(AppFOQEstimator):
    name = "ConE"

    def __init__(self, n_entity, n_relation, entity_dim, relation_dim, hidden_dim, num_layer,
                 negative_sample_size, gamma, device, center_reg=None, drop=0.):
        super(ConEstimator, self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.negative_size = negative_sample_size
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.device = device

        self.loss = torch.nn.MSELoss()

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.w = nn.Parameter(
            torch.Tensor([2.0]),
            requires_grad=True
        )

        self.b = nn.Parameter(
            torch.Tensor([0.0]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / entity_dim]),
            requires_grad=False
        )

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.cen = center_reg

        # entity only have axis but no arg
        self.entity_embeddings = nn.Embedding(n_entity, self.entity_dim)  # axis for entities
        self.angle_scale = AngleScale(self.embedding_range.item())  # scale axis embeddings to [-pi, pi]

        self.modulus = nn.Parameter(torch.Tensor([1.5 * self.embedding_range.item()]), requires_grad=False)


        #self.f = nn.LeakyReLU(0.1)
        self.f = nn.ReLU()
        #self.loss = torch.nn.HuberLoss('mean', 0.1)

        self.axis_scale = 1.0
        self.arg_scale = 1.0

        nn.init.uniform_(
            tensor=self.entity_embeddings.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embeddings = nn.Embedding(n_relation, self.relation_dim)
        nn.init.uniform_(
            tensor=self.relation_embeddings.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.projection_net = ConeProjection(self.entity_dim, hidden_dim, num_layer)
        self.cone_intersection = ConeIntersection(self.entity_dim, drop)
        self.cone_negation = ConeNegation()

    def get_float_embedding(self, floats):

        float_emb = torch.zeros(floats.shape[0], 2 * self.entity_dim, device=self.device)
        div_term = torch.exp((torch.arange(0, 2 * self.entity_dim, 2, dtype=torch.float) *
                            -(math.log(10000.0) / self.entity_dim)))
        div_term = div_term.to(self.device)
        
        float_emb[:, 0::2] = torch.sin(floats.unsqueeze(-1) * div_term.unsqueeze(0))
        float_emb[:, 1::2] = torch.cos(floats.unsqueeze(-1) * div_term.unsqueeze(0))

        return float_emb

    def get_entity_embedding(self, entity_ids: torch.Tensor):
        emb = self.entity_embeddings(entity_ids)
        axis_emb = convert_to_axis(self.angle_scale(emb, self.axis_scale))
        arg_emb = torch.zeros_like(axis_emb)
        return torch.cat((axis_emb, arg_emb), dim=-1)

    def get_projection_embedding(self, proj_ids: torch.Tensor, emb):
        entity_axis_emb, entity_arg_emb = torch.chunk(emb, 2, dim=-1)
        proj_ids, alpha_floats, beta_floats = proj_ids[0], proj_ids[1], proj_ids[2]
        assert emb.shape[0] == len(proj_ids)
        alpha_emb = self.get_float_embedding(alpha_floats)
        beta_emb = self.get_float_embedding(beta_floats)
        rel_emb = self.relation_embeddings(proj_ids.to(int))
        rel_axis_emb, rel_arg_emb = torch.chunk(rel_emb, 2, dim=-1)
        rel_axis_emb = convert_to_axis(self.angle_scale(rel_axis_emb, self.axis_scale))
        rel_arg_emb = convert_to_axis(self.angle_scale(rel_arg_emb, self.arg_scale))  # "convert_to_axis" also

        alpha_axis_emb, alpha_arg_emb = torch.chunk(alpha_emb, 2, dim=-1)
        alpha_axis_emb = convert_to_axis(self.angle_scale(alpha_axis_emb, self.axis_scale))
        alpha_arg_emb = convert_to_axis(self.angle_scale(alpha_arg_emb, self.arg_scale)) 

        beta_axis_emb, beta_arg_emb = torch.chunk(beta_emb, 2, dim=-1)
        beta_axis_emb = convert_to_axis(self.angle_scale(beta_axis_emb, self.axis_scale))
        beta_arg_emb = convert_to_axis(self.angle_scale(beta_arg_emb, self.arg_scale)) 

        pro_axis_emb, pro_arg_emb = self.projection_net(entity_axis_emb, entity_arg_emb, rel_axis_emb,
                                                        rel_arg_emb, alpha_axis_emb, alpha_arg_emb,
                                                        beta_axis_emb, beta_arg_emb)
        pro_emb = torch.cat((pro_axis_emb, pro_arg_emb), dim=-1)
        return pro_emb

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor]):
        axis_emb_list, arg_emb_list = [], []
        for single_emb in conj_emb:
            single_axis_emb, single_arg_emb = torch.chunk(single_emb, 2, dim=-1)
            axis_emb_list.append(single_axis_emb)
            arg_emb_list.append(single_arg_emb)
        stacked_axis_emb = torch.stack(axis_emb_list)
        stacked_arg_emb = torch.stack(arg_emb_list)
        new_axis_emb, new_arg_emb = self.cone_intersection(stacked_axis_emb, stacked_arg_emb)
        return torch.cat((new_axis_emb, new_arg_emb), dim=-1)

    def get_negation_embedding(self, emb: torch.Tensor):
        axis_emb, arg_emb = torch.chunk(emb, 2, dim=-1)
        neg_axis_emb, neg_arg_emb = self.cone_negation(axis_emb, arg_emb)
        return torch.cat((neg_axis_emb, neg_arg_emb), dim=-1)

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor]):
        return torch.stack(disj_emb, dim=1)

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        assert False, 'Do not use d in ConE'

    def get_multiple_difference_embedding(self, emb: List[torch.Tensor], kwargs):
        assert False, 'Do not use D in ConE'

    def compute_logit(self, entity_emb, query_emb):
        entity_axis_emb, _ = torch.chunk(entity_emb, 2, dim=-1)
        query_axis_emb, query_arg_emb = torch.chunk(query_emb, 2, dim=-1)
        delta1 = entity_axis_emb - (query_axis_emb - query_arg_emb)
        delta2 = entity_axis_emb - (query_axis_emb + query_arg_emb)

        distance2axis = torch.abs(torch.sin((entity_axis_emb - query_axis_emb) / 2))
        distance_base = torch.abs(torch.sin(query_arg_emb / 2))

        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
        distance_out[indicator_in] = 0.

        distance_in = torch.min(distance2axis, distance_base)

        distance = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)
        #logit = self.gamma - distance * self.modulus  # weird since modulus is a learnable parameter
        logit = self.f(distance * self.modulus - self.gamma)
        return logit

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList], value_set: List[IntList], union: bool = False):
        assert pred_emb.shape[0] == len(answer_set)
        chosen_ans, chosen_scores, chosen_false_ans, subsampling_weight = \
            inclusion_sampling(answer_set,value_set, negative_size=self.negative_size, entity_num=self.n_entity)
        answer_embedding = self.get_entity_embedding(torch.tensor(np.array(chosen_ans), device=self.device))
        neg_embedding = self.get_entity_embedding(torch.tensor(np.array(chosen_false_ans), device=self.device).view(-1))  # n*dim
        neg_embedding = neg_embedding.view(-1, self.negative_size, 2 * self.entity_dim)  # batch*negative*dim
        pred_emb = pred_emb.unsqueeze(dim=-2)
        if union:
            positive_union_logit = self.compute_logit(answer_embedding.unsqueeze(1), pred_emb)
            positive_tmp = torch.max(positive_union_logit, dim=1)[0]
            positive_logit = (positive_tmp - chosen_scores)**2
            negative_union_logit = self.compute_logit(neg_embedding.unsqueeze(1), pred_emb)
            negative_tmp = torch.max(negative_union_logit, dim=1)[0]
            negative_logit = negative_tmp**2
        else:
            chosen_scores = torch.tensor(chosen_scores, device=self.device)
            positive_tmp = self.compute_logit(answer_embedding, pred_emb)
            positive_logit = (positive_tmp - chosen_scores)**2
            negative_tmp = self.compute_logit(neg_embedding, pred_emb)  # b*negative
            negative_logit = negative_tmp**2

        
        return positive_logit, negative_logit, subsampling_weight.to(self.device)

    def compute_all_entity_logit(self, pred_emb_list: List[torch.Tensor], union: bool = False):
        if union:
            pred_emb = torch.stack(pred_emb_list, dim=0)
        else:
            pred_emb = pred_emb_list[0]
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding = self.get_entity_embedding(all_entities)  # n_entity*dim
        pred_emb = pred_emb.unsqueeze(-2)  # batch*(disj)*1*dim
        batch_num = find_optimal_batch(all_embedding, query_dist=pred_emb, compute_logit=self.compute_logit,
                                       union=union)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            if union:
                union_part = self.compute_logit(answer_part.unsqueeze(0).unsqueeze(0), pred_emb)
                logit_part = torch.max(union_part, dim=0)[0]
            else:
                logit_part = self.compute_logit(answer_part.unsqueeze(dim=0), pred_emb)
                # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit
