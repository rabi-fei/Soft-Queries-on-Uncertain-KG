from typing import Union, List
from itertools import chain
import json
from random import shuffle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.language.fol import FirstOrderFormula
from src.language.grammar import parse_lstr_to_lformula


def _iter_triple_from_tsv(triple_file):
    with open(triple_file, 'rt') as f:
        for line in f.readlines():
            tp = line.strip().split()
            assert len(tp) == 3
            triple = [int(t) for t in tp]
            yield triple


def iter_triple_from_tsv(triple_files):
    if isinstance(triple_files, list):
        return chain(*[iter_triple_from_tsv(tfile) for tfile in triple_files])
    elif isinstance(triple_files, str):
        return _iter_triple_from_tsv(triple_files)
    else:
        raise NotImplementedError("invalid input of triple files")


def tensorize_batch_entities(
        entities: Union[List[int], List[List[int]], torch.Tensor],
        device) -> torch.Tensor:
    """
    convert the entities into the tensor formulation
    in the shape of [batch_size, num_entities]
    we interprete three cases
    1. List[int] batch size = 1
    2. List[List[int]], each inner list is a sample
    3. torch.Tensor in shape [batch_size, num_entities]
    """
    if isinstance(entities, list):
        if isinstance(entities[0], int):
            # in this case, batch size = 1
            entity_tensor = torch.tensor(
                entities, device=device).reshape(-1, 1)
        elif isinstance(entities[0], list):
            # in this case, batch size = len(entities)
            assert isinstance(entities[0][0], int)
            entity_tensor = torch.tensor(
                entities, device=device).reshape(len(entities), -1)
        else:
            raise NotImplementedError(
                "higher order nested list is not supported")
    elif isinstance(entities, torch.Tensor):
        assert entities.dim() == 2
        entity_tensor = entities.to(device)
    else:
        raise NotImplementedError("unsupported input entities type")
    return entity_tensor


class RaggedBatch:
    def __init__(self, flatten, sizes):
        self.flatten = flatten
        self.sizes = sizes

    def run_ops_on_flatten(self, opfunc):
        return RaggedBatch(
            flatten=opfunc(self.flatten),
            sizes=self.sizes)

    def to_dense_matrix(self, padding_value):
        # split the first axis of the flattened Tensor by sizes
        flatten_sliced = torch.split(
            self.flatten, split_size_or_sections=self.sizes, dim=0)
        dense_matrix = pad_sequence(
            flatten_sliced, batch_first=True, padding_value=padding_value)
        # if the self.flattened is of shape [L, *]
        # then dense_matrix is of shape [batch_size, max_of_self.sizes, *]
        return dense_matrix

class QAACollatorWithNegativeSampling:
    def __init__(self, lstr, answer_size=-1, negative_sample_size=-1):
        self.lformula = parse_lstr_to_lformula(lstr)
        self.answer_size = answer_size
        self.negative_sample_size = negative_sample_size

    def __call__(self, batch_input):
        positive_fof = FirstOrderFormula(self.lformula)
        negative_fof = FirstOrderFormula(self.lformula)

        for rsdict, _easy_ans, _hard_ans in batch_input:
            positive_fof.append_qa_instances_as_sentence(rsdict,
                                                         answers=_easy_ans)

            noisy_samples_tensor = torch.randint(
                low=0, high=self.answer_size, size=(self.negative_sample_size))
            noisy_samples = noisy_samples_tensor.numpy().tolist()

            negative_fof.append_qa_instances_as_sentence(rsdict,
                                                         answers=noisy_samples)

        return positive_fof, negative_fof

class QAACollator:
    def __init__(self, lstr):
        self.lformula = parse_lstr_to_lformula(lstr)

    def __call__(self, batch_input):
        fof = FirstOrderFormula(self.lformula, easy_answer=[], hard_answer=[])
        for rsdict, _easy_ans, _hard_ans in batch_input:
            easy_ans = torch.tensor(_easy_ans).view(1, -1)
            hard_ans = torch.tensor(_hard_ans).view(1, -1)

            fof.append_relation_and_symbols(rsdict)
            fof.easy_answer_list.append(easy_ans)
            fof.hard_answer_list.append(hard_ans)

        return fof

class QueryAnsweringSeqDataLoader:
    def __init__(self, qaafile, **dataloader_kwargs) -> None:
        self.dataloader_kwargs = dataloader_kwargs

        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)

        self.lstr_iterator = {}
        self.batch_buffer = []

    def __iter__(self):
        for lstr, qaa in self.lstr_qaa.items():
            self.lstr_iterator[lstr] = iter(DataLoader(qaa,
                collate_fn=QAACollator(lstr),
                **self.dataloader_kwargs))
        return self

    def __next__(self):
        if len(self.batch_buffer) == 0:
            for lstr, iterator in self.lstr_iterator.items():
                try:
                    self.batch_buffer.append(
                        next(iterator)
                    )
                except StopIteration:
                    print(f"{lstr} iterator run out")

            if len(self.batch_buffer) == 0:
                raise StopIteration
            else:
                shuffle(self.batch_buffer)

        return [self.batch_buffer.pop()]


class QueryAnsweringMixDataLoader:
    def __init__(self, qaafile, **dataloader_kwargs) -> None:
        self.dataloader_kwargs = dataloader_kwargs

        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)

        samples_per_query = {}
        total_samplers = 0
        for k in self.lstr_qaa:
            size_k = len(self.lstr_qaa[k])
            samples_per_query[k] = size_k
            total_samplers += size_k

        total_num_iterations = total_samplers//dataloader_kwargs.pop('batch_size')+1

        self.batch_size_per_query = {
            k: samples_per_query[k] // total_num_iterations + 1
            for k in samples_per_query}
        self.lstr_iterator = {}

    def __iter__(self):
        for lstr, qaa in self.lstr_qaa.items():
            self.lstr_iterator[lstr] = iter(DataLoader(qaa,
                batch_size=self.batch_size_per_query[lstr],
                collate_fn=QAACollator(lstr),
                **self.dataloader_kwargs))

        return self

    def __next__(self):
        buffer = []
        for _, dataloader in self.lstr_iterator.items():
            try:
                buffer.append(next(dataloader))
            except StopIteration:
                pass

        if len(buffer) == 0:
            raise StopIteration

        return buffer

class TrainQueryAnsweringWithSentenceVerificationDataLoader:
    def __init__(self, qaafile, answer_size, neg_sample_size, **dataloader_kwargs) -> None:
        self.qaafile = qaafile
        self.answer_size = answer_size
        self.neg_sample_size = neg_sample_size
        self.dataloader_kwargs = dataloader_kwargs

        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)

        self.lstr_iterator = {}
        self.batch_buffer = []

    def __iter__(self):
        for lstr, qaa in self.lstr_qaa.items():
            self.lstr_iterator[lstr] = iter(DataLoader(qaa,
                collate_fn=QAACollatorWithNegativeSampling(
                    lstr, self.answer_size, self.neg_sample_size),
                **self.dataloader_kwargs))
        return self

    def __next__(self):
        if len(self.batch_buffer) == 0:
            for lstr, iterator in self.lstr_iterator.items():
                try:
                    self.batch_buffer.append(
                        next(iterator)
                    )
                except StopIteration:
                    print(f"{lstr} iterator run out")

            if len(self.batch_buffer) == 0:
                raise StopIteration
            else:
                shuffle(self.batch_buffer)
        print("fetched buffer")

        return [self.batch_buffer.pop()]