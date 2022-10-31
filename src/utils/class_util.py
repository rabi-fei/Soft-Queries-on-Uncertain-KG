import collections
from functools import partial
from itertools import repeat
import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


def nested_dict(): return collections.defaultdict(nested_dict)


def fixed_depth_nested_dict(default_factory, depth=1):
    result = partial(collections.defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(collections.defaultdict, result)
    return result()


def rename_ordered_dict(old_dict, old_key, new_key):
    """
    Create a new OrderedDict for rename a given key in old OrderedDict
    """
    new_dict = collections.OrderedDict((new_key if k == old_key else k, v) for k, v in old_dict.items())
    return new_dict


def compare_torch_dict(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    final_compare_dict = {}
    for key in dict1:
        if isinstance(dict1[key], dict):
            sub_compare = compare_torch_dict(dict1[key], dict2[key])
            final_compare_dict[key] = sub_compare
        elif isinstance(dict1[key], torch.Tensor):
            final_compare_dict[key] = torch.all(dict1[key] == dict2[key])
        else:
            final_compare_dict[key] = (dict1[key] == dict2[key])
    return final_compare_dict


