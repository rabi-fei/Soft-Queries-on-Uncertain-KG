from .abstract_models import KnowledgeGraph, NeuralBinaryPredicate
from .transe import TransE

def get(name):
    if name.lower() == 'transe':
        return TransE
    else:
        return NotImplementedError