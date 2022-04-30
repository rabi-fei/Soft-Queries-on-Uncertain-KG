from .knowledge_graph import KnowledgeGraph
from .knowledge_graph_index import KGIndex
from .neural_binary_predicate import NeuralBinaryPredicate
from .transe import TransE


def get(name):
    if name.lower() == 'transe':
        return TransE
    else:
        return NotImplementedError
