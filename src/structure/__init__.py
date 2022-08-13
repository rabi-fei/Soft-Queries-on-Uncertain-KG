from .knowledge_graph_index import KGIndex
from .knowledge_graph import KnowledgeGraph
from .neural_binary_predicate import NeuralBinaryPredicate
from .nbp_complex import ComplEx
from .nbp_transe import TransE
from .nbp_swtranse import SWTransE


def get(name):
    if name.lower() == 'transe':
        return TransE
    if name.lower() == 'swtranse':
        return SWTransE