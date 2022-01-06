from .abstract_task import AbstractTask
from .link_prediction import LinkPrediction
from .query_answering import QueryAnswering

def get(name):
    if name.lower() == 'linkprediction':
        return LinkPrediction
    elif name.lower() == 'queryanswering':
        return QueryAnswering
    else:
        raise NotImplementedError