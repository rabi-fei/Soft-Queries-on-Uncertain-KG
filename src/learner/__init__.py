from .abstract import Learner
from .isomorphic import IsomorphicLearner
from .elementary import ElementaryLearner

def get(name):
    if name == 'I':
        return IsomorphicLearner
    elif name == 'E':
        return ElementaryLearner
    else:
        raise NotImplementedError