from abc import abstractmethod, ABC
import json
from typing import List


class EFOF(ABC):
    @abstractmethod
    def normalize(self, *args, **kwargs):
        pass

    @abstractmethod
    def embed(self, *args, **kwargs):
        pass

class Connective(ABC):
    pass


class Atom(EFOF):
    pass

class Item(EFOF):
    pass

class Form(EFOF):
    pass

class Formula:
    def __init__(self, evars: List, fvars: List, efof: EFOF):
        self.evar = {v: None for v in evars}
        self.fvar = {v: None for v in fvars}
        self.efof = efof

    @classmethod
    def load(s):
        obj = json.loads(s)
        evars = obj['EVAR'].split(',')
        fvars = obj['FVAR'].split(',')
        efos = obj['EFOF']
        efof = Formula.parse(efos)
        return Formula(evars, fvars, efof)

    @classmethod
    def parse(efos):
        pass
