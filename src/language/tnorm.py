from abc import abstractmethod
import torch


class Tnorm:
    @classmethod
    def negation(self, a):
        return 1-a


class ProductTNorm(Tnorm):
    @classmethod
    def conjunction(self, a, b):
        return a * b

    @classmethod
    def disjunction(self, a, b):
        return a + b - a * b
