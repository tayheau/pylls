from typing import Any
import numpy as np
from ..base import Tensor
from .module import Module
from math import sqrt

from . import functions as F
class Linear(Module):
    """
    Linear layer x @ w + b
    """
    def __init__(self, in_features:int, out_features:int, requires_biais:bool = True, dtype= None):
        self.in_features = in_features
        self.out_features = out_features
        self.requires_biais = requires_biais 
        self.dtype = dtype

        self.weights = Tensor(np.random.uniform(-sqrt(1/self.in_features), sqrt(1/self.in_features), (out_features, in_features)), requires_grad=True)

        if self.requires_biais: 
            self.biais = Tensor(np.random.uniform(-sqrt(1/self.in_features), sqrt(1/self.in_features), (out_features)), requires_grad=True)
        else:
            self.biais = None

    def forward(self, X:Tensor) -> Tensor:
        return F.linear(X, self.weights, self.biais) 
