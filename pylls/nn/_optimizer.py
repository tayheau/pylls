from typing import Any, List

from pylls.base.base import Tensor

class Optimizer():
    def __init__(self, params: List[Tensor], lr=0.001):
        self.params = params
        self.lr = lr
        
    def step(self) -> None:
        raise NotImplementedError("'step()' needs to be implemented")


    def zero_grad(self) -> None:
        for tensor in self.params:
            tensor._zero_grad()
