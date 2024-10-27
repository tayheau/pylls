from pylls.base.base import Tensor
from ._optimizer import Optimizer
from typing import List

class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr=0.001) -> None:
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data -= (self.lr * param.grad)

    
