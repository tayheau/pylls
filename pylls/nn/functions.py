from typing import Any, Union
import numpy as np
from pylls.base.base import Tensor

Arrays = Union[np.ndarray, Tensor]

def linear(input:Tensor, weight:Tensor, biais:Union[Tensor, None]= None) -> Arrays:
    out = input @ weight.transpose()  
    return out + biais if biais else out 

def tanh(input: Tensor) -> Tensor:
    out = Tensor(np.tanh(input.data), children=(input, ))

    if input._requires_grad and Tensor.gradient_enabled:
        def _backprop():
            input.grad += out.grad * (1 - out.data**2) 
        out._backprop = _backprop
        out._requires_grad()
    
    return out 

def MSE(x:Tensor, y:Tensor) -> Tensor:
    out = ((y - x) ** 2).sum()
    return out

#TODO: 
######ACTIVATIONS#####
#   - sigmoid
#   - ReLU
#   - softmax
######LOSS###########
#   - MSE
