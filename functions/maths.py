import numpy as np 
from base.base import Tensor
from typing import Union


class Maths():
    def __add__(self, other: Union[np.ndarray, 'Tensor']) -> 'Tensor':
        if not isinstance(other, Tensor) : other = Tensor(other)
        o = Tensor(self.data + other.data, children = (self, other))
        def _backprop():
            self.grad += o.grad
            other.grad += o.grad
        o._backprop = _backprop
        return o
    
    def __mul__(self, other: Union[np.ndarray, 'Tensor']) -> 'Tensor':
        if not isinstance(other, Tensor) : other = Tensor(other)
        o = Tensor(self.data * other.data, children = (self, other))

        def _backprop():
            self.grad += other.data * o.grad
            other.grad += self.data * o.grad 
        o._backprop = _backprop
        return o