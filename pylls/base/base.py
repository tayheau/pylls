import numpy as np
from typing import Union, Tuple
from ..tools.tools import Hyperparameters

Arrays = Union[np.ndarray]

class Tensor(Hyperparameters):

    gradient_enabled : bool = True

    def __init__(self, data: np.ndarray, children = (), requires_grad: bool = False):
        assert isinstance(data, np.ndarray), f'Data must be of type numpy.ndarray, got {type(data)}'
        self.save_hyperparameters()
        self.grad = 0
        self._backprop = lambda : None
        self.shape = self.data.shape

    def _requires_grad(self, bool: bool = True) -> None:
        self.requires_grad = bool

    def _broadcasted_axis(self, new_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Return a tuple of the axis to be broadcasted, useful for the sum over those axis during backprop.

        ex : 
            old_shape = (3,)
            new_shape = (2, 2, 3)
            returns (0, 1)
        """
        old_shape = self.shape
        augmented_shape = (1,) * (len(new_shape) - len(old_shape)) + old_shape
        return tuple([i for i in range(len(new_shape)) if new_shape[i] > augmented_shape[i]])

    def _broadcast(self, new_shape: Tuple[int, ...]) -> 'Tensor':
        """
        Broadcast a Tensor do a given shape and save the trace if gradient is enabled
        """
        out = Tensor(np.broadcast_to(self.data, new_shape), children=(self, ))
        brodcasted_axis = self._broadcasted_axis(new_shape)
        if self.requires_grad and self.gradient_enabled:
            def _backprop():
                self.grad += np.sum(out.data, axis=brodcasted_axis)
            out._backprop = _backprop
            out._requires_grad()
        return out

    def _preprocess(self, other: Union[Arrays, 'Tensor']) -> Tuple['Tensor', 'Tensor']:
        """
        Preprocessing of the Tensors, checking their type and shape, broadcast if needed.
        """
        if not isinstance(other, Tensor) : other = Tensor(other)
        if self.shape != other.data.shape:
            shape = np.broadcast_shapes(self.shape, other.data.shape)
            self.data, other.data = self._broadcast(shape), other._broadcast(shape)
        return self, other



    def __repr__(self):
        return f'Tensor({self.data})'
    

    def __add__(self, other: Union[Arrays, 'Tensor']) -> 'Tensor':
        self, other = self._preprocess(other)
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
    
    def __pow__(self, exponent:Union[int, float]) -> 'Tensor':
        new_data = self.data ** exponent if exponent > 0 else 1 / (self.data ** -exponent)
        o = Tensor(self.data ** exponent, children=(self, ))
        if self.gradient_enabled and self.requires_grad:
            def _backprop():
                self.grad += exponent * (self.data ** (exponent - 1)) * o.grad
            o._backprop = _backprop
            o._requires_grad()
        return o


   
    def backprop(self) -> None:
        """
        topological oredering using DFS to implement 
        """
        visited = set()
        topo_order = []
        def visit(node):
            if node not in visited:
                visited.add(node)
                for children in set(node.children):
                    visit(children)
                topo_order.append(node)
        visit(self)
        self.grad = 1
        for value in reversed(topo_order):
            value._backprop()
    


def hello_world():
    print('Hello, World! 4')


