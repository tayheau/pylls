import numpy as np
from typing import Union, Tuple
from ..tools.tools import Hyperparameters

Data = Union[np.ndarray, list, int, np.int64, float] 

class no_grad():
    """
    Context manager to deactivate gradient computing
    """
    def __enter__(self):
        self.previous_gradient_state = Tensor.gradient_enabled
        Tensor.gradient_enabled = False
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        Tensor.gradient_enabled = self.previous_gradient_state

class Tensor(Hyperparameters):
    gradient_enabled : bool = True
    def __init__(self, data: Data, children = (), requires_grad: bool = False):
        assert isinstance(data, Data), f'Data must be of type {Data}, got {type(data)}'
        if type(data) != np.ndarray:
            data = np.array(data)
        self.save_hyperparameters()
        self.grad = np.zeros_like(self.data)
        self._backprop = lambda : None
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        
    def __repr__(self) -> str:
        return f'Tensor({self.data})'

    def _zero_grad(self) -> None:
        """
        Allow to reset the gradient
        """
        self.grad = np.zeros_like(self.grad)

    def _requires_grad(self, ops: bool = True) -> None:
        self.requires_grad = ops

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
        broadcasted_axis = self._broadcasted_axis(new_shape)
        if self.requires_grad and self.gradient_enabled:
            def _backprop():
                self.grad += np.sum(out.grad, axis=broadcasted_axis)
            out._backprop = _backprop
            out._requires_grad()
        return out

    def _preprocess(self, other: Union[Arrays, 'Tensor']) -> Tuple['Tensor', 'Tensor']:
        """
        Preprocessing of the Tensors, checking their type and shape, broadcast if needed.
        """
        if not isinstance(other, Tensor) : other = Tensor(other)
        if self.shape != other.shape:
            shape = np.broadcast_shapes(self.shape, other.data.shape)
            self, other = self._broadcast(shape), other._broadcast(shape)
        return self, other
    
    def __add__(self, other: Union[Arrays, 'Tensor']) -> 'Tensor':
        _, other = self._preprocess(other)
        o = Tensor(self.data + other.data, children = (self, other))
        if self.requires_grad and self.gradient_enabled:
            def _backprop():
                self.grad += o.grad
                other.grad += o.grad
            o._backprop = _backprop
            o._requires_grad()
        return o
    
    def __mul__(self, other: Union[np.ndarray, 'Tensor']) -> 'Tensor':
        self, other = self._preprocess(other)
        o = Tensor(self.data * other.data, children = (self, other))
        if self.requires_grad and self.gradient_enabled:
            def _backprop():
                self.grad += other.data * o.grad
                other.grad += self.data * o.grad 
            o._backprop = _backprop
            o._requires_grad()
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

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        o = Tensor(self.data @ other.data, children=(self, other))
        if self.gradient_enabled and self.requires_grad:
            def _backprop():
                self.grad += o.grad @ other.data.T
                other.grad += self.data.T @ o.grad
            o._backprop = _backprop
            o._requires_grad()
        return o

    def log(self) -> 'Tensor':
        """
        Compute the natural log
        """
        o = Tensor(np.log(self.data), children=(self, ))
        if self.gradient_enabled and self.requires_grad:
            def _backprop():
                self.grad += (o.grad / self.data)
            o._backprop = _backprop
            o._requires_grad()
        return o

    def transpose(self) -> 'Tensor':
        """
        Transpose the given Tensor
        """
        out = Tensor(self.data.T, children=(self, ))
        if self.requires_grad and self.gradient_enabled:
            def _backprop():
                self.grad = self.grad.T
            out.backprop = _backprop
            out._requires_grad()
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other 

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backprop(self) -> None:
        """
        topological oredering using DFS to implement 
        """
        assert self.requires_grad, f'grad not computed, not activated'
        visited = set()
        topo_order = []
        def visit(node):
            if node not in visited:
                visited.add(node)
                for children in set(node.children):
                    visit(children)
                topo_order.append(node)
        visit(self)
        self.grad = np.ones_like(self.data)
        for value in reversed(topo_order):
            value._backprop()

    #TODO: 
    #   - 
