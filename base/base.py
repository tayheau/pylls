import numpy as np
from typing import Union
from tools.tools import Hyperparameters
from fuctions.maths import Maths

class Tensor(Hyperparameters, Maths):
    def __init__(self, data: np.ndarray, children = ()):
        assert isinstance(data, np.ndarray), f'Data must be of type numpy.ndarray, got {type(data)}'
        self.save_hyperparameters()
        self.grad = 0
        self._backprop = lambda : None

    def __repr__(self):
        return f'Tensor({self.data})'
    
    """
    topological oredering using DFS to implement 

    def backprop(self):
        visited = set()
        topo_order = []
        def visit(node):
    """


def hello_world():
    print('Hello, World! 3')


