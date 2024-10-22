from typing import Any, List

from ..base import Tensor 

class Module():
    """
    Base classes for other parts
    """
    def zero_grad(self) -> None:
        """
        will zero grad all Tensor with require_grad 
        """
        for tensor in self._get_tensors():
            tensor._zero_grad()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Forward pass not implemented")
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        allow to call the forward method by the name of the instance
        """
        return self.forward(args, kwargs)

    def _get_tensors(self) -> List[Tensor]:
        return [i for i in self.__dict__.values() if isinstance(i, Tensor)]
        
    #TODO : 
    #   - return all elements of the module
    #       - tensors ---done---
    #       - other modules
    #       - implement zero_grad() ---done---
