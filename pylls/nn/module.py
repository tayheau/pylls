from typing import Any, List

from ..base import Tensor 

class Module():
    """
    Base classes for other parts
    """
    def __repr__(self) -> str:
        return f"{self._get_modules()}"

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
        return self.forward(*args, **kwargs)

    def _get_parameters(self) -> List[Tensor]:
        return [ i for i in self._get_tensors() if i.requires_grad ]

    def _get_tensors(self) -> List[Tensor]:
        return [
            sub_args
            for arg in self.__dict__.values() 
            for sub_args in (arg._get_tensors() if isinstance(arg, Module) else [arg])
            if isinstance(sub_args, Tensor)
        ]

    def _get_modules(self) -> List["Module"]:
        return [i for i in self.__dict__.values() if isinstance(i, Module)]

    def __str__(self) -> str:
       return f"Not implemented yet" 
    #TODO : 
    #   - OrderedDictModule
    #   - __str__ method to give the architecture of the model  
