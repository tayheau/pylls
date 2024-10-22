from typing import Any, Union

from pylls.base.base import Tensor

Arrays = Union[np.ndarray, Tensor]

def linear(input:Arrays, weight:Arrays, biais= None) -> Arrays:
    out = input @ weight.transpose()  
    return out + biais if biais else out 
