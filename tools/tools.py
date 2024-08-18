import inspect 
from typing import Union

class Hyperparameters():
    """
    
    Taken from https://d2l.ai/ 
    A class used to save automaticly the hyperparameters of method or function - excepted the ones in the `ignore` list or starting with `_` - as attributes of the class instance.

    Methods
    -------
    save_hyperparameters(ignore = [])
        Save all parameters from the calling frame as instance attributes excepted the ones in the `ignore` list or starting with `_`.
    """
    def save_hyperparameters(self, ignore = []):
        frame = inspect.currentframe().f_back
        _, _, _, locals = inspect.getargvalues(frame)
        clean_locals = {k: v for k, v in locals.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in clean_locals.items():
            setattr(self, k, v)