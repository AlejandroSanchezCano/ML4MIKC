"""
===============================================================================
Title:      Utils module
Outline:    Utility functions for general purposes:
                - Pickle and unpickle objects.
Docs:       https://docs.python.org/3/library/pickle.html
Author:     Alejandro Sánchez Cano
Date:       2024-10-02
===============================================================================
"""

# Built-in modules
import pickle as pkl
from typing import Any

def pickle(data: Any, path: str) -> None:
    '''
    Pickle an object and store it.

    Parameters
    ----------
    data : Any
        Pickable object that will be stored.
    path : str
        Storing path.
    '''
    with open(path, 'wb') as handle:
        pkl.dump(
            obj = data,
            file = handle, 
            protocol = pkl.HIGHEST_PROTOCOL
            )

def unpickle(path: str) -> Any:
    '''
    Retrieves and unpickles a pickled object.

    Parameters
    ----------
    path : str
        Storing path of the object to unpickle.

    Returns
    -------
    Any
        Unpickled object.
    '''
    with open(path, 'rb') as handle:
        return pkl.load(file = handle)