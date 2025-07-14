# Third-party modules
import torch
import numpy as np
from torch.utils.data import Dataset

# Custom modules
from src.misc.logger import logger

class CustomDataset(Dataset):

    def __init__(self, y: torch.Tensor, **Xs: torch.Tensor):
        assert np.all([len(X) == len(y) for X in Xs.values()]), 'Number of X and y must be equal'
        self.y = y.long()
        self.Xs = Xs
        self.indeces = torch.arange(len(self.y))

        # Logging
        for key, X in self.Xs.items():
            logger.info(f'Data "{key}" shape: {X.shape}')
        logger.info(f'y shape: {self.y.shape}')
        logger.info(f'Positive y: {sum(self.y).item()}')
        logger.info(f'Negative y: {(len(self.y) - sum(self.y)).item()}')

    def __len__(self) -> int:
        '''
        Length method to return the size of the dataset.

        Returns
        -------
        int
            Size of the dataset.
        '''
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        '''
        Get item method to return a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple[torch.Tensor, int]
            Tuple with the image and the label.
        '''
        logger.debug(f"Fetching index {idx}: label {self.y[idx]}, index {self.indeces[idx]}")
        
        index = self.indeces[idx]
        y = self.y[idx]
        Xs = [X[idx] for X in self.Xs.values()]
        
        return index, y, *Xs