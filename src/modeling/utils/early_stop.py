"""
===============================================================================
Title:      Early stopping
Outline:    EarlyStopping class to halt training when the validation/test loss
            does not improve after a certain number of epochs.
Author:     Alejandro Sánchez Cano
Date:       2024-10-11
Version:    2025-03-03
License:    MIT
===============================================================================
"""

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        '''
        Class constructor.

        Parameters
        ----------
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        min_delta : float
            Minimum change in the monitored quantity to qualify as an improvement.
        '''

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss: float) -> bool:
        '''
        Check if the training should stop.

        Parameters
        ----------
        loss : float
            Validation/test loss.

        Returns
        -------
        bool
            True if the training should stop, False otherwise.
        '''
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True