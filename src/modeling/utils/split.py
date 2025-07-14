"""
===============================================================================
Title:      Split
Outline:    Split class to split a dataset into train, validation, and test
            sets in a simple or k-fold manner.
Author:     Alejandro Sánchez Cano
Date:       2024-10-01
Version:    2025-03-03
License:    MIT
===============================================================================
"""

# Built-in modules
import random

# Third-party modules
import math
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, random_split

# Custom modules
from src.misc.logger import logger

class Split:

    def __init__(
            self, 
            data_loader: DataLoader, 
            batch_size: int, 
            sizes: list, 
            kfold: int = 0,
            cluster_membership: dict = None,
            ):
        '''
        Class constructor.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader with the dataset to split.
        batch_size : int
            Batch size.
        sizes : list
            List with the sizes of the train, validation, and test sets.
            If cluster_memership is provided, the sizes are the proportions of 
            clusters use for each set.
        kfold : int, optional
            Number of folds for the k-fold split, by default 0.
        cluster_membership : dict, optional
            Dictionary with the cluster membership information, by default None.
        '''
        # Attributes
        self.dataset = data_loader.dataset
        self.batch_size = batch_size
        self.train_size, self.val_size, self.test_size = sizes
        self.kfold = kfold
        self.cluster_membership = cluster_membership

        # Validate sizes
        assert abs(self.train_size + self.val_size + self.test_size - 1) < 0.001 , "Sizes must sum up to 1"

        # Loader attributes
        self.train_loader, self.val_loader, self.test_loader = self.__split()

    def __split(self) -> tuple:
        '''
        Decides which split to apply.

        Returns
        -------
        tuple
            Train, validation, and test loaders.
        '''
        if self.kfold:
            return self.__kfold_split()
        elif self.cluster_membership:
            return self.__cluster_split()
        else:
            return self.__simple_split()
        
    def __simple_split(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        '''
        Simple split of the dataset into train, validation, and test sets.

        Returns
        -------
        tuple[DataLoader, DataLoader, DataLoader]
            Train, validation, and test loaders.
        '''

        # Datsaset sizes
        train_size = math.ceil(self.train_size * len(self.dataset))
        val_size = math.ceil(self.val_size * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Logging
        logger.info(f'Train size: {len(train_dataset)}')
        logger.info(f'Validation size: {len(val_dataset)}')
        logger.info(f'Test size: {len(test_dataset)}')

        return train_loader, val_loader, test_loader
    
    def __cluster_split(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        '''
        Split the dataset into train, validation, and test sets based on 
        cluster membership.

        Returns
        -------
        tuple[DataLoader, DataLoader, DataLoader]
            Train, validation, and test loaders.
        '''
        # Cluster sizes
        unique_cluster_IDs = list(self.cluster_membership.keys())
        train_size = math.ceil(self.train_size * len(unique_cluster_IDs))
        val_size = math.ceil(self.val_size * len(unique_cluster_IDs))
        test_size = len(unique_cluster_IDs) - train_size - val_size

        # Split clusters
        random.shuffle(unique_cluster_IDs)
        train_cluster_IDs = unique_cluster_IDs[:train_size]
        val_cluster_IDs = unique_cluster_IDs[train_size:train_size + val_size]
        test_cluster_IDs = unique_cluster_IDs[train_size + val_size:]   

        # Create data loaders
        train_idx = [idx for cluster_ID in train_cluster_IDs for idx in self.cluster_membership[cluster_ID]]
        val_idx = [idx for cluster_ID in val_cluster_IDs for idx in self.cluster_membership[cluster_ID]]
        test_idx = [idx for cluster_ID in test_cluster_IDs for idx in self.cluster_membership[cluster_ID]]
        train_dataset = Subset(self.dataset, train_idx)
        val_dataset = Subset(self.dataset, val_idx)
        test_dataset = Subset(self.dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Logging
        logger.info(f'Train size: {len(train_dataset)}')
        logger.info(f'Validation size: {len(val_dataset)}')
        logger.info(f'Test size: {len(test_dataset)}')

        return train_loader, val_loader, test_loader
    
    def __kfold_split(self) -> tuple[list[DataLoader], list[DataLoader], DataLoader]:
        '''
        K-fold split of the dataset into train, validation, and test sets.

        Returns
        -------
        tuple[list[DataLoader], list[DataLoader], DataLoader]
            Train, validation, and test loaders.
        '''
        # Dataset sizes
        train_val_size = math.ceil((self.train_size + self.val_size) * len(self.dataset))
        test_size = len(self.dataset) - train_val_size

        # Split dataset
        train_val_dataset, test_dataset = random_split(
            self.dataset, [train_val_size, test_size]
        )

        # KFold train and validation sets
        train_loaders, val_loaders = [], []
        kf = KFold(n_splits = self.kfold, shuffle=True)
        for train_idx, val_idx in kf.split(train_val_dataset):
            train_subset = Subset(train_val_dataset, train_idx)
            val_subset = Subset(train_val_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle =True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)

        # Test loader
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Logging
        logger.info(f'Train + validation size: {len(train_subset) + len(val_subset)} divided in {self.kfold} folds')
        logger.info(f'Each fold has train size: {len(train_subset)} and validation size: {len(val_subset)}')
        logger.info(f'Test size: {len(test_dataset)}')

        return train_loaders, val_loaders, test_loader
    
    def C1_C2_C3(): 
        # TODO : Implement
        # IDEA: no only C1, C2, C3 but see whether enzymes that only have
        #  negative/positive labels are predicted +/- more heavily.
        pass


if __name__ == '__main__':
    # Example usage
    from torch.utils.data import Dataset, DataLoader

    class ExampleDataset(Dataset):
        def __init__(self, size):
            self.data = list(range(size))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = ExampleDataset(9)
    data_loader = DataLoader(dataset, batch_size=10)
    cluster_membership = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11]}

    split = Split(data_loader, batch_size=1, sizes=[0.5, 0.25, 0.25], cluster_membership=cluster_membership)

