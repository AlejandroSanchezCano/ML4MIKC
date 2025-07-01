"""
===============================================================================
Title:      Map
Outline:    Map class to handle operations regarding:
            - Contact maps
            - Distance maps
            - PAE martrices
            - Attention maps
            It supports padding, plotting, and removing the main diagonal.
Author:     Alejandro Sánchez Cano
Date:       30/06/2025
===============================================================================
"""

# Built-in imports
from string import ascii_uppercase, ascii_lowercase

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

class Map:

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
    
    def __repr__(self):
        return str(self.matrix)

    def __len__(self) -> int:
        '''
        Returns the size of the map matrix.

        Returns
        -------
        int
            Size of the map matrix.
        '''
        return len(self.matrix)

    def pad(self, max_size: int) -> 'Map':
        '''
        Pads the map matrix to a square matrix of size max_size x max_size on
        the bottom and right sides with zeros. This is done to make the maps
        of uniform size during model training and evaluation.

        Parameters
        ----------
        max_size : int
            The target size of the padded map matrix.
        
        Returns
        -------
        Map
            A new Map object with the padded matrix.
        '''
        # Calculate the amount of padding needed
        target_rows, target_cols = max_size, max_size
        rows, cols = self.matrix.shape
        pad_rows = target_rows - rows
        pad_cols = target_cols - cols
        assert pad_rows >= 0 and pad_cols >= 0, 'The matrix is larger than the target size.'

        # Pad only the bottom and right sides
        padded_matrix = np.pad(self.matrix, 
                            ((0, pad_rows), (0, pad_cols)), 
                            mode='constant', 
                            constant_values=0)

        return Map(padded_matrix)        

    def plot(self, lengths: list = [], save: str = '') -> None:
        '''
        Plots the map.

        Parameters
        ----------
        lengths : list, optional
            List of lengths of the chains in the map. Used to draw vertical and 
            horizontal lines separating the chains. 
        save : str, optional
            If provided, the plot will be saved to this file path.
        '''
        # Plot
        plt.figure()
        plt.title("Map")
        plt.xlabel('Residue position')    
        plt.ylabel('Residue position')
        vmax = 1 if self.matrix.max() < 1 else self.matrix.max()
        vmin = 0 
        plt.imshow(self.matrix, cmap="Greys", vmin=vmin, vmax=vmax, extent=(0, len(self), len(self), 0))

        # Multiple chains
        lengths = np.array([len(self.matrix)]) if not lengths else np.array(lengths)
        assert sum(lengths) == len(self.matrix), 'Chain lengths do not match the size of the contact map matrix.'

        # Horizontal and vertical lines
        chain_separation = np.cumsum(lengths)
        plt.vlines(x = chain_separation[:-1], ymin = 0, ymax = chain_separation[-1], colors = 'black')
        plt.hlines(y = chain_separation[:-1], xmin = 0, xmax = chain_separation[-1], colors = 'black')

        # Ticks
        ticks = np.append(0, chain_separation)
        ticks = (ticks[1:] + ticks[:-1])/2
        alphabet_list = list(ascii_uppercase+ascii_lowercase)
        plt.xticks(ticks, alphabet_list[:len(ticks)])
        plt.yticks(ticks, alphabet_list[:len(ticks)])

        # Show
        plt.colorbar()
        plt.show()
        if save:
            plt.savefig(save)
        plt.clf()

    def remove_diagonal(self) -> 'Map':
        '''
        Removes the main diagonal of the map matrix, and also the two
        immediate diagonals above and below the main diagonal, since they are
        always contacts and therefore not relevant for the prediction of 
        protein-protein interactions.

        Returns
        -------
        Map
            A new Map object with the main diagonal and the two immediate
            diagonals above and below the main diagonal set to zero.
        '''
        matrix = np.copy(self.matrix)
        for idx in range(len(self)):
            for increment in range(-2, 3):
                if idx + increment >= 0 and idx + increment < len(self):
                    matrix[idx, idx + increment] = 0
                    matrix[idx + increment, idx] = 0
        return Map(matrix)

    def cmap(self, threshold: int) -> 'Map':
        '''
        Returns a contact map of the map matrix, where all values below a 
        distant threshold of x Angstroms are consider a contact (1) and the 
        rest are considered non-contact (0).

        Parameters
        ----------
        threshold : int
            Distance threshold in Angstroms to consider a contact.
        Returns
        -------
        Map
            Contact Map
        '''
        cmap_matrix = np.where(self.matrix > threshold, 1, 0)
        return Map(cmap_matrix)

if __name__ == '__main__':
    '''Test class.'''
    map = Map(np.random.rand(10, 10) )
    print(len(map))
    map = map.pad(15)
    map = map.cmap(0.5)
    map.plot(save = 'test_map.png')