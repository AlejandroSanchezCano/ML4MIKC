"""
===============================================================================
Title:      pLDDT
Outline:    PLDDT class to handle visualization of pLDDT values for protein
            structures.
Author:     Alejandro Sánchez Cano
Date:       20/06/2024
===============================================================================
"""

# Built-in modules
from typing import Any
from string import ascii_uppercase, ascii_lowercase

# Third-party modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class PLDDT:
    def __init__(self, plddt: np.ndarray):
        self.plddt = self.__format(plddt)

    def __format(self, plddt: Any) -> str:

        # To numpy array
        if isinstance(plddt, list):
            return np.array(plddt)
        # To percentage
        if plddt.max() < 1:
            return np.array(plddt) * 100
        
        return plddt
        
    def __repr__(self) -> str:
        '''
        Prints instance attributes for debugging purposes.

        Returns
        -------
        str
            Instance attribute mapping.
        '''
        return str(self.plddt)
    
    def plot(self, lengths: list = []) -> None:
        '''
        Plots pLDDT values.
        
        Parameters
        ----------
        lengths : list
            List of chain lengths.
        '''
        # Scatterplot and rugplot
        y = self.plddt
        x = list(range(len(y)))
        hue = ['Very high' if i >= 90 else 'Condifent' if i >= 70 else 'Low' if i >= 50 else 'Very low' for i in y]
        plddt_palette = {'Very high': '#0053D6', 'Condifent':'#65CBF3', 'Low':'#FFDB13', 'Very low':'#FF7D45'}
        fig = sns.scatterplot(x=x, y=y, color='black')
        sns.rugplot(x=x, hue = hue, palette = plddt_palette, legend = False, linewidth = 2, expand_margins=True)

        # Horizontal spans
        ax = plt.gca()
        ax.axhspan(90, 100, alpha = 0.5, color = '#0053D6')
        ax.axhspan(70, 90, alpha = 0.5, color = '#65CBF3')
        ax.axhspan(50, 70, alpha = 0.5, color = '#FFDB13')
        ax.axhspan(0, 50, alpha = 0.5, color = '#FF7D45')
        ax.set_ylim(0, 100)

        # Multiple chains
        if lengths:
            assert sum(lengths) == len(self.plddt), 'Chain lengths do not match the pLDDT values.'

            # Vertical sepratory lines      
            chain_separation = np.cumsum(lengths)
            plt.vlines(x = chain_separation[:-1], ymin = 0, ymax = chain_separation[-1], colors = 'black')

            # Chain ticks
            ticks = np.append(0, chain_separation)
            ticks = (ticks[1:] + ticks[:-1])/2
            alphabet_list = list(ascii_uppercase+ascii_lowercase)
            plt.xticks(ticks, alphabet_list[:len(ticks)])

        # Axis labels
        fig.set_xlabel('Residue index')
        fig.set_ylabel('pLDDT')

        # Save plot
        plt.savefig(f'plddt_pi_svp.png', bbox_inches='tight')
        plt.show()
    
if __name__ == '__main__':
    '''Test class'''
    import numpy as np
    plddt = PLDDT(np.arange(100, 90, -1))
    print(plddt.plot())