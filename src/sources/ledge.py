"""
===============================================================================
Title:      Ledge
Outline:    Ledge class to operate with the 'LEDGE.xlsx' file, whose content
            varies depend on the context:
            1) Literature mining -> contains every protein listed in the papers
            and how we arrived at their sequences.
            2) Scoring -> contains the sequences and other metadata of all MIKC
            proteins used in the domain swaps experiments.
            In general, it is used to check if a BioOD is in the document and
            fetch the content of other columns on the same row.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
===============================================================================
"""

# Third-party modules
import pandas as pd

# Custom modules
from src.misc import path

class Ledge:

    def __init__(self, path: str = path.LITERATUREMINING / 'LEDGE.xlsx'):
        self.df = pd.read_excel(path)

    def bioID_in_ledge(self, bioID: str) -> bool:
        '''
        Check if the BioID is in the DataFrame.

        Parameters
        ----------
        bioID : str
            BioID to search for.

        Returns
        -------
        bool
            True if the BioID is in the DataFrame, False otherwise.
        '''
        assert bioID in self.df['bioID'].values, f'{bioID} not found in LEDGE data frame'

    def fetch(self, what: str, column_from: str, column_to: str) -> str:
        '''
        Fetch the value from the DataFrame.

        Parameters
        ----------
        what : str
            Value to search for.
        column_from : str
            Column to search in.
        column_to : str
            Column to return.

        Returns
        -------
        str
            Value from the DataFrame.
        '''
        return self.df.loc[self.df[column_from] == what, column_to].values[0]
    
if __name__ == "__main__":
    '''Test class'''
    ledge = Ledge()
    print(ledge.bioID_in_ledge('TtAG1-del3'))