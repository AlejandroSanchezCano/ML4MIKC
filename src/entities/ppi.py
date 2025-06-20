"""
===============================================================================
Title:      PPI
Outline:    PPI class to store and manage the dimer protein-protein
            interactions (PPI) in the database. It provides methods to create
            new instances, check if a PPI is in the database, iterate over all
            PPIs, and pickle the instance.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
===============================================================================
"""

# Buit-in modules
import pprint
from typing import Generator

# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc import utils
from src.entities.plddt import PLDDT
from src.entities.protein import Protein

class PPI:

    def __init__(self, p1: Protein, p2: Protein, new = False):

        # File stem
        file_stem = f'{p1.__hash__()}={p2.__hash__()}'

        # Instantiate from folder (from __dict__)
        try:
            __dict__ = utils.unpickle(path.PPI / f'{file_stem}.ppi')
            for attribute, value in __dict__.items():
                setattr(self, attribute, value)
        
        # Not found
        except FileNotFoundError:
            if not new:
                raise FileNotFoundError(f'No accession in PPI database with the proteins {p1} and {p2}')

    def __repr__(self) -> str:
        '''
        Prints instance attributes for debugging purposes.

        Returns
        -------
        str
            Instance attribute mapping.
        '''
        return pprint.pformat(self.__dict__)

    def __hash__(self) -> str:
        '''
        Hashes the protein sequence.

        Returns
        -------
        str
            Hashed protein sequence.
        '''
        return f'{self.p1.__hash__()}={self.p2.__hash__()}'

    @classmethod
    def new(cls, **kwargs: dict) -> None:
        '''
        Creates a new instance of the PPI class.

        Parameters
        ----------
        **kwargs : dict
            Instance attributes.
        '''
        ppi = cls(kwargs['p1'], kwargs['p2'], new = True)
        for attribute, value in kwargs.items():
            setattr(ppi, attribute, value)
        return ppi

    @staticmethod
    def iterate() -> Generator['PPI', None, None]:
        '''
        Iterates over all PPI objects in the PPI folder.

        Yields
        ------
        Protein
            Protein object.
        '''
        total = len(list(path.PPI.glob('*')))
        for file_name in tqdm(path.PPI.iterdir(), total = total):
            p1, p2 = file_name.stem.split('=')
            p1 = Protein(p1)
            p2 = Protein(p2)
            yield PPI(p1, p2)

    def pickle(self) -> None:
        '''
        Pickles the __dict__ of the PPI object to the 
        PPI folder and saves it as a .ppi file with the md5 
        hash of the sequences (separated by =) as the file stem.
        Custom (un)pickling methods avoid excesive use of the utils 
        module and provides higher code abstraction. 
        '''
        file_stem = self.__hash__()
        filepath = path.PPI / f'{file_stem}.ppi'
        self.p1 = self.p1.__hash__()
        self.p2 = self.p2.__hash__()
        utils.pickle(data = self.__dict__, path = filepath)
    
    @property
    def p1(self) -> Protein:
    # ONLY SHOW P1 == PROTEIN WHEN USING THE P1 ATTRIBUTE. ELSE IT IS KEPT AS THE HASH
    # THIS ADDS FLEXIBILITY  IN CASE THE OTHER CLASSES CHANGE
        if isinstance(self.__dict__['p1'], str):
            return Protein(self.__dict__['p1'])
        elif isinstance(self.__dict__['p1'], Protein):
            return self.__dict__['p1']

    @p1.setter
    def p1(self, p1: Protein) -> None:
        self.__dict__['p1'] = p1

    @property
    def p2(self) -> Protein:
        if isinstance(self.__dict__['p2'], str):
            return Protein(self.__dict__['p2'])
        elif isinstance(self.__dict__['p2'], Protein):
            return self.__dict__['p2']

    @p2.setter
    def p2(self, p2: Protein) -> None:
        self.__dict__['p2'] = p2

if __name__ == '__main__':
    pass