"""
===============================================================================
Title:      Interctor
Outline:    Interactor class to store MIKC protein information before building
            the final protein-protein interaction network. It uses the 
            UniProt ID as the unique identifier and stores metadata obtained
            from UniProt as a pickled .int file in the INTERACTORS folder. 
            It stores:
            - Uniprot ID
            - Domains: {'IPR002100':[(0,20), (30,50)], 'IPR003000':[(60,80)]}
            - Taxon ID: 3702
            - Section: 'TrEMBL'  or 'Swiss-Prot'
            - Primary accession: 'A0A022PSB5'
            - Secondary accession : ['A0A022PSB6', 'A0A022PSB6']
            - Sequence: 'MSTNPKPQRK...'
            - Structure: AlphaFold structure in PDB format.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
===============================================================================
"""

# Built-in modules
import pprint
from typing import Generator

# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc import utils


class Interactor:

    def __init__(self, uniprot_id: str):
        # Instantiate from folder (from __dict__)
        try:
            __dict__ = utils.unpickle(path.INTERACTORS / f'{uniprot_id}.int')
            for attribute, value in __dict__.items():
                setattr(self, attribute, value)
        
        # New instance
        except FileNotFoundError:
            self.uniprot_id = uniprot_id
            self.domains = {}
            self.taxon_id = ''
            self.section = ''
            self.primary_accession = ''
            self.secondary_accession = ''
            self.seq = ''
            self.structure = ''

    def __repr__(self) -> str:
        '''
        Prints instance attributes for debugging purposes.

        Returns
        -------
        str
            Instance attribute mapping.
        '''
        return pprint.pformat(self.__dict__)

    def pickle(self) -> None:
        '''
        Pickles the __dict__ of the Interactor object.
        Custom (un)pickling methods avoid excesive use of the utils 
        module and provides higher code abstraction. 
        '''
        filepath = path.INTERACTORS / f'{self.uniprot_id}.int'
        utils.pickle(data = self.__dict__, path = filepath)
    
    @staticmethod
    def iterate() -> Generator['Interactor', None, None]:
        '''
        Iterates over all Interactor objects in the INTERACTORS folder.

        Yields
        ------
        Interactor
            Interactor object.
        '''
        total = len(list(path.INTERACTORS.glob('*')))
        for file in tqdm(path.INTERACTORS.iterdir(), total = total):
            yield Interactor(file.stem)
    
if __name__ == '__main__':
    '''Test class'''
    interactor = Interactor('Q6GWV3')
    print(interactor)