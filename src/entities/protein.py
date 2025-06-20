"""
===============================================================================
Title:      Protein
Outline:    Protein class to store and manage the individual proteins in the
            database. It provides methods to create new instances, check if a
            sequence is in the database, iterate over all proteins, and pickle
            the instance.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
===============================================================================
"""

# Built-in modules
import pprint
import hashlib
from typing import Generator

# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc import utils

class Protein:

    def __init__(self, seq: str, new = False):
        '''
        Constructor method for the Protein class.

        Parameters
        ----------
        seq : str
            Protein sequence.
        new : bool
            Flag for new instance creation.

        Attributes
        ----------
        bioID : str
            Biological name.
        uniprotID : str
            UniProt ID.
        seq : str
            Protein sequence.
        taxonID : int
            NCBI Taxon ID.
        species : str
            Species name.
        domains : dict[str, tuple[int, int]]
            InterPro protein domains.
        '''
        # File stem
        seq = seq.split('.')[0]
        if not any(char.isdigit() for char in seq):
            file_stem = hashlib.md5(seq.encode()).hexdigest()
        else:
            file_stem = seq

        # Instantiate from folder (from __dict__)
        try:
            __dict__ = utils.unpickle(path.PROTEIN / f'{file_stem}.prot')
            for attribute, value in __dict__.items():
                setattr(self, attribute, value)
        
        # Not found
        except FileNotFoundError:
            if not new:
                raise FileNotFoundError(f'No accession in Protein database with the sequence {seq}')

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
        return hashlib.md5(self.seq.encode()).hexdigest()

    @classmethod
    def new(cls, **kwargs: dict) -> None:
        '''
        Creates a new instance of the Protein class.

        Parameters
        ----------
        **kwargs : dict
            Instance attributes.
        '''
        protein = cls(kwargs['seq'], new = True)
        for attribute, value in kwargs.items():
            setattr(protein, attribute, value)
        return protein
    
    @staticmethod
    def in_database(seq: str) -> bool:
        '''
        Checks if a sequence is in the Protein database.

        Parameters
        ----------
        seq : str
            Protein sequence.

        Returns
        -------
        bool
            True if sequence is in the database, False otherwise.
        '''
        if not any(char.isdigit() for char in seq):
            file_stem = hashlib.md5(seq.encode()).hexdigest()
        else:
            file_stem = seq
        return (path.PROTEIN / f'{file_stem}.prot').exists()

    @staticmethod
    def iterate() -> Generator['Protein', None, None]:
        '''
        Iterates over all Protein objects in the PROTEIN folder.

        Yields
        ------
        Protein
            Protein object.
        '''
        total = len(list(path.PROTEIN.glob('*')))
        for protein in tqdm(path.PROTEIN.iterdir(), total = total):
            yield Protein(protein.stem)
    
    def pickle(self) -> None:
        '''
        Pickles the __dict__ of the Protein object to the PROTEIN folder
        and saves it as a .prot file with the md5 hash of the sequence 
        as the file stem.
        Custom (un)pickling methods avoid excesive use of the utils 
        module and provides higher code abstraction. 
        '''
        file_stem = self.__hash__()
        filepath = path.PROTEIN / f'{file_stem}.prot'
        utils.pickle(data = self.__dict__, path = filepath)

    def mikc(self) -> tuple[str, str, str, str]:
        '''
        Returns the M-I-K-C sequences of the protein.

        Returns
        -------
        tuple[str, str, str, str]
            M-I-K-C sequences of the protein.
        '''
        if self.bioID in ['SlRIN', 'SlTM3', 'ZaMADS70', 'SlMBP13', 'TtAG1-del3-del13', 'Ta42G17' , 'Ta57H08']:
            # 2K, 0M, 0M, noMIKC, dels deleted kink k1-k2, 0K2K3
            print(self.__hash__())
            return 0, 0, 0, 0
        try:
            assert 'IPR002100' in self.domains, 'Domain IPR002100 not found'
            assert 'IPR002487' in self.domains, 'Domain IPR002487 not found'
            assert len(self.domains['IPR002100']) == 1, 'Multiple domains IPR002100 found'
            assert len(self.domains['IPR002487']) == 1, 'Multiple domains IPR002487 found'
        except AssertionError as e:
            print(self.bioID)
            print(self.domains)
            print(e)
        
        try:
            m_i_limit = int(self.domains['IPR002100'][0][-1] + 1)
            i_k_limit = int(self.domains['IPR002487'][0][0] + 1)
            k_c_limit = int(self.domains['IPR002487'][0][-1] + 1)
            m, i, k, c = self.seq[:m_i_limit], self.seq[m_i_limit:i_k_limit], self.seq[i_k_limit:k_c_limit], self.seq[k_c_limit:]
            assert len(self.seq) == len(m) + len(i) + len(k) + len(c), 'Domain lengths do not add up to protein sequence'

        except IndexError as e:
            print(self.bioID)
            print(self.domains)
            print(e)
            return 0, 0, 0, 0
    
        except AssertionError as e:
            print(self.bioID)
            print(self.domains)
            raise e

        return m, i, k, c
    
if __name__ == '__main__':
    '''Test class'''
    p = Protein('0a19b272632f21ece4014fb718b68022')
    print(p.bioID)
    print(p.domains)
    print(p.mikc())
