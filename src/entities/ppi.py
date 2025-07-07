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
from collections import Counter

# Third-party modules
import pandas as pd
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

    def interact(self) -> int | str:
        '''
        Calculates the interaction score of the PPI by considering the
        different interaction cases. A positive interaction is scored as 1,
        a negative interaction as 0, and an uncertain interaction as '?'.
        - Single origin:
            - Single value: return it if not NC, AUTO, ND, NLW or nan
            - Two values: return 1 if one of the values is 1, 0 if one of the 
            values is 0, and ? if both are NC
            - Multiple values: average and round them. If average around 0.5, return ?.
        - Multiple origins:
            - All from Scoring independent dataset: treat as single origin
            - Different origins:
                - All values are the same: return the value
                - Different values:
                    - Majority vote: return the majority value
                    - No majority: ?
        Returns
        -------
        int | str
            Interaction score of the PPI -> 1, 0, or '?'.
        '''
        def score_single_origin_interactions(values: list[int | str]) -> int | str:
            '''
            Scores a single origin interaction based on the values provided.
            
            Parameters
            ----------
            values : list[int | str]
                List of interaction values to score.
            Returns
            -------
            int | str
                Interaction score -> 1, 0, or '?'.
            '''
            # Fully disregard nan, 'AUTO', 'ND', 'NLW', as they are not useful
            values = [v for v in values if v not in (None, 'AUTO', 'ND', 'NLW') and not pd.isna(v)]
            if values == []: return '?'
            # Consider 'NC' as 0.5, as it is sth in between 0 and 1
            values = [0.5 if v == 'NC' else v for v in values]
            # Single value -> return it if != 0.5
            if len(values) == 1:
                if values[0] == 0.5:
                    return '?'
                return int(values[0])
            # Two values -> [1, 0] = 1 and [NC, NC] = ?
            if len(values) == 2:
                if 1 in values:
                    return 1
                elif 0 in values:
                    return 0
                elif values == [0.5, 0.5]:
                    return '?'
                else:
                    raise ValueError(f'Unexpected values: {values}')
            # Multiple values -> average
            average = sum(values) / len(values)
            if average > 3/5:
                return 1
            elif average < 2/5:
                return 0
            else:
                return '?'

        # Single origin
        if len(self.origin) == 1:
            return score_single_origin_interactions(self.interaction[0])
        # Multiple origins -> condense Isa's interactions
        origin2interactions = {origin:interaction for origin, interaction in zip(self.origin, self.interaction)}
        scoring_origin = [origin for origin in self.origin if '&' in origin]
        origin2interactions['scoring'] = []
        for k in scoring_origin:
            if k in scoring_origin:
                origin2interactions['scoring'].extend(origin2interactions[k])
                del origin2interactions[k]
        if origin2interactions['scoring'] == []:
            del origin2interactions['scoring']
        # All interactions from Scoring independent dataset
        if len(origin2interactions) == 1 and 'scoring' in origin2interactions:
            scored = score_single_origin_interactions(origin2interactions['scoring'])
            return scored
        # Multiple origins with different interactions
        else:
            scored = [score_single_origin_interactions(interactions) for interactions in origin2interactions.values()]
            if len(set(scored)) == 1:
                return scored[0]
            scored = [s for s in scored if s != '?']
            freqs = list(Counter(scored).values())
            is_balanced = lambda x: max(x) == min(x) 
            if len(set(scored)) == 1 or not is_balanced(freqs):
                return max(set(scored), key = scored.count)
            else:
                return '?'

    @staticmethod
    def iterate(interact: bool = False) -> Generator['PPI', None, None]:
        '''
        Iterates over all PPI objects in the PPI folder.

        Parameters
        ----------
        interact : bool, optional
            If True, calculates the interaction score of each PPI.
            Default is False.

        Yields
        ------
        Protein
            Protein object.
        '''
        files = sorted(list(path.PPI.glob('*')))
        total = len(files)
        for file_name in tqdm(files, total=total):
            p1, p2 = file_name.stem.split('=')
            p1 = Protein(p1)
            p2 = Protein(p2)
            ppi = PPI(p1, p2)
            if interact:
                ppi.interaction = ppi.interact()
                if ppi.interaction != '?':
                    yield ppi
            else:
                yield ppi

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

    @staticmethod
    def create_backup() -> None:
        '''
        Creates a backup of the PPI database by copying all files from the PPI 
        folder to the PPI_BAKCUP folder. This is useful to avoid losing
        information in case of accidental deletion or corruption of the PPI
        database.
        '''
        for ppi in PPI.iterate():
            ppi.p1 = ppi.p1.__hash__()
            ppi.p2 = ppi.p2.__hash__()
            backup_path = path.BACKUP / f'{ppi.__hash__()}.ppi'
            utils.pickle(data = ppi.__dict__, path = backup_path)
            

    def restore_from_backup(self) -> None:
        '''
        Restores the PPI object from a backup file. The backup file is expected
        to be in the PPI_BACKUP folder and have the same name as the PPI object.
        This method is useful to recover a PPI object that has been deleted or
        corrupted.
        '''
        file_stem = self.__hash__()
        filepath = path.BACKUP / f'{file_stem}.ppi'
        if not filepath.exists():
            raise FileNotFoundError(f'No backup found for {file_stem}')
        __dict__ = utils.unpickle(filepath)
        for attribute, value in __dict__.items():
            setattr(self, attribute, value)

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
    ppi_files = sorted(list(path.PPI.glob('*')))
    total = len(ppi_files)
    for file_name in tqdm(ppi_files, total = total):
        print(file_name)