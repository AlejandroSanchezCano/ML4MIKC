"""
===============================================================================
Title:      PPI
Outline:    PPI class to store and manage the dimer protein-protein
            interactions (PPI) in the database. It provides methods to create
            new instances, check if a PPI is in the database, iterate over all
            PPIs, and pickle the instance.
Author:     Alejandro Sánchez Cano
Date:       10/07/2025
===============================================================================
"""

# I BELIEVE CACHING CAN BE IMPROVED USING ALSO MULTITHREADING BUT I AM NOT SURE
# ALSO, PYTHON IMPLEMENTS CACHING ITSELF, SO OPENING THE PROTEINS GOES FROM 12 TO 3 SECS

# Buit-in modules
import pprint
from functools import cache
from collections import Counter
from typing import Any, Generator
from concurrent.futures import ThreadPoolExecutor

# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc import utils
from src.misc.logger import logger
from src.entities.plddt import PLDDT
from src.entities.protein import Protein


class PPI:

    def __init__(self, p1: Protein, p2: Protein, *args):
        # Instantiate from folder
        self.p1 = p1
        self.p2 = p2
        for arg in args:
            self._add_argument(arg)
    
    def _add_argument(self, arg: str) -> None:
        '''
        Multipurpose method that thakes an argument name, unpickles the file
        associated with the PPI and said name, and adds it to the instance in 
        a nested dictionary structure: 
        arg.subarg1.subarg2 -> {arg: {subarg1: {subarg2: ... : obj}}}
        If the argument already exists, it updates the existing dictionary with
        the new data.

        Parameters
        ----------
        arg : str
            Name of the argument to be added to the instance.
        '''
        file_stem = f'{self.p1.__hash__()}={self.p2.__hash__()}'
        file_path = path.PPI / f'{file_stem}.{arg}'
        try:
            obj = utils.unpickle(file_path)
        except FileNotFoundError:
            obj = None
        arg, *subargs = arg.split('.')
        existent = getattr(self, arg, None)
        nested = current = {}
        for idx, subarg in enumerate(subargs):
            current[subarg] = {}
            if idx < len(subargs) - 1:
                current = current[subarg]
            else:
                current[subarg] = obj

        if existent is None:
            if subargs:
                setattr(self, arg, nested)
            else:
                setattr(self, arg, obj)
        else:
            for key, value in nested.items():
                if key in existent:
                    existent[key].update(value)
                else:
                    existent[key] = value

            setattr(self, arg, existent)

    @classmethod
    def new(cls, **kwargs: dict) -> None:
        '''
        Creates a new instance of the PPI class.

        Parameters
        ----------
        **kwargs : dict
            Instance attributes.
        '''
        ppi = cls(kwargs['p1'], kwargs['p2'])
        for attribute, value in kwargs.items():
            setattr(ppi, attribute, value)
        return ppi

    @property
    def p1(self) -> Protein:
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

    @staticmethod
    def iterate(*args) -> Generator['PPI', None, None]:
        '''
        Iterates over all PPI objects in the database, unpickling them and
        returning a list of PPI instances. A couple of tricks are used to
        speed up the process:
        1. Caching Protein object instantiation
        2. Unpickling PPI objects with multithreading

        Parameters
        ----------
        *args : str
            Names of the arguments to be added to the PPI instances.
            Some are always present by default.

        Returns
        -------
        list[PPI]
            List of PPI instances.
        '''
        # Utils functions
        args = (*args, 'interaction', 'origin', 'partition')
        def instantiate(proteins: str) -> 'PPI':
            p1, p2 = proteins
            return PPI(p1, p2, *args)
        @cache
        def hash2protein(hash_: str) -> Protein:
            return Protein(hash_)

        # Fetch Protein files
        files = sorted(list(path.PPI.glob('*.p1')))
        protein_filenames = [file_name.stem.split('=') for file_name in files]
        logger.info(f'Unpickling Protein objects...')
        proteins = [(hash2protein(p1), hash2protein(p2)) for p1, p2 in tqdm(protein_filenames)]
        
        # Fetch PPI files
        ppis = []
        logger.info(f'Unpickling {len(proteins)} PPI objects...')
        with ThreadPoolExecutor(max_workers=50) as executor:
            for result in tqdm(executor.map(instantiate, proteins), total=len(proteins)):
                ppis.append(result)

        # Return as generator
        for ppi in tqdm(ppis):
            yield ppi

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

    def pickle(self) -> None:
        '''
        Pickles the content of the PPI instance to several files in the PPI 
        folder with the md5 hash of the sequences (separated by =) as the file
        stem. The extensions of the files are the same as the arguments
        passed to the PPI instance. 
        '''
        def save_data(key: str, data: Any, filepath: str):
            preserved_dicts = ['interface_features']
            if isinstance(data, dict) and k not in preserved_dicts:
                for key, value in data.items():
                    save_data('', value, f'{filepath}.{key}')
            else:
                if data is not None:
                    utils.pickle(data=data, path=filepath)
        file_stem = self.__hash__()
        self.p1 = self.p1.__hash__()
        self.p2 = self.p2.__hash__()
        for k, v in self.__dict__.items():
            save_data(k, v, f'./{file_stem}.{k}')

