"""
===============================================================================
Title:      InterPro
Outline:    InterPro class to interact with the InterPro API given an
            InterPro accession ID  (e.g. IPR002100, PF00319, ...) or an
            InterPro domain architecture ID (IDA) (e.g. 
            9b1d1537f57a287fce1f0a861665d2876b831672).
            It allows to:
            - Access metadata:
                + Accession name 
                + Number of proteins 
                + Number of AlphaFold structures
                + Countes of proteins per IDA
            - Retrieve the number of UniProt accessions
Docs:       https://github.com/ProteinsWebTeam/interpro7-api
Author:     Alejandro Sánchez Cano
Date:       01/10/2024
===============================================================================
"""

# Built-in modules
import math
import time
from typing import Any
from abc import ABC, abstractmethod

# Third-party modules
import requests
from tqdm import tqdm

# Custom modules
from src.misc.logger import logger

class InterPro(ABC):

    # Static attribute
    url = "https://www.ebi.ac.uk/interpro/api"

    def __repr__(self) -> str:
        '''
        Prints instance attributes for debugging purposes.

        Returns
        -------
        str
            Instance attribute mapping.
        '''
        return str(self.__dict__)

    def _request(self, url: str) -> dict[str, Any]:
        '''
        Given an InterPro API url, performs a request and returns the 
        response as a python-interactable JSON. 

        Parameters
        ----------
        url : str
            InterPro API url.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response.
        '''
        request = requests.get(url)
        return request.json()

    @abstractmethod
    def get_uniprot(self) -> None:
        pass

class InterProAccession(InterPro):

    def __init__(self, accession: str = None):
        super().__init__()
        # Modified upon instantiation
        self.accession = accession
        self.source_database = self.__get_source_database() if self.accession else None

        # Modified with 'get_metadata()' method
        self.name = None
        self.number_of_proteins = None
        self.number_of_alphafolds = None

    def __get_source_database(self) -> str:
        '''
        Parses the Interpro accession ID to find what database it 
        belongs to within InterPro (CDD, Profile, Panther, PFAM ...). 
        It returns the corresponding string used in API requests.

        Returns
        -------
        str
            InterPro
        '''
        if self.accession.startswith('IPR'):
            return 'interpro'
        elif self.accession.startswith('cd'):
            return 'cdd'
        elif self.accession.startswith('G3DSA'):
            return 'cathgene3d'
        elif self.accession.startswith('P'):
            return 'profile'
    
    def get_metadata(self) -> None:
        '''
        Performs an InterPro API request to retrieve the name of the 
        accession and the number of UniProt and AlphaFold accessions it 
        contains. Updates 'self.name', 'self.number_of_proteins' and
        'self.number_of_alphafolds' attributes.
        '''
        # Request to InterPro API
        url = f'{self.__class__.url}/entry/interpro/{self.accession}'
        json = self._request(url)
        
        # Access metadata in API response
        self.name = json['metadata']['name']['name']
        self.number_of_proteins = json['metadata']['counters']['proteins']
        self.number_of_alphafolds = json['metadata']['counters']['structural_models']['alphafold']

        # Logging
        logger.info(f'{self.name=}')
        logger.info(f'{self.number_of_proteins=}')
        logger.info(f'{self.number_of_alphafolds=}')

    def get_uniprot(self, batch_size: int = 200) -> None:
        '''
        Use InterPro API to retrieve necessary the UniProt IDs of the proteins 
        belonging to the self.accession InterPro ID.
        It cannot be parallized due to API pagination.

        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 200
        '''
        # Initialize list of UniProt IDs
        uniprot_ids = []

        # InterPro API URL
        url = f'{self.__class__.url}/protein/uniprot/entry/{self.source_database}/{self.accession}?page_size={batch_size}'

        # Manage API pagination
        total_batches = math.ceil(self.number_of_proteins/batch_size)
        with tqdm(total = total_batches, desc="Accessing API URLs") as pbar:
            while url:
                
                # Page response JSON
                json = self._request(url)
                for result in json['results']:
                    
                    # Access protein info
                    uniprot_id = result['metadata']['accession']
                    
                    # Append to results list
                    uniprot_ids.append(uniprot_id)

                # Prepare for next batch
                url = json['next']
                pbar.update(1)
                time.sleep(0.2)

        # Logging
        logger.info(f'{len(uniprot_ids)} UniProt IDs retrieved for {self.accession}')
        
        return uniprot_ids

class InterProDomainArchitecture(InterPro):

    def __init__(self, ida: str = None):
        super().__init__()
        self.ida = ida

    def get_uniprot(self, batch_size: int = 200) -> None:
        '''
        Use InterPro API to retrieve necessary the UniProt IDs of the proteins 
        belonging to the self.ida InterPro domain architecture ID.

        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 200

        Returns
        -------
        list[str]
            List of UniProt IDs.
        '''
        # Initialize list of UniProt IDs
        uniprot_ids = []

        # InterPro API URL
        url = f'{self.__class__.url}/protein/uniprot/?ida={self.ida}&page_size={batch_size}'

        # Manage API pagination
        logger.warning('Number of proteins miscounted in API, tqdm bar total might be slightly inaccurate')
        with tqdm(desc="Accessing API URLs") as pbar:
            while url:
                
                # Page response JSON
                json = self._request(url)

                # Update tqdm total if first iteration
                if pbar.total is None:
                    total_batches = math.ceil(json['count']/batch_size)
                    pbar.total = total_batches
                    pbar.refresh()
                
                # Gather UniProt IDs
                for result in json['results']:
                    uniprot_id = result['metadata']['accession']
                    uniprot_ids.append(uniprot_id)

                # Prepare for next batch
                url = json['next']
                pbar.update(1)
                time.sleep(0.1)

        # Logging
        logger.info(f'{len(uniprot_ids)} UniProt IDs retrieved for {self.ida}')
        
        return uniprot_ids

if __name__ == '__main__':
    '''Test class'''
    # Test class with small InterPro ID 
    test = InterProAccession('IPR011364')
    test.get_metadata()
    uniprot_ids = test.get_uniprot()
    print(uniprot_ids[:10])