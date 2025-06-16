"""
===============================================================================
Title:      InterPro Database
Outline:    InterPro Database class to interact with the InterPro API to 
            retrieve metadata (accession name and number of proteins and 
            AlphaFold structures) and the UniProt accessions associatd with 
            an InterPro accession IDs (e.g. IPR002100, PF00319, ...).
Docs:       https://github.com/ProteinsWebTeam/interpro7-api
Author:     Alejandro Sánchez Cano
Date:       01/10/2024
===============================================================================
"""

# Built-in modules
import math
import time
from typing import Any

# Third-party modules
import requests
from tqdm import tqdm

# Custom modules
from src.misc.logger import logger

class InterProDatabase:

    # Static attributes
    url = "https://www.ebi.ac.uk/interpro/api"

    def __init__(self, accession: str):
        # Modified upon instantiation
        self.accession = accession
        self.source_database = self.__get_source_database()

        # Modified with 'get_metadata()' method
        self.name = ''
        self.number_of_proteins = 0
        self.number_of_alphafolds = 0

    def __repr__(self) -> str:
        '''
        Prints instance attributes for debugging purposes.

        Returns
        -------
        str
            Instance attribute mapping.
        '''
        return str(self.__dict__)

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
    
    def __request(self, url: str) -> dict[str, Any]:
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
    
    def get_metadata(self) -> None:
        '''
        Performs an InterPro API request to retrieve the name of the 
        accession and the number of UniProt and AlphaFold accessions it 
        contains. Updates 'self.name', 'self.number_of_proteins' and
        'self.number_of_alphafolds' attributes.
        '''
        # Request to InterPro API
        url = f'{InterProDatabase.url}/entry/interpro/{self.accession}'
        json = self.__request(url)
        
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
        Use InterPro API to retrieve necessary the UniProt IDs, taxons 
        and domains of the proteins belonging to the self.accession 
        InterPro ID.

        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 200
        '''
        # Initialize list of UniProt IDs
        uniprot_ids = []

        # InterPro API URL
        url = f'{InterProDatabase.url}/protein/uniprot/entry/{self.source_database}/{self.accession}?page_size={batch_size}'

        # Manage API pagination
        total_batches = math.ceil(self.number_of_proteins/batch_size)
        with tqdm(total = total_batches) as pbar:
            while url:
                
                # Page response JSON
                json = self.__request(url)
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

if __name__ == '__main__':
    '''Test class'''
    # Test class with small InterPro ID 
    test = InterProDatabase('IPR011364')
    test.get_metadata()
    uniprot_ids = test.get_uniprot()
    print(uniprot_ids[:10])