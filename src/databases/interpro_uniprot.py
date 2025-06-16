"""
===============================================================================
Title:      InterPro UniProt
Outline:    InterProUniProt class to interact with the InterPro API given an
            UniProt ID and retrieve the start and end positions of InterPro
            accessions IDs domains found in the UniProt ID using many InterPro 
            source databases: InterPro, CDD, CathGene3D, Profile, Prints, 
            SMART, Prosite, PFAM, Panther, SSF, Hamap, Pirsf, and NCBIFam.
            The result is a dictionary with the InterPro accessions IDs as keys
            and a list of tuples with the start and end positions of the
            domains as values:
            {
                'IPR002100': [(0, 50), (100, 150)],
                'PF00319': [(60, 120)]
            }
Docs:       https://github.com/ProteinsWebTeam/interpro7-api
Author:     Alejandro Sánchez Cano
Date:       01/10/2024
===============================================================================
"""

# Built-in modules
from typing import Any
from collections import defaultdict

# Third-party modules
import requests

# Custom modules
from src.misc.logger import logger

class InterProUniProt:
    
    def __init__(self, uniprot_id: str):
        self.uniprot_id = uniprot_id

    def __repr__(self) -> str:
        '''
        Prints instance attributes for debugging purposes.

        Returns
        -------
        str
            Instance attribute mapping.
        '''
        return str(self.__dict__)
    
    def __request(self, url: str) -> dict[str, Any]:
        '''
        Given an InterPro API url, performs a request and returns the 
        response as a python-interactable JSON. Manages 200 and 204
        status codes, raising an exception for any other status code.

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
        status = request.status_code
        match status:
            case 200:
                return request.json()
            case 204:
                return {}
            case _:
                raise Exception(f'Error {status} in request')
    
    def get_domains(self) -> dict[str, list[tuple[int, int]]]:
        '''
        Uses the InterPro API to fetch the domains of a UniProt ID in 
        the InterPro database from a series of source databases like 
        InterPro, PFAM, etc. The response is carefully parsed and the 
        domains are stored in a dictionary with the accession as key 
        and a list of tuples with the start and end of the domain as 
        value.

        Returns
        -------
        dict[str, list[tuple[int, int]]]
            Dictionary with the accession as key and a list of tuples
            with the start and end of the domain as value.
        '''
        # Logging
        logger.info(f'Processing {self.uniprot_id}')

        # Dictionary to store domains
        domains = defaultdict(list)

        # Source databases to search for domains
        source_databases = [
            'interpro', 'cdd', 'cathgene3d', 'profile', 'prints', 'smart',
            'prosite', 'pfam', 'panther', 'ssf', 'hamap', 'pirsf', 'ncbifam'
            ]
        
        # Iterate over source databases
        for database in source_databases:
            # Make request
            url = f'https://www.ebi.ac.uk/interpro/api/entry/{database}/protein/uniprot/{self.uniprot_id}'
            logger.debug(f'Processing {database}')
            logger.debug(f'URL: {url}')
            json = self.__request(url)
            # Empty response
            if not json:
                continue
            # Pagination not implemented
            assert json['next'] is None, 'Pagination needs to be implemented'

            # Navigate JSON response
            for result in json['results']:
                accession = result['metadata']['accession']
                subdatabases = result['metadata']['member_databases']
                subdatabases = subdatabases.keys() if subdatabases else []
                assert all(subdatabase in source_databases for subdatabase in subdatabases), f'Unknown source database in {subdatabases} for {self.uniprot_id}'
                assert len(result['proteins']) == 1, f'Multiple proteins found in {accession} for {self.uniprot_id}'
                for location in result['proteins'][0]['entry_protein_locations']:
                    for fragment in location['fragments']:
                        
                        # Extract domain start and end
                        start = int(fragment['start']) - 1
                        end = int(fragment['end']) - 1
                        logger.debug(f'{self.uniprot_id} {result["metadata"]["accession"]} {start}-{end}')
                        
                        # Store domain
                        domains[accession] += [(start, end)]
        
        return domains

if __name__ == '__main__':
    '''Test class'''
    uniprot = InterProUniProt('A0A0B2NT15')
    domains = uniprot.get_domains()
    print(domains)