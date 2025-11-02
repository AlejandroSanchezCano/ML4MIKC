"""
===============================================================================
Title:      UniProt
Outline:    UniProt class to represent a UniProt accession ID. It supports:
            - UniProt API wrapper:
                + Fetch metadata (taxon ID, TrEMBL/Swiss-Prot, primary and
                  secondary accessions)
                + Fetch sequence
                + Fetch structure from AlphaFold DB
            - InterPro API wrapper:
                + Fetch start and end of domains from InterPro and its source
                  databases: InterPro, CDD, CathGene3D, Profile, Prints, SMART,
                  Prosite, PFAM, Panther, SSF, Hamap, Pirsf, and NCBIFam. For
                  example:
                  {
                     'IPR002100': [(0, 50), (100, 150)],
                     'PF00319': [(60, 120)]
                  }
Docs:       https://bioservices.readthedocs.io/en/main/references.html#bioservices.uniprot.UniProt
            https://github.com/ProteinsWebTeam/interpro7-api
Author:     Alejandro Sánchez Cano
Date:       02/11/2025
===============================================================================
"""
# Built-in modules
import subprocess
from collections import defaultdict

# Third-party modules
import requests
from bioservices.uniprot import UniProt as UniProtAPI

# Custom modules
from src.misc.logger import logger

class UniProtError(Exception):
    '''Custom exception for the UniProt class.'''
    pass

class UniProt:

    # Initialize UniProt API
    api = UniProtAPI(verbose = False)

    def __init__(self, accession: str):
        self.accession = accession

    def __repr__(self) -> str:
        '''
        Prints instance attributes for debugging purposes.

        Returns
        -------
        str
            Instance attribute mapping.
        '''
        return str(self.__dict__)

    def fetch_metadata(self) -> tuple[int, str, str, str]:
        '''
        Fetches metadata of a UniProt accession using the bioservices UniProt
        API. The metadata includes:
        - Taxon ID
        - TrEMBL/Swiss-Prot section
        - Primary accession
        - Secondary accessions
        If the UniProt accession is inactive, a custom exception is raised.

        Returns
        -------
        tuple[int, str, str, str]
            Taxon ID, section, primary accession, and secondary accessions.
        '''
        # Use UniProt API
        entry_json = self.__class__.api.retrieve(
            uniprot_id = self.accession,
            frmt = 'json',
            database = 'uniprot'
            )

        # Handle inactive UniProt IDs
        if entry_json['entryType'] == 'Inactive':
            raise UniProtError(f'{self.accession} is an inactive UniProt ID')

        # Parse metadata
        taxon_id = int(entry_json['organism']['taxonId'])
        section = 'TrEMBL' if entry_json['entryType'].endswith('(TrEMBL)') else 'Swiss-Prot'
        primary_accession = entry_json.get('primaryAccession', '')
        secondary_accession = entry_json.get('secondaryAccessions', [])

        # Logging
        logger.debug(f'Metadata fetched for {self.accession}')

        return taxon_id, section, primary_accession, secondary_accession

    def fetch_sequence(self) -> str:
        '''
        Fetches the sequence of a UniProt accession using the bioservices
        UniProt API. It is faster than downloading the FASTA file with
        curl/wget.

        Returns
        -------
        str
            Sequence of the UniProt ID.
        '''
        # Fetch sequence
        fasta = self.__class__.api.get_fasta(self.accession)
        sequence = fasta.split('\n', 1)[1].replace('\n', '')

        # Logging
        logger.debug(f'Sequence fetched for {self.accession}')

        return sequence   

    def fetch_structure(self) -> str:
        '''
        Downloads the structure of a UniProt accession from the AlphaFold
        Protein Structure Database.

        Returns
        -------
        str
            Structure of the UniProt ID.
        '''
        # Download structure
        cmd = f'curl "https://alphafold.ebi.ac.uk/files/AF-{self.accession}-F1-model_v4.pdb"'
        response = subprocess.run(cmd, capture_output = True, text = True, shell = True).stdout
        status = 404 if response.endswith('</Error>') else 200

        # Logging
        logger.debug(f'Structure fetched for {self.accession}')

        # Only save the non-empty responses (> 200 characters)
        return response if status == 200 else ''         
    
    def interpro_domains(self) -> dict[str, list[tuple[int, int]]]:
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
        logger.info(f'Processing {self.accession}...')

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
            url = f'https://www.ebi.ac.uk/interpro/api/entry/{database}/protein/uniprot/{self.accession}'
            logger.debug(f'Processing {database}...')
            logger.debug(f'URL: {url}')
            request = requests.get(url)
            status = request.status_code
            match status:
                case 200: json = request.json()
                case 204: json = {}
                case _:
                    raise ValueError(f'Unexpected status code {status} for {url}')

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
                assert all(subdatabase in source_databases for subdatabase in subdatabases), f'Unknown source database in {subdatabases} for {self.accession}'
                assert len(result['proteins']) == 1, f'Multiple proteins found in {accession} for {self.accession}'
                for location in result['proteins'][0]['entry_protein_locations']:
                    for fragment in location['fragments']:
                        
                        # Extract domain start and end
                        start = int(fragment['start']) - 1
                        end = int(fragment['end']) - 1
                        logger.debug(f'{self.accession} {result["metadata"]["accession"]} {start}-{end}')
                        
                        # Store domain
                        domains[accession] += [(start, end)]
        
        return domains

if __name__ == '__main__':
    '''Test class'''
    uniprot = UniProt('P48007')
    print(uniprot.fetch_metadata())
    print(uniprot.fetch_sequence())
    print(uniprot.fetch_structure())
    print(uniprot.interpro_domains())
    uniprot = UniProt('A0A2H5NPF5') # Inactive UniProt ID -> raises UniProtError
    print(uniprot.fetch_metadata())
    print(uniprot.fetch_sequence())
    print(uniprot.fetch_structure())
    print(uniprot.interpro_domains())