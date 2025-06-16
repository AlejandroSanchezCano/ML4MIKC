"""
===============================================================================
Title:      UniProt API
Outline:    UniProt class to use the bioservices UniProt API to fetch the
            sequence, structure, and metadata (taxon ID, TrEMBL/Swiss-Prot,
            primary, and secondary accession) of a UniProt ID.
Docs:       https://bioservices.readthedocs.io/en/main/references.html#bioservices.uniprot.UniProt
Author:     Alejandro Sánchez Cano
Date:       03/10/2025
===============================================================================
"""

# Built-in modules
import subprocess

# Third-party modules
from bioservices.uniprot import UniProt as UniProtAPI

# Custom modules
from src.misc.logger import logger

class UniProtError(Exception):
    '''Custom exception for the UniProt class.'''
    pass

class UniProt:

    # Initialize UniProt API
    uniprot_api = UniProtAPI(verbose = False)

    def __init__(self, uniprot_id: str):
        self.uniprot_id = uniprot_id
    
    def fetch_metadata(self) -> tuple[int, str, str, str]:
        '''
        Fetches the metadata of a UniProt ID using the bioservices
        UniProt API. The metadata includes the taxon ID, the section
        (TrEMBL/Swiss-Prot), the primary accession, and the secondary
        accessions. If the UniProt ID is inactive, a custom exception
        is raised.

        Returns
        -------
        tuple[int, str, str, str]
            Taxon ID, section, primary accession, and secondary accessions.
        '''

        # Use UniProt API
        entry_json = UniProt.uniprot_api.retrieve(
            uniprot_id = self.uniprot_id,
            frmt = 'json',
            database = 'uniprot'
            )

        # Handle inactice UniProt IDs
        if entry_json['entryType'] == 'Inactive':
            raise UniProtError(f'{self.uniprot_id} is an inactive UniProt ID')

        # Parse metadata
        taxon_id = int(entry_json['organism']['taxonId'])
        section = 'TrEMBL' if entry_json['entryType'].endswith('(TrEMBL)') else 'Swiss-Prot'
        primary_accession = entry_json.get('primaryAccession', '')
        secondary_accession = entry_json.get('secondaryAccessions', [])

        # Logging
        logger.debug(f'Metadata fetched for {self.uniprot_id}')

        return taxon_id, section, primary_accession, secondary_accession
    
    def fetch_sequence(self) -> str:
        '''
        Fetches the sequence of a UniProt ID using the bioservices
        UniProt API. It is faster than down loading the FASTA file with
        curl/wget.

        Returns
        -------
        str
            Sequence of the UniProt ID.
        '''
        # Fetch sequence
        fasta = UniProt.uniprot_api.get_fasta(self.uniprot_id)
        sequence = fasta.split('\n', 1)[1].replace('\n', '')

        # Logging
        logger.debug(f'Sequence fetched for {self.uniprot_id}')
        return sequence
    
    def fetch_structure(self) -> str:
        '''
        Downloads the structure of a UniProt ID from the AlphaFold
        Protein Structure Database.

        Returns
        -------
        str
            Structure of the UniProt ID.
        '''
        # Download structure
        cmd = f'curl "https://alphafold.ebi.ac.uk/files/AF-{self.uniprot_id}-F1-model_v4.pdb"'
        response = subprocess.run(cmd, capture_output = True, text = True, shell = True).stdout
        status = 404 if response.endswith('</Error>') else 200

        # Logging
        logger.debug(f'Structure fetched for {self.uniprot_id}')

        # Only save the non-empty responses (> 200 characters)
        return response if status == 200 else ''

if __name__ == '__main__':
    '''Test class'''
    uniprot = UniProt('P48007')
    print(uniprot.fetch_metadata())
    print(uniprot.fetch_sequence())
    print(uniprot.fetch_structure())
    uniprot = UniProt('A0A2H5NPF5') # Inactive UniProt ID -> raises UniProtError
    print(uniprot.fetch_metadata())
    print(uniprot.fetch_sequence())
    print(uniprot.fetch_structure())
    