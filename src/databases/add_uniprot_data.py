"""
===============================================================================
Title:      Add UniProt data to Interactor objects
Outline:    Uses the UniProt class to fetch the metadata, sequence, and
            structure of MIKC proteins using their UniProt IDs. The data is 
            stored in the Interactor objects and pickled.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
Time:       3h
===============================================================================
"""

# Built-in modules
import logging

# Custom modules
from src.misc.logger import logger
from src.databases.uniprot import UniProt, UniProtError
from src.entities.interactor import Interactor
logger.setLevel(logging.INFO)

for interactor in Interactor.iterate():

    # Logging
    logger.info(f'Fetching UniProt data for {interactor.uniprot_id}')

    # Initialize UniProt object
    uniprot = UniProt(interactor.uniprot_id)

    # Fetch data
    try:
        taxon_id, section, primary_accession, secondary_accession = uniprot.fetch_metadata()
        sequence = uniprot.fetch_sequence()
        structure = uniprot.fetch_structure()
    except UniProtError:
        interactor.section = 'Inactive'
        logger.error(f'{interactor.uniprot_id} is an inactive UniProt ID')
        continue

    # Add data to Interactor object
    interactor.taxon_id = taxon_id
    interactor.section = section
    interactor.primary_accession = primary_accession
    interactor.secondary_accession = secondary_accession
    interactor.seq = sequence
    interactor.structure = structure

    # Save Interactor object
    interactor.pickle()