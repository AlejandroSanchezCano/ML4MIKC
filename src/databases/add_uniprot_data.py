"""
===============================================================================
Title:      Add UniProt data to Protein objects
Outline:    Uses the UniProt class to fetch the metadata, sequence, (and
            structure) of MIKC proteins using their UniProt IDs. The data is 
            stored in the Protein objects and pickled.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
Time:       3h 30min
===============================================================================
"""

# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.protein import Protein
from src.entities.collection import ProteinCollection
from src.databases.uniprot import UniProt, UniProtError
logger.setLevel(20)
logger.info('Importing modules completed')

# Load MIKC UniProt accessions
mikc_uniprots = []
file = path.DATA / 'm_and_k_uniprot_ids.txt'
with open(file, 'r') as handle:
    for line in handle:
        mikc_uniprots.append(line.strip())
logger.info(f'{len(mikc_uniprots)} MIKC UniProt accessions loaded')

# Retrieve data per UniProt accession
proteins = []
for uniprot in tqdm(mikc_uniprots, desc="Fetching UniProt data"):

    # Logging
    logger.info(f'Fetching UniProt data for accession {uniprot}...')
    
    # Initialize UniProt object
    uniprot = UniProt(uniprot)
    
    # Fetch UniProt metadata
    try:
        metadata = uniprot.fetch_metadata()
        taxon_id, section, primary_accession, secondary_accessions = metadata
        seq = uniprot.fetch_sequence()
    except UniProtError:
        logger.warning(f'{uniprot.accession} is inactive, skipping...')
        continue

    # Create and append Protein object
    protein = Protein(
        seq = seq,
        uniprot = uniprot.accession,
        taxon = taxon_id,
        section = section,
        primary_accession = primary_accession,
        secondary_accessions = secondary_accessions
        )
    proteins.append(protein)

# Save Protein objects
file_path = path.DATA / 'mikc_proteins.h5'
collection = ProteinCollection(file_path=file_path, items=proteins)
collection.to_hdf5()

# Logging
logger.info(f'{len(proteins)} MIKC Protein objects saved')
logger.info(f'{len(mikc_uniprots) - len(proteins)} inactive UniProt accessions skipped')