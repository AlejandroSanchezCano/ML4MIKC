"""
===============================================================================
Title:      Add UniProt data to Protein objects
Outline:    Uses the UniProt class to fetch the metadata, sequence, (and
            structure) of MIKC proteins using their UniProt IDs. The data is 
            stored in the Protein objects and pickled.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
Time:       3h
===============================================================================
"""


# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.protein import Protein
from src.databases.uniprot import UniProt, UniProtError
logger.setLevel(20)

# Load MIKC UniProt accessions
mikc_uniprots = []
file = path.DATA / 'm_and_k_uniprot_ids.txt'
with open(file, 'r') as handle:
    for line in handle:
        mikc_uniprots.append(line.strip())
logger.info(f'{len(mikc_uniprots)} MIKC UniProt accessions loaded')

# Iterate over UniProt accessions
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

    # Create Protein object
    protein = Protein(
        seq = seq,
        uniprot = uniprot.accession,
        taxon = taxon_id,
        section = section,
        primary_accession = primary_accession,
        secondary_accessions = secondary_accessions
        )
    
    # Save Protein object
    save_dir = path.DATA / 'MIKC_proteins'
    protein.pickle(dir = save_dir)

# Logging
n_prots = len(list(save_dir.glob('*')))
logger.info(f'{n_prots} MIKC Protein objects saved')
logger.info(f'{len(mikc_uniprots) - n_prots} inactive UniProt accessions skipped')