"""
===============================================================================
Title:      Add UniProt data to Protein objects
Outline:    Uses the UniProt class to fetch the metadata, sequence, (and
            structure) of M proteins using their UniProt IDs. The data is 
            stored in the Protein objects and saved.
            Some UniProt accessions are inactive and therefore skipped.
            Also, accessions with identical sequence and taxon are considered
            redundant and only one of them is kept (preferentially Swiss-Prot).
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
Time:       4h
===============================================================================
"""

# Built-in modules
from collections import defaultdict

# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.protein import Protein
from src.entities.collection import ProteinCollection
from src.families.uniprot import UniProt, UniProtError
logger.setLevel(20)
logger.info('Importing modules completed')

# Load M UniProt accessions
m_uniprots = []
file = path.DATA / 'm_uniprot_ids.txt'
with open(file, 'r') as handle:
    for line in handle:
        m_uniprots.append(line.strip())
logger.info(f'{len(m_uniprots)} M UniProt accessions loaded')

# Retrieve data per UniProt accession
proteins = []
inactive_uniprots = []
for uniprot in tqdm(m_uniprots, desc="Fetching UniProt data"):

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
        inactive_uniprots.append(uniprot.accession)
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

# Remove redundant Protein objects (keep Swiss-Prot if available)
nr_proteins = []
seqs2protein = defaultdict(list)
# Group Protein objects by (sequence, taxon)
for protein in proteins:
    seqs2protein[(protein.seq, protein.taxon)].append(protein)
# Select non-redundant Protein objects
for seq_taxon, prot_list in seqs2protein.items():
    # Only one Protein object
    if len(prot_list) == 1:
        nr_proteins.append(prot_list[0])
        continue
    # Multiple Protein objects: prefer Swiss-Prot
    sections = [prot.section for prot in prot_list]
    if 'Swiss-Prot' in sections:
        swiss_prot = prot_list[sections.index('Swiss-Prot')]
        nr_proteins.append(swiss_prot)
        continue
    # Otherwise, keep the first one (TrEMBL)
    nr_proteins.append(prot_list[0])

# Save Protein objects
file_path = path.DATA / 'm_proteins.h5'
collection = ProteinCollection(file_path=file_path, items=nr_proteins)
collection.to_hdf5()

# Logging
logger.info(f'{len(inactive_uniprots)} inactive UniProt accessions skipped')
logger.info(f'{len(proteins) - len(nr_proteins)} redundant M Protein objects skipped')
logger.info(f'{len(nr_proteins)} M Protein objects saved')
