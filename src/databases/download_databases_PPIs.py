"""
===============================================================================
Title:      Download data from PPI databases
Outline:    Downloads the data from the IntAct, BioGRID, and PlaPPISite
            databases, reduces it to plant interactors if neccessary, 
            filters it to only MADS-MADS interactions, and standardizes it
            to a common format used by the Network class. 
Docs:       https://downloads.thebiogrid.org/BioGRID
            https://ftp.ebi.ac.uk/pub/databases/intact/
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
Time:       - BioGrid: 3 min
            - IntAct: 5 min
            - PlaPPISite: 20 min
===============================================================================
"""

# Custom modules
from src.families.uniprot import UniProt
from src.databases.intact import IntAct
from src.databases.biogrid import BioGRID
from src.databases.plappisite import PlaPPISite
from src.entities.collection import ProteinCollection

# Custom modules
from src.misc import path
from src.misc.logger import logger
logger.info('Importing modules completed')

# Gather UniProt IDs of MADS-box proteins
file_path = path.DATA / 'mikc_proteins.h5'
collection = ProteinCollection(file_path=file_path, datasets=['uniprot'])
mads_uniprots = set([protein.uniprot for protein in collection])

# BioGRID
biogrid = BioGRID('5.0.251')
biogrid.download_files()
biogrid.reduce_to_plants()
biogrid.mads_vs_all(mads_uniprots)
biogrid.mads_vs_mads(mads_uniprots)
biogrid_ppi_uniprot_accessions = biogrid.ppi_uniprot_accessions()

## IntAct
intact = IntAct('2025-08-08') 
intact.download_files()
intact.reduce_to_plants()
intact.mads_vs_all(mads_uniprots)
intact.mads_vs_mads(mads_uniprots)
intact_ppi_uniprot_accessions = intact.ppi_uniprot_accessions()

# PlaPPISite
plappisite = PlaPPISite()
plappisite.mads_vs_all(mads_uniprots)
plappisite.mads_vs_mads(mads_uniprots)
plappisite_ppi_uniprot_accessions = plappisite.ppi_uniprot_accessions()

# Save
ppi_uniprot_accessions = {
    'BioGRID': biogrid_ppi_uniprot_accessions, 
    'IntAct': intact_ppi_uniprot_accessions, 
    'PlaPPISite': plappisite_ppi_uniprot_accessions
    }
filepath = path.DATABASES / 'ppi_uniprot_accessions.txt'
with open(filepath, 'w') as f:
    for database, (uniprot_A, uniprot_B) in ppi_uniprot_accessions.items():
        f.write(f'{database}\t{uniprot_A}\t{uniprot_B}\n')
