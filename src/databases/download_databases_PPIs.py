"""
===============================================================================
Title:      Download data from PPI databases
Outline:    Downloads the data from the IntAct, BioGRID, and PlaPPISite
            databases, reduces it to plant interactors if neccessary, 
            filters it to only MADS-MADS interactions. Save their UniProt
            accessions to a text file.
Docs:       https://downloads.thebiogrid.org/BioGRID
            https://ftp.ebi.ac.uk/pub/databases/intact/
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
Time:       - BioGrid: 3 min
            - IntAct: 5 min
            - PlaPPISite: 20 min
===============================================================================
"""
# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.databases.intact import IntAct
from src.families.uniprot import UniProt
from src.databases.biogrid import BioGRID
from src.databases.plappisite import PlaPPISite
from src.entities.collection import ProteinCollection
logger.info('Importing modules completed')

# Gather UniProt IDs of MADS-box proteins
file_path = path.DATA / 'mikc_proteins.h5'
collection = ProteinCollection(
    file_path=file_path, 
    datasets=['uniprot', 'taxon', 'seq']
    )
mads_uniprots = set([protein.uniprot for protein in collection])
uniprot2protein = {protein.uniprot: protein for protein in collection}

# BioGRID
biogrid = BioGRID('5.0.251')
#biogrid.download_files()
#biogrid.reduce_to_plants()
#biogrid.mads_vs_all(mads_uniprots)
#biogrid.mads_vs_mads(mads_uniprots)
biogrid_ppi_uniprot_accessions = biogrid.ppi_uniprot_accessions()

## IntAct
intact = IntAct('2025-08-08') 
#intact.download_files()
#intact.reduce_to_plants()
#intact.mads_vs_all(mads_uniprots)
#intact.mads_vs_mads(mads_uniprots)
intact_ppi_uniprot_accessions = intact.ppi_uniprot_accessions()

# PlaPPISite
plappisite = PlaPPISite()
#plappisite.mads_vs_all(mads_uniprots)
#plappisite.mads_vs_mads(mads_uniprots)
plappisite_ppi_uniprot_accessions = plappisite.ppi_uniprot_accessions()

# Save
ppi_uniprot_accessions = {
    'IntAct': intact_ppi_uniprot_accessions, 
    'BioGRID': biogrid_ppi_uniprot_accessions, 
    'PlaPPISite': plappisite_ppi_uniprot_accessions
    }

# Create dataframe
df = {
    'Database': [],
    'UniProt_A': [],
    'UniProt_B': [],
    'Taxon_A': [],
    'Taxon_B': [],
    'Seq_A': [],
    'Seq_B': []
}

# Fetch data from UniProt
for database in tqdm(ppi_uniprot_accessions, desc='Obtaining UniProt data'):
    for uniprot_A, uniprot_B in ppi_uniprot_accessions[database]:
        # Get taxon and seq data
        if uniprot_A in uniprot2protein:
            taxon_A = uniprot2protein[uniprot_A].taxon
            seq_A = uniprot2protein[uniprot_A].seq
        else: 
            uniprot = UniProt(uniprot_A)
            taxon_A = uniprot.fetch_metadata()[0]
            seq_A = uniprot.fetch_sequence()
        if uniprot_B in uniprot2protein:
            taxon_B = uniprot2protein[uniprot_B].taxon
            seq_B = uniprot2protein[uniprot_B].seq
        else:
            uniprot = UniProt(uniprot_B)
            taxon_B = uniprot.fetch_metadata()[0]
            seq_B = uniprot.fetch_sequence()
        # Sort based on seq
        if seq_A < seq_B:
            A = (uniprot_A, taxon_A, seq_A)
            B = (uniprot_B, taxon_B, seq_B)
        else:
            A = (uniprot_B, taxon_B, seq_B)
            B = (uniprot_A, taxon_A, seq_A)
        # Append to df
        df['Database'].append(database)
        df['UniProt_A'].append(A[0])
        df['Taxon_A'].append(A[1])
        df['Seq_A'].append(A[2])
        df['UniProt_B'].append(B[0])
        df['Taxon_B'].append(B[1])
        df['Seq_B'].append(B[2])

# Save DataFrame
df = pd.DataFrame(df)
file_path = path.DATABASES / 'ppi_uniprot_accessions.txt'
df.to_csv(file_path, sep = '\t', index = False)
logger.info(f'Saved PPI UniProt accessions data -> dim({df.shape})')