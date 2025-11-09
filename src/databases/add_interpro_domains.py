"""
===============================================================================
Title:      Find protein domains in InterPro
Outline:    Uses the UniProt class to fetch from the InterPro API the 
            InterPro domains of MIKC proteins using their UniProt IDs. The 
            obtained domains are stored in a Protein object and saved.
            Coroutines with asyncio are implemented to speed up the process.
Author:     Alejandro Sánchez Cano
Date:       04/11/2025
Time:       25 min
===============================================================================
"""

# Built-in modules
import asyncio
import concurrent.futures

# Third-party modules
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.protein import Protein
from src.databases.uniprot import UniProt
from src.entities.collection import ProteinCollection
logger.setLevel(20)
logger.info('Importing modules completed')

async def main() -> tuple[dict[str, dict], list[Protein]]:
    '''
    Function that wraps main functionality.

    Returns
    -------
    tuple[dict[str, dict], list[Protein]]
        A tuple containing a dictionary mapping UniProt accessions to their 
        InterPro domains and a list of Protein objects.
    '''
    # Limit concurrent requests
    semaphore = asyncio.Semaphore(10)  

    # Gather tasks
    file_path = path.DATA / 'mikc_proteins.h5'
    collection = ProteinCollection(file_path=file_path)
    proteins = [protein for protein in collection]
    uniprots = [UniProt(protein.uniprot) for protein in proteins]
    tasks = [uniprot.interpro_domains(semaphore) for uniprot in uniprots]

    # Collect fetched domains
    results = {}
    for coroutine in tqdm_asyncio.as_completed(
        tasks, 
        total=len(tasks),
        desc='Fetching InterPro domains'
        ):
        uniprot_accession, domains = await coroutine
        results[uniprot_accession] = domains
    
    return results, proteins

# Run event loop
results, proteins = asyncio.run(main())

# Save domains in Protein objects
for protein in tqdm(proteins, desc='Saving protein objects with domains'):
    protein.interpro_domains = dict(results[protein.uniprot])
collection = ProteinCollection(
    file_path=path.DATA / 'mikc_proteins.h5', 
    items=proteins
    )
collection.to_hdf5()
