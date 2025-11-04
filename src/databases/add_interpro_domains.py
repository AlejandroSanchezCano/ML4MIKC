"""
===============================================================================
Title:      Find protein domains in InterPro
Outline:    Uses the UniProt class to fetch from the InterPro API the 
            InterPro domains of MIKC proteins using their UniProt IDs. The 
            obtained domains are stored in a Protein object and pickled.
            Multithreading is implemented to speed up the process, and it can
            take up from 3h to 8h, depending on the API speed and benevolence.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
Time:       3h 20min but depends a lot on API speed and benevolence
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
from src.databases.uniprot import UniProt
from src.entities.protein import Protein
from src.entities.collection import Collection
logger.setLevel(20)

async def main():
    # Limit concurrent requests
    semaphore = asyncio.Semaphore(10)  

    # Gather tasks
    collection = Collection(dir = path.MIKC_PROTS, class_type = Protein)
    proteins = [protein for protein in collection]
    uniprots = [UniProt(protein.uniprot) for protein in proteins]
    tasks = [uniprot.interpro_domains(semaphore) for uniprot in uniprots]

    # Collect fetched domains
    results = {}
    for coroutine in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        uniprot_accession, domains = await coroutine
        results[uniprot_accession] = domains

    # Save domains in Protein objects
    for protein in tqdm(proteins, desc='Saving protein objects with domains'):
        protein.interpro_domains = dict(results[protein.uniprot]) # Defaultdicts are not compatible with dataclasses.asdict
        protein.pickle(dir = path.MIKC_PROTS)

# Run event loop
results = asyncio.run(main())