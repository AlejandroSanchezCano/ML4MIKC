"""
===============================================================================
Title:      Find protein domains in InterPro
Outline:    Uses the InterProUniProt class to fetch from the InterPro API the 
            InterPro domains of MIKC proteins using their UniProt IDs, which
            are stored in a file. The domains are stored in an Interactor
            object and pickled. Multithreading is implemented to speed up the
            process, but it is thought that it does not make a real difference.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
Time:       3h 20min
===============================================================================
"""

# Built-in modules
import logging
import concurrent.futures

# Third party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.interactor import Interactor
from src.databases.interpro_uniprot import InterProUniProt
logger.setLevel(logging.INFO)

# Load MIKC UniProt IDs
mikc_uniprot_ids = []
with open(path.DATA / 'm_and_k_uniprot_ids.txt', 'r') as handle:
    for line in handle:
        mikc_uniprot_ids.append(line.strip())

# Function that will be executed in parallel
def fetch_domains(uniprot_id: str) -> None:
    '''
    Given a UniProt ID, fetches the InterPro domains and stores them in
    an Interactor object. The Interactor object is then pickled.

    Parameters
    ----------
    uniprot_id : str
        UniProt ID to fetch domains of.
    '''
    # Get domains
    protein = InterProUniProt(uniprot_id)
    domains = protein.get_domains()

    # Add domains to Interactor object
    interactor = Interactor(uniprot_id)
    interactor.domains = domains
    interactor.pickle()

# Manage multithreading
num_threads = 20
with concurrent.futures.ThreadPoolExecutor(max_workers = num_threads) as executor:
    list(tqdm(executor.map(fetch_domains, mikc_uniprot_ids), total=len(mikc_uniprot_ids)))