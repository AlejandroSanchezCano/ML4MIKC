
# Built-in modules
import concurrent.futures

# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.databases.uniprot import UniProt
from src.entities.protein import Protein
from src.entities.collection import Collection
logger.setLevel(20)

# Iterate over Protein objects
collection = Collection(dir = path.MIKC_PROTS, class_type = 'Protein')
proteins = [protein for protein in collection]

def add_domains(protein: Protein) -> dict[str, tuple[int, int, str]]:
    '''
    Fetches InterPro domains for a given UniProt accessions. And adds them to 
    the corresponding Protein object.

    Parameters
    ----------
    protein : Protein
        Protein object.

    Returns
    -------
    dict[str, tuple[int, int, str]]
        Mapping of domain ID to (start, end, description).
    '''
    if not protein.interpro_domains:
        uniprot = UniProt(protein.uniprot)
        domains = uniprot.interpro_domains()
        protein.interpro_domains = dict(domains) # Defaultdicts are not compatible with dataclasses.asdict
        protein.pickle(dir = path.MIKC_PROTS)

# Multithreading
num_threads = 7
with concurrent.futures.ThreadPoolExecutor(max_workers = num_threads) as executor:
    list(tqdm(executor.map(add_domains, proteins), total=len(proteins)))