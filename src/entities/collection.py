"""
===============================================================================
Title:      Collection
Outline:    Collection class that represents a collection of entities such as
            Proteins, PPIs, etc. It hosts functionalities that concern 
            collections as a whole and not individual entities. It supports:
            - Multithreaded instantiation from a directory of pickled objects.
Author:     Alejandro Sánchez Cano
Date:       02/11/2025
===============================================================================
"""

# Built-in modules
from pathlib import Path
import concurrent.futures

# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc import utils
from src.misc.logger import logger
from src.entities.protein import Protein

class Collection:
    
    def __init__(self, dir: str | Path, class_type: str):
        self.dir = Path(dir)
        self.class_type = class_type

        def instantiate(file_path: Path):
            '''
            Instantiates an object of the specified class_type.

            Parameters
            ----------
            file_path : Path
                Object file path.

            Returns
            -------
            object
                Instantiated object of the specified class_type.
            '''
            obj = Protein(file_path)
            return obj

        proteins = []
        num_threads = 50
        files = sorted(list(self.dir.glob('*')))
        logger.info(f'Unpickling {len(files)} {class_type} objects from {self.dir}...')
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for result in tqdm(executor.map(instantiate, files), total=len(files)):
                proteins.append(result)

        self.proteins = proteins

    def __iter__(self):
        return iter(self.proteins)
        
if __name__ == "__main__":
    collection = CollectionParallel(dir = path.MIKC_PROTS, class_type = 'Protein')
    for item in collection:
        pass
    