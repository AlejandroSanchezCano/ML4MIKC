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
from collections import defaultdict

# Third-party modules
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

# Custom modules
from src.misc import path
from src.misc import utils
from src.misc.logger import logger
from src.entities.protein import Protein

class Collection:
    
    def __init__(
        self, 
        dir: str | Path, 
        class_type: type, 
        num_threads: int = 50
        ):
        self.dir = Path(dir)
        self.class_type = class_type
        self.num_threads = num_threads
        self.items = self._load_items()

    def _load_items(self):
        items = []
        files = sorted(list(self.dir.glob('*')))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for result in tqdm(
                executor.map(self._instantiate, files), 
                total=len(files),
                desc=f'Unpickling {len(files)} {self.class_type.__name__}'
                ):
                items.append(result)

        return items

    def _instantiate(self, file_path: Path):
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
        obj = self.class_type(file_path)
        return obj

    def __iter__(self):
        return iter(tqdm(self.items, desc=f'Iterating over {self.class_type.__name__} collection'))


class ProteinCollection(Collection):
    
    def __init__(self, dir: str | Path):
        super().__init__(dir, class_type=Protein)

    def sequence_report(self) -> None:
        '''
        Prints and plots a report of the sequences in the collection.
        '''
        # Calculate lengths
        lengths = [len(protein.seq) for protein in self.items]
        # Logging
        logger.info(f'Number of proteins: {len(lengths)}')
        logger.info(f'Min length: {min(lengths)}')
        logger.info(f'Max length: {max(lengths)}')
        logger.info(f'Mean length: {sum(lengths) / len(lengths):.2f}')
        # Plot length distribution
        sns.kdeplot(lengths)
        plt.savefig('sequence_length_distribution.png')

    def fasta(self, out_path: str | Path, header_attributes: list[str]) -> None:
        '''
        Exports all Protein sequences in the collection to a FASTA file. The
        headers include specified attributes separated by '|'.

        Parameters
        ----------
        out_path : str | Path
            Output FASTA file path.
        
        header_attributes : list[str]
            List of Protein attributes to include in the FASTA header.
        '''
        out_path = Path(out_path)
        with out_path.open('w') as f:
            for protein in self.items:
                values = [str(getattr(protein, attr)) for attr in header_attributes]
                header = '|'.join(values)
                f.write(f'>{header}\n')
                f.write(f'{protein.seq}\n')

    def fastas_per_species(self, out_dir: str | Path, header_attributes: list[str]) -> None:
        '''
        Exports Protein sequences in the collection to separate FASTA files
        per species. The headers include specified attributes separated by '|'.
        This serves as input for OrthoFinder.

        Parameters
        ----------
        out_dir : str | Path
            Output directory for FASTA files.
        
        header_attributes : list[str]
            List of Protein attributes to include in the FASTA header.
        '''
        # Create output directory
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Group proteins by species
        species2proteins = defaultdict(list)
        for protein in self.items:
            species2proteins[protein.taxon].append(protein)


        for taxon, proteins in species2proteins.items():
            out_path = out_dir / f'{taxon}.fasta'
            with out_path.open('w') as f:
                for protein in proteins:
                    values = [str(getattr(protein, attr)) for attr in header_attributes]
                    header = '|'.join(values)
                    f.write(f'>{header}\n')
                    f.write(f'{protein.seq}\n')

if __name__ == "__main__":
    collection = ProteinCollection(dir = path.MIKC_PROTS)
    for item in collection:
        pass
    collection.fastas_per_species('lol', ['uniprot', 'taxon', ])
    collection.sequence_report()