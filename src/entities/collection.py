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
import json
from pathlib import Path
import concurrent.futures
from abc import ABC, abstractmethod
from collections import defaultdict

# Third-party modules
import h5py
import numpy as np
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.protein import Protein

# Lazy modules
#import seaborn as sns
#from matplotlib import pyplot as plt

class Collection(ABC):

    @abstractmethod
    def __init__(
        self, 
        file_path: str | Path | None = None, 
        items: list | None = None
        ):
        pass

    @abstractmethod
    def __iter__(self):
        pass 
    
    def __contains__(self, item) -> bool:
        return any(x == item for x in self)

    @abstractmethod
    def to_hdf5(self) -> None:
        pass

class ProteinCollection(Collection):
    
    def __init__(
        self, 
        file_path: str | Path | None = None, 
        items: list | None = None
        ):
        self.file_path = Path(file_path) if file_path else None
        self.proteins = items if items else []

    def __iter__(self):
        # Already loaded
        if self.proteins:
            for protein in tqdm(self.proteins, desc='Iterating over Protein collection'):
                yield protein

        # Load from HDF5
        with h5py.File(self.file_path, 'r') as file:
            seq_array = file['seq'][:]
            uniprot_array = file['uniprot'][:]
            taxon_array = file['taxon'][:]
            section_array = file['section'][:]
            primary_accession_array = file['primary_accession'][:]
            secondary_accessions_array = file['secondary_accessions'][:]
        
        # Yield Protein objects
        for idx in tqdm(range(len(seq_array)), desc='Loading Protein collection from HDF5'):
            protein = Protein(
                seq = seq_array[idx].decode('utf-8'),
                uniprot = uniprot_array[idx].decode('utf-8'),
                taxon = int(taxon_array[idx]),
                section = section_array[idx].decode('utf-8'),
                primary_accession = primary_accession_array[idx].decode('utf-8'),
                secondary_accessions = [string.decode('utf-8') for string in secondary_accessions_array[idx]]
            )
            self.proteins.append(protein)
            yield protein

    def to_hdf5(self) -> None:
        # Create HDF5 file
        with h5py.File(self.file_path, 'w') as file:
            # Iterate over attributes
            attributes = self.proteins[0].__dataclass_fields__.keys()
            for attr in attributes:
                # Skip attributes with all default values
                all_default = all(getattr(protein, attr) in (None, {}, []) for protein in self.proteins)
                if all_default: continue
                # Save non-default attributes
                match attr:
                    # Handle string attributes
                    case 'seq' | 'uniprot' | 'section' | 'primary_accession' | 'closest_arabidopsis':
                        dtype = h5py.string_dtype(encoding='utf-8')
                        array = np.array([getattr(protein, attr) for protein in self.proteins], dtype=dtype)
                        file.create_dataset(attr, data=array, compression="gzip", compression_opts=4, chunks=True)
                    # Handle integer attributes
                    case 'taxon':
                        dtype = np.int32
                        array = np.array([getattr(protein, attr) for protein in self.proteins], dtype=dtype)
                        file.create_dataset(attr, data=array, compression="gzip", compression_opts=4, chunks=True)
                    # Handle list of strings attributes
                    case 'secondary_accessions':
                        lst = [protein.secondary_accessions for protein in self.proteins]
                        dtype = h5py.vlen_dtype(h5py.string_dtype(encoding='utf-8'))
                        lst = [np.array(sublist, dtype=dtype) for sublist in lst]
                        array = np.array(lst, dtype=dtype)
                        file.create_dataset(attr, data=array, compression="gzip", compression_opts=4, chunks=True)
                    # Handle dictionary attributes
                    case 'interpro_domains':
                        lst = [json.dumps(protein.interpro_domains) for protein in self.proteins]
                        dtype = h5py.string_dtype(encoding='utf-8')
                        array = np.array(lst, dtype=dtype)
                        file.create_dataset(attr, data=array, compression="gzip", compression_opts=4, chunks=True)
                    # Handle special attributes
                    case 'esm2_embeddings': 
                        pass
                        # THIS WILL CHANGE!
                        #group = file.create_group('650M')
                        #array = np.random.rand(30, 200, 1280).astype(np.float32)
                        #group.create_dataset(attr, data=array, compression="gzip", compression_opts=4, chunks=True)

    def sequence_report(self) -> None:
        '''
        Prints and plots a report of the sequences in the collection.
        '''
        # Lazy imports
        import seaborn as sns
        from matplotlib import pyplot as plt
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
    collection = ProteinCollection(file_path = path.DATA / 'mikc_proteins.h5')
    prots = [p for p in collection]
