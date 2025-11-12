"""
===============================================================================
Title:      Collection
Outline:    Collection class that represents a collection of entities such as
            Proteins, PPIs, etc. It hosts functionalities that concern 
            collections as a whole and not individual entities.
            + ProteinCollection:
              - Load and save Protein objects from/to HDF5 files.
              - Iterate over Protein objects in the collection.
              - Generate sequence reports (length distributions, etc).
              - Export sequences to FASTA files (single or per species).
Author:     Alejandro Sánchez Cano
Date:       02/11/2025
===============================================================================
"""

# Built-in modules
import json
from pathlib import Path
import concurrent.futures
from typing import Iterable
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
        items: list | None = None,
        datasets: list[str] | str | None = None,
        limit: int | None = None
        ):
        self.file_path = Path(file_path) if file_path else None
        self.items = items if items else []
        self.limit = limit
        self.datasets = datasets 

    def _should_load(self, dataset: str, file: h5py.File) -> bool:
        '''
        Determines whether a specific dataset should be loaded based on the
        datasets attribute provided at initialization.

        Parameters
        ----------
        dataset : str
            Name of the dataset to check.
        file : h5py.File
            HDF5 file object.

        Returns
        -------
        bool
            True if the dataset should be loaded, False otherwise.
        '''
        # All datasets
        if self.datasets is None:
            return dataset in list(file.keys())
        # Specific datasets
        if isinstance(self.datasets, list):
            return dataset in self.datasets and dataset in list(file.keys())
        # Lightweight datasets
        if self.datasets == 'lightweight':
            lightweight_datasets = [
                'seq', 'uniprot', 'taxon', 'section', 
                'primary_accession', 'secondary_accessions', 
                'interpro_domains', 'closest_arabidopsis'
            ]
            return dataset in lightweight_datasets

    def __iter__(self) -> Iterable[Protein]:
        '''
        If items are already in memory either because they were provided at 
        initialization or because they were previously loaded from HDF5, yield
        them directly. Otherwise, load them from the HDF5 file and yield them
        one by one. For this, the HDF5 file is expected to have a specific
        structure of groups and datasets corresponding to Protein attributes, 
        which are differentially loaded. 

        Yields
        ------
        Protein
            Protein objects in the collection.
        '''
        # Already loaded
        if self.items:
            for protein in tqdm(self.items, desc='Iterating over Protein collection'):
                yield protein

        # Load from HDF5
        logger.info(f'Loading ProteinCollection from HDF5 file...')
        with h5py.File(self.file_path, 'r') as file:
            # Load attributes
            num_items = file.attrs['num_items']

            # Load datasets
            seq_array = file['seq'][:self.limit] if self._should_load('seq', file) else None
            uniprot_array = file['uniprot'][:self.limit] if self._should_load('uniprot', file) else None
            taxon_array = file['taxon'][:self.limit] if self._should_load('taxon', file) else None
            section_array = file['section'][:self.limit] if self._should_load('section', file) else None
            primary_accession_array = file['primary_accession'][:self.limit] if self._should_load('primary_accession', file) else None
            secondary_accessions_array = file['secondary_accessions'][:self.limit] if self._should_load('secondary_accessions', file) else None
            closest_arabidopsis_array = file['closest_arabidopsis'][:self.limit] if self._should_load('closest_arabidopsis', file) else None
            interpro_domains_array = file['interpro_domains'][:self.limit] if self._should_load('interpro_domains', file) else None
            esm2_embeddings_array = {key: file['esm2_embeddings'][key][:self.limit] for key in file['esm2_embeddings']} if self._should_load('esm2_embeddings', file) else None
            
            for key, value in file.attrs.items():
                print(f"  {key}: {value}")
                
        # Utils dict for esm2 embeddings
        model2dim = {
            '8M': 320,
            '35M': 480,
            '150M': 640,
            '650M': 1280,
            '3B': 2560,
            '15B': 5120
        }
        
        # Yield Protein objects
        for idx in tqdm(range(num_items), desc='Loading Protein collection from HDF5'):
            kwargs = {}
            if seq_array is not None:
                kwargs['seq'] = seq_array[idx].decode('utf-8')
            if uniprot_array is not None:
                kwargs['uniprot'] = uniprot_array[idx].decode('utf-8')
            if taxon_array is not None:
                kwargs['taxon'] = int(taxon_array[idx])
            if section_array is not None:
                kwargs['section'] = section_array[idx].decode('utf-8')
            if primary_accession_array is not None:
                kwargs['primary_accession'] = primary_accession_array[idx].decode('utf-8')
            if secondary_accessions_array is not None:
                kwargs['secondary_accessions'] = [string.decode('utf-8') for string in secondary_accessions_array[idx]]
            if interpro_domains_array is not None:
                kwargs['interpro_domains'] = json.loads(interpro_domains_array[idx].decode('utf-8'))
            if closest_arabidopsis_array is not None:
                kwargs['closest_arabidopsis'] = closest_arabidopsis_array[idx].decode('utf-8')
            if esm2_embeddings_array is not None:
                kwargs['esm2_embeddings'] = {
                    model: esm2_embeddings_array[model][idx].reshape(-1, model2dim[model])
                    for model in esm2_embeddings_array.keys()
                }

            protein = Protein(**kwargs)
            self.items.append(protein)
            yield protein

    def to_hdf5(self) -> None:
        '''
        Save the ProteinCollection to an HDF5 file. Based on the attributes
        present in the Protein objects, and their types, its contents are saved
        accordingly. Therefore, it requires string matching for each attribute.
        Note: existing data is frozen and not overwritten.
        '''
        # Create HDF5 file
        with h5py.File(self.file_path, 'a') as file:

            # Number of items as attribute
            file.attrs['num_items'] = len(self.items)

            # Iterate over attributes
            attributes = self.items[0].__dataclass_fields__.keys()
            for attr in attributes:
                # Skip existing datasets
                if attr in file.keys():
                    continue
                # Skip attributes with all default values
                all_default = all(getattr(protein, attr) in (None, {}, []) for protein in self.items)
                if all_default: continue
                # Save non-default attributes
                match attr:
                    # Handle string attributes
                    case 'seq' | 'uniprot' | 'section' | 'primary_accession' | 'closest_arabidopsis':
                        dtype = h5py.string_dtype(encoding='utf-8')
                        array = np.array([getattr(protein, attr) for protein in self.items], dtype=dtype)
                        file.create_dataset(attr, data=array, compression="gzip", compression_opts=9, chunks=True)
                    # Handle integer attributes
                    case 'taxon':
                        dtype = np.int32
                        array = np.array([getattr(protein, attr) for protein in self.items], dtype=dtype)
                        file.create_dataset(attr, data=array, compression="gzip", compression_opts=9, chunks=True)
                    # Handle list of strings attributes
                    case 'secondary_accessions':
                        lst = [protein.secondary_accessions for protein in self.items]
                        dtype = h5py.vlen_dtype(h5py.string_dtype(encoding='utf-8'))
                        lst = [np.array(sublist, dtype=dtype) for sublist in lst]
                        array = np.array(lst, dtype=dtype)
                        file.create_dataset(attr, data=array, compression="gzip", compression_opts=9, chunks=True)
                    # Handle dictionary attributes
                    case 'interpro_domains':
                        lst = [json.dumps(protein.interpro_domains) for protein in self.items]
                        dtype = h5py.string_dtype(encoding='utf-8')
                        array = np.array(lst, dtype=dtype)
                        file.create_dataset(attr, data=array, compression="gzip", compression_opts=9, chunks=True)
                    # Handle special attributes
                    case 'esm2_embeddings': 
                        for model in self.items[0].esm2_embeddings.keys():
                            group = file.create_group('esm2_embeddings')
                            embeddings = [item.esm2_embeddings[model] for item in self.items]
                            embeddings = [
                                (
                                    embedding.flatten()
                                    if embedding is not None 
                                    else np.array([], dtype=np.float32)
                                )
                                for embedding in embeddings
                            ] # vlen supports 1D only                            
                            dtype = h5py.vlen_dtype(np.dtype('float32'))
                            data_array = np.array(embeddings, dtype=dtype)
                            group.create_dataset(model, data=data_array, compression="gzip", compression_opts=9, chunks=True)

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
    collection = ProteinCollection(
        file_path = path.DATA / 'm_proteins.h5',
        limit = 1_000,
        datasets = ['seq']
        )
    prots = [p for p in collection]
    print(prots[0])
