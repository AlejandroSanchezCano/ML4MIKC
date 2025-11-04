"""
===============================================================================
Title:      Protein
Outline:    Protein dataclass that represents a protein entity.
            Attributes
            ----------
            - seq: Protein sequence (or file path to pickled Protein object).
            - uniprot: UniProt ID.
            - taxon: Taxon ID (e.g. 9606).
            - section: 'TrEMBL' or 'Swiss-Prot'.
            - structure: AlphaFold structure in PDB format.
            - primary_accession: Primary UniProt accession (e.g. 'A0A022PSB5')
            - secondary_accessions: List of secondary UniProt accessions.
            - interpro_domains: InterPro sources mapped to their start and end
                location in the protein sequence, e.g. {'IPR002100':[(0,20), 
                (30,50)], 'IPR003000':[(60,80)]}
            - esm2_embeddings: ESM2 per-residue embeddings mapped by model
                name, e.g. {'650M': [...], '3B': [...]}
Author:     Alejandro Sánchez Cano
Date:       02/11/2025
===============================================================================
"""

# Built-in modules
import pprint
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field

# Custom modules
from src.misc import utils

@dataclass(slots=True)
class Protein:
    # Compulsory attribute
    seq: str | Path

    # Optional attributes
    uniprot: str = None
    taxon: int = None
    section: str = None
    structure: str = None
    primary_accession: str = None
    secondary_accessions: list[str] = field(default_factory=list)
    interpro_domains: dict[str, tuple[int, int]] = field(default_factory=dict)
    esm2_embeddings: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        '''
        Intantiate from file path with hashed sequence as name.
        '''
        if isinstance(self.seq, Path) or '/' in self.seq:
            file_path = self.seq
            content = utils.unpickle(file_path)
            for key, value in content.items():
                setattr(self, key, value)

    def pickle(self, dir: str = '.') -> None:
        '''
        Serializes the __dict__ object with pickle module and saves it to a 
        file named after the MD5 hash of the protein sequence, the unique 
        element of proteins.

        Parameters
        ----------
        dir : str, optional
            Directory to save the pickle file, by default the current directory.
        '''
        hashed_seq = hashlib.md5(self.seq.encode()).hexdigest()
        filename = f'{hashed_seq}.pkl'
        save_path = Path(dir) / filename
        utils.pickle(asdict(self), save_path)


if __name__ == "__main__":
    p1 = Protein(seq = '/home/asanchez/chonky/data/MIKC_proteins/0a0e442c05d2df598c01f13ea6121d59.pkl')
    pprint.pprint(p1)