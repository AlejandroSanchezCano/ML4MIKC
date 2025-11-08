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
from dataclasses import dataclass, field

@dataclass(slots=True)
class Protein:
    seq: str
    uniprot: str = None
    taxon: int = None
    section: str = None
    primary_accession: str = None
    secondary_accessions: list[str] = field(default_factory=list)
    interpro_domains: dict[str, tuple[int, int]] = field(default_factory=dict)
    esm2_embeddings: dict[str, list[float]] = field(default_factory=dict)
    closest_arabidopsis: str = None