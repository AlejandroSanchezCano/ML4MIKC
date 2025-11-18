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
    seq: str = None
    name: str = None
    uniprot: str = None
    taxon: int = None
    section: str = None
    primary_accession: str = None
    secondary_accessions: list[str] = field(default_factory=list)
    interpro_domains: dict[str, tuple[int, int]] = field(default_factory=dict)
    esm2_embeddings: dict[str, list[float]] = field(default_factory=dict)
    closest_arabidopsis: str = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Protein):
            return False
        return self.seq == other.seq and self.taxon == other.taxon

    def __hash__(self) -> int:
        hash_input = f"{self.seq}_{self.taxon}".encode('utf-8')
        return int(hashlib.md5(hash_input).hexdigest(), 16)

if __name__ == "__main__":
    p1 = Protein(seq="MK", taxon=9606, uniprot="P12345")
    p2 = Protein(seq="MK", taxon=9606, uniprot="Q67890")
    p3 = Protein(seq="ACDE", taxon=9606, uniprot="P54321")
    prots = {p1, p2, p3}
    pprint.pprint(prots)