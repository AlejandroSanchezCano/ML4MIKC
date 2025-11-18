# Built-in modules
import pprint
from pathlib import Path
from dataclasses import dataclass, field

# Custom modules
from src.entities.protein import Protein

@dataclass(slots=True)
class PPI:
    p1: Protein = None
    p2: Protein = None
    interaction: list[int|str] = field(default_factory=list)
    origin: list[str] = field(default_factory=list)

if __name__ == "__main__":
    p1 = Protein(seq="MK", taxon=9606, uniprot="P12345")
    p2 = Protein(seq="ACDE", taxon=9606, uniprot="P54321")
    p3 = Protein(seq="GGG", taxon=10090, uniprot="Q67890")

    # Store proteins
    file_path = Path('here_proteins.h5')
    from src.entities.collection import ProteinCollection
    protein_collection = ProteinCollection(file_path=file_path, items=[p1, p2, p3])
    pprint.pprint(protein_collection)
    protein_collection.to_hdf5()
    # Store PPIs
    ppi1 = PPI(p1=p1, p2=p2, interaction=[1, 0], origin=["BioGRID", "Paper"])
    ppi2 = PPI(p1=p2, p2=p2, interaction=[1], origin=["IntAct"])
    ppi3 = PPI(p1=p1, p2=p3, interaction=[0], origin=["STRING"])
    ppi4 = PPI(p1=p2, p2=p3, interaction=[1, 1], origin=["DIP", "MINT"])
    from src.entities.collection import PPICollection
    file_path = 'here.h5'
    ppi_collection = PPICollection(
        file_path=file_path, 
        protein_path='here_proteins.h5',
        items=[ppi1, ppi2, ppi3, ppi4],
        )
    pprint.pprint(ppi_collection)
    ppi_collection.to_hdf5()
    
    # Store
    loaded_ppi_collection = PPICollection(
        file_path=file_path,
        protein_path='here_proteins.h5',
        #items=[ppi1, ppi2, ppi3, ppi4],
        )
    for ppi in loaded_ppi_collection:
        pprint.pprint(ppi)