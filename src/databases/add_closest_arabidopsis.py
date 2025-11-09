"""
===============================================================================
Title:      Add orthologs protein from Arabidopsis
Outline:    ESM2 embeddings cluster by orthology, so we can visualize it by 
            finding related proteins from Arabidopsis thaliana using MMseqs2.
            Some proteins were not initially found, so the sensitivity as well
            as the max number of accepted hits is increased.
            Also, the bits are normalized by the alignment length.
Docs:       https://matplotlib.org/stable/users/explain/customizing.html
Author:     Alejandro Sánchez Cano
Date:       15/10/2025
Time:       1 min
===============================================================================
"""

# Built-in modules
import subprocess

# Third-party modules
import pandas as pd

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.collection import ProteinCollection
logger.info('Importing modules completed')

# Fasta Arabidopsis proteins
df = pd.read_csv(path.DATA / "arabidopsis_mikc_prots.tsv", sep="\t")
with open("ara.fasta", "w") as f:
    for index, row in df.iterrows():
        f.write(f">{row['Protein']}\n{row['TOTAL']}\n")

# Fasta DB proteins
file_path = path.DATA / 'mikc_proteins.h5'
collection = ProteinCollection(file_path=file_path)
proteins = [protein for protein in collection]
collection.fasta(out_path = "db.fasta", header_attributes = ['uniprot'])

# MMseqs2 search
cmd = 'mmseqs easy-search ara.fasta db.fasta aln.m8 tmp -s 7.5 --max-accept 10000 --max-seqs 100000'
subprocess.run(cmd, shell=True, check=True)

# Read alignment
df = pd.read_csv("aln.m8", sep="\t")
df.columns = ["query", "target", "pident", "alnlen", "mismatches", "gapopens", "qstart", "qend", "tstart", "tend", "evalue", "bits"]
best_hits = df.loc[df.groupby("target")["bits"].idxmax()]
if len(best_hits) < len(collection.items):
    logger.error(f"Warning: Only {len(best_hits)} hits found for {len(collection.items)} proteins.")

# Remove files
subprocess.run("rm -r ara.fasta db.fasta aln.m8 tmp", shell=True, check=True)

# Assign closest Arabidopsis protein to each DB protein
best_hits_dict = best_hits.set_index("target")["query"].to_dict()
for protein in proteins:
    ara_bioID = best_hits_dict.get(protein.uniprot, 'No hit')
    protein.closest_arabidopsis = ara_bioID

# Save updated proteins
collection = ProteinCollection(
    file_path=path.DATA / 'mikc_proteins.h5', 
    items=proteins
    )
collection.to_hdf5()


