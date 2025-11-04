"""
===============================================================================
Title:      Add ESMFold embeddings
Outline:    Use ESM2 to compute per-sequence embeddings for the proteins in the
            dataset and store them as a new attribute in the OODB Protein.
            Per-residue embeddings are not stored, as they can be obtained from
            the per-sequence embeddings. Model 650M is used as a compromise
            between memory requirements and performance.

            CAUTION: Model loading adds many GB to the ~/.cache/torch
            directory, where the ESM2 models are downloaded. Either we make 
            sure to have enough space, or we clean them up after running this 
            script.
Author:     Alejandro Sánchez Cano
Date:       01/07/2025
Time:       22 min (models already downloaded)
===============================================================================
"""

# Third-party modules
from tqdm import tqdm

# Custom modules
from esm2 import ESM2
from src.misc import path
from src.entities.collection import ProteinCollection

# Choose ESM2 models
models = [
    #'8M', 
    #'35M', 
    #'150M', 
    '650M',
    #'3B',
    #'15B' Too big
    ]

import torch

# Gather Protein objects
collection = ProteinCollection(dir = path.MIKC_PROTS)
proteins = [protein for protein in collection]

# Compute and store embeddings
for model in models:
    esm2 = ESM2(model)
    for protein in tqdm(proteins, desc = f'Computing {model} embeddings'):
        if model in protein.esm2_embeddings: continue
        data = [('Protein', protein.seq)]
        try:
            esm2.prepare_data(data)
        except ValueError as e:
            logger.error(f'Error computing {model} embeddings for {protein.accession} due to sequence length ({len(protein.seq)}): {e}')
        esm2.run_model()
        r, s = esm2.extract_representations()
        protein.esm2_embeddings[model] = r
        protein.pickle(dir = path.MIKC_PROTS)