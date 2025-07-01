"""
===============================================================================
Title:      Add ESMFold embeddings
Outline:    Use ESM2 to compute per-sequence embeddings for the proteins in the
            dataset and store them as a new attribute in the OODB Protein. The
            embeddings per PPI can be also obtained by concatenating the
            sequences but it will be omitted as we saw a detrimental effect on 
            model performance.
            Per-residue embeddings are not stored, as they can be obtained from
            the per-sequence embeddings. All ESM2 models are used, except
            '15B', which is too big and leads to a disk quota error when 
            attempting download it.
Author:     Alejandro Sánchez Cano
Date:       01/07/2025
Time:       3 min if models are already downloaded
===============================================================================
"""

# Custom modules
from esm2 import ESM2
from src.entities.protein import Protein

# TODO: add perplexity, attention maps and contact maps to PPIs

models = [
    '8M', 
    '35M', 
    '150M', 
    '650M',
    '3B',
    #'15B' Too big
    ]

for model in models:
    esm2 = ESM2(model)
    for protein in Protein.iterate():
        if not hasattr(protein, 'esm2_embeddings'):
            protein.esm2_embeddings = {}
        seq = protein.seq
        data = [('Protein', seq)]
        esm2.prepare_data(data)
        esm2.run_model()
        r, s = esm2.extract_representations()
        protein.esm2_embeddings[model] = s
        protein.pickle()