# Third-party modules
import umap
import numpy as np
import matplotlib.pyplot as plt

# Custom modules
from src.misc import path
from src.misc.style import *
from src.entities.collection import CollectionParallel

# Gather embeddings
proteins = CollectionParallel(dir = path.MIKC_PROTS, class_type = 'Protein')
residue_embeddings = [protein.esm2_embeddings['650M'] for protein in proteins]
sequence_embeddings = np.array(residue_embeddings).mean(axis = 1)
print(sequence_embeddings.shape)

# UMAP reduction
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(sequence_embeddings)

# Plotting
plt.figure()
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=5, alpha=0.7)
plt.title('UMAP Projection of Protein Embeddings')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('./umap_projection.png')