# Built-in modules
import warnings
warnings.filterwarnings('ignore', module='numba')

# Third-party modules
import umap
import umap.plot
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom modules
from src.misc import path
#from src.misc.style import *
from src.misc.logger import logger
from src.entities.collection import ProteinCollection
logger.info('Importing modules completed')

# Gather embeddings
file_path = path.DATA / 'mikc_proteins.h5'
collection = ProteinCollection(file_path=file_path)
proteins = [p for p in collection if p.esm2_embeddings['650M'].size]
residue_embeddings = [protein.esm2_embeddings['650M'] for protein in proteins]
sequence_embeddings = np.array([
    emb.mean(axis=0) 
    for emb in tqdm(residue_embeddings, desc='Calculating sequence embeddings')
])

# Color labels
closest_arabidopsis = np.array([p.closest_arabidopsis for p in proteins])
taxon = np.array([p.taxon for p in proteins])

# UMAP reduction
reducer = umap.UMAP(n_components=2, random_state=42)
mapper = reducer.fit(sequence_embeddings)

# Plot
umap.plot.points(mapper)
plt.savefig('./umap_projection.png')
plt.clf()

# Plot by closest Arabidopsis protein
umap.plot.points(mapper, labels=closest_arabidopsis)
plt.savefig('./umap_projection_closest.png')
plt.clf()

# Plot by taxon
umap.plot.points(mapper, labels=taxon)
plt.savefig('./umap_projection_taxon.png')
plt.clf()


## UMAP reduction
#reducer = umap.UMAP(n_components=2, random_state=42)
#embedding_2d = reducer.fit_transform(sequence_embeddings)
#
## Plotting
#plt.figure()
#plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=5, alpha=0.7)
#plt.title('UMAP Projection of Protein Embeddings')
#plt.xlabel('UMAP 1')
#plt.ylabel('UMAP 2')
#plt.savefig('./umap_projection.png')
#plt.clf()
#
## Plotting colored by closest Arabidopsis protein
#unique_colors = list(set(protein.closest_arabidopsis for protein in proteins))
#palette = sns.color_palette("hsv", len(unique_colors))
#color_map = {prot: palette[i] for i, prot in enumerate(unique_colors)}
#colors = [color_map[protein.closest_arabidopsis] for protein in proteins]
#plt.figure()
#plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=5, alpha=0.7, c=colors)
#plt.title('UMAP Projection of Protein Embeddings (Colored by Closest Arabidopsis Protein)')
#plt.xlabel('UMAP 1')
#plt.ylabel('UMAP 2')
#plt.savefig('./umap_projection_colored.png')
#plt.clf()
#
#umap.version



##TMAP
#import tmap as tm
#import numpy as np
#from faerun import Faerun
#import pandas as pd
#
## Example embedding matrix (20000 x 1280)
#embeddings = means
#
## Example categorical string labels
#labels = closest
#int_labels, uniques = pd.factorize(labels)
#legend_labels = [(int_label, unique) for int_label, unique in zip(int_labels, uniques)]
#
## TMAP configuration
#CFG = tm.LayoutConfiguration()
#CFG.node_size = 1 / 50
#
## MinHash for binary vectors
#dims = embeddings.shape[1]
#enc = tm.Minhash(dims)
#lf = tm.LSHForest(dims, 128)
#
## Convert embeddings to binary vectors for MinHash
#tmp = []
#for vec in embeddings:
#    avg = vec.mean()
#    tmp.append(tm.VectorUchar([1 if x >= avg else 0 for x in vec]))
#
## Batch add to LSHForest
#lf.batch_add(enc.batch_from_binary_array(tmp))
#lf.index()
#
## Generate TMAP layout
#x, y, s, t, _ = tm.layout_from_lsh_forest(lf, CFG)
#
## Faerun visualization
#faerun = Faerun(view="front", clear_color="#111111", coords=False)
#faerun.add_scatter(
#    "Embeddings",
#    {
#        "x": x,
#        "y": y,
#        "c": int_labels,
#    },
#    colormap="tab10",
#    shader="smoothCircle",
#    point_scale=2.5,
#    max_point_size=10,
#    has_legend=True,
#    categorical=True,  # categorical for string labels
#    legend_labels=legend_labels,
#)
#faerun.add_tree("Tree", {"from": s, "to": t}, point_helper="Embeddings", color="#666666")
#faerun.plot("embedding_tmap_strings")
#