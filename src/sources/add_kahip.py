"""
===============================================================================
Title:      Add KaHIP partitions
Outline:    In order to avoid data leakege from training set into the test set
            due to sequence similarity, we need to partition the PPI dataset
            in such a way that training and test sets are as dissimilar as 
            possible in terms of sequence similarity. For that, we use MMseqs2
            to measure the all-vs-all pairwaise similarity and use the
            (normalized) bitscore to construct a similarity graph. Then, we
            use KaHIP (Karlsruhe High Quality Partitioning) to partition the
            graph into two sets of roughly equal size such that some
            objective function is minimized such as number of edges between the
            blocks. We define three datasets:
            - INTRA0: both proteins in the PPI are from the same partition (0)
            - INTRA1: both proteins in the PPI are from the same partition (1)
            - INTER: proteins in the PPI are from different partitions
            This methodolody is described in the following paper:
            "Cracking the black box of deep sequence-based protein-protein
            interaction prediction" only that they used SIMAP2 bitscores and
            reduced redundancy with CD-HIT.

Docs:       https://github.com/KaHIP/KaHIP
            https://ar5iv.labs.arxiv.org/html/1311.1714
            https://pmc.ncbi.nlm.nih.gov/articles/PMC10939362/
Author:     Alejandro Sánchez Cano
Date:       26/06/2025
Time:       5 min
===============================================================================
"""

# Custom modules
import tempfile
import subprocess
from typing import Literal
from collections import defaultdict

# Third-party modules
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# Custom modules
from src.entities.ppi import PPI
from src.misc.logger import logger
from src.entities.protein import Protein

# Run MMseqs2
seqs = [prot.seq for prot in Protein.iterate()]
with tempfile.TemporaryDirectory() as temp_dir:
    # Create FASTA file
    fasta_file = f"{temp_dir}/seqs.fasta"
    with open(fasta_file, 'w') as f:
        for i, seq in enumerate(seqs):
            f.write(f">{i}\n{seq}\n")
    # Run MMseqs2 easy-search
    alignment_file = f"{temp_dir}/alignment.tsv"
    temp_file = f"{temp_dir}/mmseqs2_temp"
    cmd = f'mmseqs easy-search {fasta_file} {fasta_file} {alignment_file} {temp_file} -s 7.5 --max-seqs 1000000'
    subprocess.run(cmd, shell=True, check=True)
    # Read the alignment file and normalize bitscores
    df = pd.read_csv(f"{temp_dir}/alignment.tsv", sep="\t", header=None)
    df.columns = ['query','target','fident','alnlen','mismatch','gapopen','qstart','qend','tstart','tend','evalue','bits']
    df['normalized_bits'] = df['bits'] / df['alnlen']

# Create graph
def create_graph(df: pd.DataFrame, weighted_by: Literal['bits', 'normalized_bits']) -> dict: 
    '''
    Create a graph from the MMseqs2 search alignment results using the proteins
    as nodes and the (normalized) bitscore as edge weight. Normally each u-v 
    pair is calculated twice, once with u as query and v as target, and vice 
    versa, so the edge between u and v is parallel (in most cases, sometimes
    the alignment is not reciprocal and God knows why). We will use the average 
    of the two weigths.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the MMseqs2 search alignment results.

    weighted_by : Literal['bits', 'normalized_bits']
        Whether to use the raw bitscore or the normalized bitscore as edge 
        weight.

    Returns
    -------
    dict
        Dictionary with the graph, where keys are tuples of (u, v) and values 
        are lists of weights for the edges between u and v.
    '''
    # Create graph from DataFrame
    graph = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        u = int(row['query'])
        v = int(row['target'])
        w = float(row[weighted_by])
        u_v = tuple(sorted((u, v)))
        if u != v:
            graph[u_v].append(w)
    
    # Graph statistics
    diffs = []
    for u_v, weights in graph.items():
        if len(weights) == 1:
            logger.warning(f"Edge {u_v[0]}-{u_v[1]} is not parallel, using single weight")
        else:
            diffs.append(abs(np.diff(weights)[0]))
    logger.info(f'Parallel edges have differences: min={np.min(diffs):.2f}, max={np.max(diffs):.2f}, mean={np.mean(diffs):.2f}, std={np.std(diffs):.2f}')
    nodes = list(sorted(df['query'].unique()))
    n_nodes = len(nodes)
    n_edges = n_nodes * (n_nodes - 1) // 2
    logger.info(f"Graph created with {n_nodes} nodes and {len(graph)} edges ({n_edges/len(graph)*100:.2f}% of possible edges)")

    # Average weights for parallel edges
    for u_v, weights in graph.items():
        multiplier = 1 if weighted_by == 'bits' else 100
        graph[u_v] = int(np.mean(weights) * multiplier)

    return graph

# Format graph for KaHIP
def format_graph(graph: dict) -> None:
    '''
    Format the graph as required by KaHIP:
    - First lines describes the number of nodes and edges and wether they have
    weights or not.
    - The ith line contains: neighbor_vertex_id_1, weight_1, 
    neighbor_vertex_id_2, weight_2, ... for the i-1th node.
    - Edges must be >= 0
    - Edges must not be parallel
    - Graph must be undirected
    - No self-loops allowed

    Parameters
    ----------
    graph : dict
        Dictionary with the graph, where keys are tuples of (u, v) and values 
        are lists of weights for the edges between u and v.
    '''
    nodes = list(sorted(set(u for u, v in graph.keys()).union(v for u, v in graph.keys())))
    n_nodes = len(nodes)
    n_edges = n_nodes * (n_nodes - 1) // 2
    mode = 1
    with open("ppi_graph.txt", "w") as f:
        f.write(f"{n_nodes} {n_edges} {mode}\n")
        for node_i in nodes:
            text = []
            for node_j in nodes:
                if node_i == node_j:
                    continue
                u_v = tuple(sorted((node_i, node_j)))
                weight = graph[u_v]
                text += [f"{node_j + 1} {weight}"]
            f.write(f"{' '.join(text)}\n")

# Run kaffpa (one of KaHIP's partitioning algorithm)
def run_kahip(k: int = 2, preconfiguration: str = 'strong') -> list[int]:
    '''
    Run KaHIP to partition the graph into k partitions.

    Parameters
    ----------
    k : int, optional
        Number of partitions, by default 2.
    
    preconfiguration : str, optional
        Preconfiguration to use, by default 'strong'.

    Returns
    -------
    list[int]
        List of partition ids for each node.
    '''
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = f'kaffpa ppi_graph.txt --k {k}  --preconfiguration={preconfiguration} --output_filename {temp_dir}/temp_file.txt'
        subprocess.run(cmd, shell=True, check=True)
        with open(f"{temp_dir}/temp_file.txt", "r") as f:
            partitions = map(int, f.read().splitlines())
        cmd = f'rm ppi_graph.txt'
        subprocess.run(cmd, shell=True, check=True)
    logger.info(f"KaHIP partitioned the graph into {k} partitions")
    
    return list(partitions)

def ppi_blocks(partitions: list[int]) -> tuple[list[PPI], list[PPI], list[PPI]]:
    '''
    Create the PPI blocks INTRA0, INTRA1, INTER based on the partitions.

    Parameters
    ----------
    partitions : list[int]
        List of partition IDS: [0, 1, 1, 0, 1, ...] where the ith element
        corresponds to the partition where the ith protein in the Protein
        collection belongs to.

    Returns
    -------
    tuple[list[PPI], list[PPI], list[PPI]]
        INTRA0, INTRA1, INTER PPI blocks.
    '''
    intra0 = []
    intra1 = []
    inter = []
    seqs_to_partitions = {prot.seq: partition for prot, partition in zip(Protein.iterate(), partitions)}
    for ppi in PPI.iterate():
        partition_p1 = seqs_to_partitions[ppi.p1.seq]
        partition_p2 = seqs_to_partitions[ppi.p2.seq]
        if partition_p1 == partition_p2:
            if partition_p1 == 1:
                intra1.append(ppi)
            else:
                intra0.append(ppi)
        else:
            inter.append(ppi)
    
    # Logging
    logger.info(f"INTRA0: {len(intra0)} PPIs")
    logger.info(f"INTRA1: {len(intra1)} PPIs")
    logger.info(f"INTER: {len(inter)} PPIs")

    return intra0, intra1, inter

# Raw bitscores
graph = create_graph(df, weighted_by='bits')
format_graph(graph)
partitions = run_kahip(k=2, preconfiguration='strong')
bits_blocks = ppi_blocks(partitions)

# Normalize bitscores
graph = create_graph(df, weighted_by='normalized_bits')
format_graph(graph)
partitions = run_kahip(k=2, preconfiguration='strong')
norm_blocks = ppi_blocks(partitions)

# Compare raw vs normalized bitscores
seqs_bits = (set(), set(), set())
seqs_norm = (set(), set(), set())
for idx, (bit_block, norm_block) in enumerate(zip(bits_blocks, norm_blocks)):
    for ppi in bit_block:
        seqs = '='.join([ppi.p1.seq, ppi.p2.seq])
        seqs_bits[idx].add(seqs)
    for ppi in norm_block:
        seqs = '='.join([ppi.p1.seq, ppi.p2.seq])
        seqs_norm[idx].add(seqs)
intra0_common = seqs_bits[0].intersection(seqs_norm[0])
intra1_common = seqs_bits[1].intersection(seqs_norm[1])
inter_common = seqs_bits[2].intersection(seqs_norm[2])
logger.info(f"INTRA0 raw vs normalized bitscores: {len(intra0_common)} common PPIs")
logger.info(f"This is {len(intra0_common) / len(seqs_bits[0]) * 100:.2f}% of the INTRA0 PPIs with raw bitscores")
logger.info(f"This is {len(intra0_common) / len(seqs_norm[0]) * 100:.2f}% of the INTRA0 PPIs with normalized bitscores")
logger.info(f"INTRA1 raw vs normalized bitscores: {len(intra1_common)} common PPIs")
logger.info(f"This is {len(intra1_common) / len(seqs_bits[1]) * 100:.2f}% of the INTRA1 PPIs with raw bitscores")
logger.info(f"This is {len(intra1_common) / len(seqs_norm[1]) * 100:.2f}% of the INTRA1 PPIs with normalized bitscores")
logger.info(f"INTER raw vs normalized bitscores: {len(inter_common)} common PPIs")
logger.info(f"This is {len(inter_common) / len(seqs_bits[2]) * 100:.2f}% of the INTER PPIs with raw bitscores")
logger.info(f"This is {len(inter_common) / len(seqs_norm[2]) * 100:.2f}% of the INTER PPIs with normalized bitscores")

# Add partitions to Proteins
for idx, protein in enumerate(Protein.iterate()):
    protein.partition = partitions[idx]
    protein.pickle()    

# Add partitions to PPIs
BITSCORE = 'normalized'
blocks = norm_blocks if BITSCORE == 'normalized' else bits_blocks
intra0, intra1, inter = blocks
for ppi in intra0:
    ppi.partition = 'INTRA0'
    ppi.pickle()
for ppi in intra1:
    ppi.partition = 'INTRA1'
    ppi.pickle()
for ppi in inter:
    ppi.partition = 'INTER'
    ppi.pickle()
