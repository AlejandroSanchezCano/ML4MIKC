"""
===============================================================================
Title:      Add domains
Outline:    Use InterProScan to annotate the domains of the proteins in the PPI
            database. The InterProScan API has a limit of 100 proteins per
            request, so the proteins are processed in batches.
Author:     Alejandro Sánchez Cano
Date:       01/07/2025
Time:       
===============================================================================
"""

# Built-in modules
import tempfile

# Custom modules
from src.misc.logger import logger
from interproscan import InterProScan
from src.entities.protein import Protein

# InterProScan API has a limit of 100 proteins per request
proteins_no_domains = [protein for protein in Protein.iterate() if protein.domains == {}]
logger.info(f"Proteins without domains: {len(proteins_no_domains)}")
seqs = [protein.seq for protein in proteins_no_domains]
batch_seqs = [seqs[i:i + 100] for i in range(0, len(seqs), 100)]
batchs_prots = [proteins_no_domains[i:i + 100] for i in range(0, len(proteins_no_domains), 100)]

# Run InterProScan
for seqs, prots in zip(batch_seqs, batchs_prots):
    with tempfile.TemporaryDirectory() as tmp_dir:
        protein2seq = {f'protein_{i}': seq for i, seq in enumerate(seqs)}
        iprscan = InterProScan(protein2seq)
        iprscan.run(
            email = 'a.sanchezcano@uva.nl',
            sequence = f'{tmp_dir}/fasta.fasta',
            stype = 'p',
            goterms = False,
            pathways = False,
            outfile = f'{tmp_dir}/output',
            verbose = True,
            title = 'protein_batch'
            )
        protein2domains = iprscan.parse_tsv()
        for idx, protein in enumerate(prots):
            protein.domains = protein2domains[f'protein_{idx}']
            protein.pickle()
