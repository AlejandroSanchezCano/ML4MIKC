"""
===============================================================================
Title:      Add ESMFold structures
Outline:    Compute the structures of the PPIs in the dataset using ESMFold and 
            store them as a new attribute in the OODB PPI. As opposed to 
            previous versions, even the C domain is included, which increases 
            computational time twofold.
Author:     Alejandro Sánchez Cano
Date:       30/06/2025
Time:       14 sec / protein * 5724 proteins ~= 22h 30 min
===============================================================================
"""

# Built-in modules
import time
import subprocess

# Third-party modules
import numpy as np

# Custom modules
from src.misc import path
from structure import Structure
from src.entities.ppi import PPI
from src.entities.map import Map
from src.entities.plddt import PLDDT


# Create temporary directory
tmp_dir = path.CHONKY / 'temp'
tmp_dir.mkdir(exist_ok=True)

# Create fasta file
for idx, ppi in enumerate(PPI.iterate()):
    with open(f"{tmp_dir}/{idx}.fasta", "w") as f:
        f.write(f">A\n{ppi.p1.seq}\n")
        f.write(f">B\n{ppi.p2.seq}\n")

# Run ESMFold
cmd = f'sbatch {path.ESMFOLD}/run.sh --fasta {tmp_dir} --out_dir {tmp_dir}'
subprocess.run(cmd, shell=True, check=True)

## Add computed objects to PPIs
#for idx, ppi in enumerate(PPI.iterate()):
#    # Wait for ESMFold to finish
#    while not (tmp_dir / f"{idx}.pdb").exists():
#        time.sleep(10)
#    # Instantiate attribute
#    if not hasattr(ppi, 'esmfold'):
#        ppi.esmfold = {}
#    # Add structure
#    structure = Structure(f"{tmp_dir}/{idx}.pdb")
#    ppi.esmfold['structure'] = structure.to_string()
#    # Add pLDDT
#    ppi.esmfold['plddt'] = PLDDT(np.load(f"{tmp_dir}/{idx}.plddt.npy"))
#    # Add PAE
#    ppi.esmfold['pae'] = Map(np.load(f"{tmp_dir}/{idx}.pae.npy"))
#    # Add contact maps
#    ppi.esmfold['contact_maps'] = {}
#    for threshold in [6, 8, 10, 12]:
#        ppi.esmfold['contact_maps'][threshold] = Map(np.load(
#            f"{tmp_dir}/{idx}.contacts{threshold}.npy"
#        ))
#    # Save PPI object
#    ppi.pickle()
#
## Clean up temporary directory
#tmp_dir.rmdir()