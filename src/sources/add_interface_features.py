"""
===============================================================================
Title:      Add interface features
Outline:    Computes different interface features for the PPI structures in the
            PPI database using CCP4 (SC and PISA) and custom methods. It uses 
            the --array option to parallelize the computation across multiple
            jobs (i.e. CPUs), because parallization inside the same CPU is not
            possible due to OOM errors.
Author:     Alejandro Sánchez Cano
Date:       01/07/2025
Time:       3h excluding sc
            60h including sc
            2h 30min with array job parallelization (40 PPIs/min)
            The speed is limited to the number of concurrent jobs:
            - rome partition: 128 CPU cores
            - genoa partition: 192 CPU cores
===============================================================================
"""

# Built-in modules
import os
import tempfile
import subprocess

# Custom modules
from src.misc import path
from structure import Structure
from src.entities.ppi import PPI
from src.misc.logger import logger

def process(idx, struc):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save structure
        pdb_path = f'{tmp_dir}/{idx}.pdb'
        with open(pdb_path, 'w') as f:
            f.write(struc)

        # Run CCP4
        cmd = f'python ccp4.py --pdb {pdb_path}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=path.TOOLS / 'CCP4')
        if result.returncode != 0:
            raise Exception(f"Error processing {pdb_path}: {result.stderr}")
        output = result.stdout.split('\n')
        sc = eval(output[0])
        pisa = eval(output[1])

    # Features from Structure
    structure = Structure.from_string(struc)
    interface_residues = len({res for residues in structure.interface_contacts().values() for res in residues.keys() | set().union(*residues.values())})
    res_type = {type: len(residues)/interface_residues for type, residues in structure.interface_residue_types().items()}
    pDockQ = structure.pDockQ()
    contact_pairs = sum([len(neighbors) for _, residues in structure.interface_contacts().items() for _, neighbors in residues.items()])
    
    # Save PPI object
    ppi.interface_features = {
        'n_interface_residues': interface_residues,
        'polar_fraction': res_type['polar'],
        'hydrophobic_fraction': res_type['hydrophobic'],
        'charged_fraction': res_type['charged'],
        'contact_pairs': contact_pairs,
        'shape_complementary': sc,
        'n_hydrogen_bonds': pisa['hbonds'],
        'n_salt_bridges': pisa['salt_bridges'],
        'int_solv_energy': pisa['int_solv_en'],
        'interface_area': pisa['intf_area'],
        'pDockQ': pDockQ
    }
    ppi.pickle()

if __name__ == '__main__':
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    logger.info(f"Processing task ID: {task_id}")
    for idx, ppi in enumerate(PPI.iterate()):
        if idx != task_id:
            continue
        process(task_id, ppi.esmfold['structure'])
        logger.info(ppi.interface_features)
        break