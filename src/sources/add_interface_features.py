"""
===============================================================================
Title:      Add interface features
Outline:    Computes different interface features for the PPI structures in the
            PPI database using CCP4 (SC and PISA) and custom methods.
Author:     Alejandro Sánchez Cano
Date:       01/07/2025
Time:       
===============================================================================
"""

# Built-in modules
import tempfile
import subprocess

# Custom modules
from structure import Structure
from src.entities.ppi import PPI

with tempfile.TemporaryDirectory() as tmp_dir:
    for idx, ppi in enumerate(PPI.iterate()):
        # Save structure
        with open(f'{tmp_dir}/{idx}.pdb', 'w') as f:
            f.write(ppi.structure)
    
        # Run CCP4
        cmd = f'python ccp4.py --pdb {tmp_dir}/{idx}.pdb'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/home/asanchez/tools/CCP4')
        if result.returncode != 0:
            raise Exception(f"Error processing {tmp_dir}/{idx}.pdb: {result.stderr}")
        
        # Read output
        output = result.stdout.split('\n')
        sc = float(output[0])
        pisa = eval(output[1])
        
        # From Structure
        structure = Structure(f'{tmp_dir}/{idx}.pdb')
        interface_residues = len({res for residues in structure.interface_contacts().values() for res in residues.keys() | set().union(*residues.values())})
        res_type = {type:len(residues)/interface_residues for type, residues in structure.interface_residue_types().items()}
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