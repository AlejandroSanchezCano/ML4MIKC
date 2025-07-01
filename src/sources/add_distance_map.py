"""
===============================================================================
Title:      Add distance maps
Outline:    Compute the distance maps for all PPI structures in the PPI
            database using PConPy.
Author:     Alejandro Sánchez Cano
Date:       30/06/2025
Time:       
===============================================================================
"""

# Built-in modules
import tempfile

# Custom modules
from pconpy import PConPy
from src.entities.ppi import PPI
from src.entities.map import Map

MEASURES = [
    'CA',
    #'CB',
    #'cmass',
    #'sccmass',
    #'minvdw',
]

with tempfile.TemporaryDirectory() as tmp_dir:
    for idx, ppi in enumerate(PPI.iterate()):
        # Save strcuture to a temporary file
        with open(f'{tmp_dir}/{idx}.pdb', 'w') as f:
            f.write(ppi.structure)
        # Iterate over measures and distances
        for measure in MEASURES:
            # Initialize attribute
            if not hasattr(ppi, 'distance_map'):
                ppi.distance_map = {}
            # Compute distance map
            pyconpy = PConPy(
                map_type='dmap',
                pdb=f'{tmp_dir}/{idx}.pdb',
                measure='CA',
            )
            dmap = pyconpy.compute_matrix()
            # Save distance map
            ppi.distance_map[measure] = Map(dmap)
            ppi.pickle()