"""
===============================================================================
Title:      Path module
Outline:    Important path variables for the project to avoid hardcoding them
            in each individual file.
Author:     Alejandro Sánchez Cano
Date:       2024-10-02
===============================================================================
"""

# Built-in modules
from pathlib import Path

# Path to the root of the project
ROOT = Path('/home/asanchez/ML4MIKC')

# First-order directories
DATA = ROOT / 'data'
SRC = ROOT / 'src'

# High-order directories in DATA
INTERACTORS = DATA / 'Interactors'
NETWORKS = DATA / 'Networks'
LITERATUREMINING = DATA / 'LiteratureMining'
SCORING = DATA / 'Scoring'
PROTEIN = DATA / 'Protein'
PPI = DATA / 'PPI'

# Databases
DATABASES = DATA / 'Databases'
BIOGRID = DATABASES / 'BioGRID'
INTACT = DATABASES / 'IntAct'

# Chonky directories
CHONKY = Path('/home/asanchez/chonky')
TOOLS = CHONKY / 'tools'
BACKUP = CHONKY / 'backup'

# TOOL directories
ESMFOLD = TOOLS / 'ESMFold'