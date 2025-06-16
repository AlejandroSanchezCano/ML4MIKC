"""
===============================================================================
Title:      Download data from PPI databases
Outline:    Downloads the data from the IntAct, BioGRID, and PlaPPISite
            databases, reduces it to plant interactors if neccessary, 
            filters it to only MADS-MADS interactions, and standardizes it
            to a common format used by the Network class. 
            It takes ~20 min for BioGRID, ~1h 30min for IntAct, and a lot for
            PlaPPISite.
Docs:       https://downloads.thebiogrid.org/BioGRID
            https://ftp.ebi.ac.uk/pub/databases/intact/
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
Time:       - BioGrid: 20 min
            - PlaPPISite: 20 min
            - IntAct: 1h 30min
===============================================================================
"""

# Custom modules
from src.databases.intact import IntAct
from src.databases.biogrid import BioGRID
from src.databases.plappisite import PlaPPISite

# BioGRID
biogrid = BioGRID('4.4.246')
biogrid.download_files()
biogrid.reduce_to_plants()
biogrid.mads_vs_all()
biogrid.mads_vs_mads()
biogrid.standarize()

# PlaPPISite
plappisite = PlaPPISite()
plappisite.mads_vs_all()
plappisite.mads_vs_mads()
plappisite.standarize()

# IntAct
intact = IntAct('2025-03-28') 
intact.download_files()
intact.reduce_to_plants()
intact.mads_vs_all()
intact.mads_vs_mads()
intact.standarize()