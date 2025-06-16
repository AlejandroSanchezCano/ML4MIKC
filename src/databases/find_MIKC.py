"""
===============================================================================
Title:      Find MIKC proteins
Outline:    Uses the InterProDatabase class to retrieve the UniProt IDs of
            proteins containing both MADS-box (IPR002100) and K-box domains
            (IPR002487), which will be our MIKC proteins. The UniProt IDs are
            saved to a file for further processing.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
Time:       1h 40min
===============================================================================
"""

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.databases.interpro_database import InterProDatabase    

# MADS proteins -> 1h 20min
mads = InterProDatabase('IPR002100')
mads.get_metadata()
m_uniprot_ids = mads.get_uniprot()

# K-box proteins -> 20min
kbox = InterProDatabase('IPR002487')
kbox.get_metadata()
k_uniprot_ids = kbox.get_uniprot()

# Union of UniProt IDs
mikc_uniprot_ids = sorted(list(set(m_uniprot_ids) & set(k_uniprot_ids)))

# Logging
logger.info(f'{len(mikc_uniprot_ids)} MIKC UniProt IDs retrieved')
logger.info(f'{len(mikc_uniprot_ids)/len(m_uniprot_ids)*100:.2f}% of MADS-box proteins')
logger.info(f'{len(mikc_uniprot_ids)/len(k_uniprot_ids)*100:.2f}% of K-box proteins')

# Save MIKC UniProt IDs
output_file = path.DATA / 'm_and_k_uniprot_ids.txt'
with open(output_file, 'w') as f:
    for uniprot_id in mikc_uniprot_ids:
        f.write(f'{uniprot_id}\n')