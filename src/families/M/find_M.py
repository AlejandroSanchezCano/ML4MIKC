"""
===============================================================================
Title:      Find M proteins
Outline:    Uses the InterPro class to retrieve the UniProt IDs of proteins
            containing only the MADS-box (IPR002100) domain, which will be our 
            M proteins. In InterPro, this domain architecture is identified
            with the IDA hash 9b1d1537f57a287fce1f0a861665d2876b831672
            The UniProt IDs are saved to a file for further processing.
Author:     Alejandro Sánchez Cano
Date:       10/11/2025
Time:       1h 40min (3 min if cached)
===============================================================================
"""

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.families.interpro import InterProAccession, InterProDomainArchitecture

# M domain architecture
ida = '9b1d1537f57a287fce1f0a861665d2876b831672'
interpro = InterProDomainArchitecture(ida=ida)
m_uniprot_ids = interpro.get_uniprot()

# Logging
logger.info(f'{len(m_uniprot_ids)} M UniProt IDs retrieved')

# Save M UniProt IDs
output_file = path.DATA / 'm_uniprot_ids.txt'
with open(output_file, 'w') as f:
    for uniprot_id in m_uniprot_ids:
        f.write(f'{uniprot_id}\n')