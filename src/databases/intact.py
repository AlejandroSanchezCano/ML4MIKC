"""
===============================================================================
Title:      IntAct
Outline:    IntAct class to download the data of any version of IntAct, reduce 
            it to only plant interactors, search for all MADS interactions, 
            filter to only MADS vs. MADS interactions, and standarize the data 
            frame to a common format that makes it compatible to the Network 
            class.
Docs:       https://www.ebi.ac.uk/intact/home
Author:     Alejandro Sánchez Cano
Date:       17/10/2025
===============================================================================
"""

# Built-in modules
import re
import subprocess
from io import StringIO

# Third-party modules
import pandas as pd
from multitax import NcbiTx

# Custom modules
from src.misc import path
from src.misc.logger import logger

class IntAct:

    def __init__(self, version: str):
        self.version = version

    def download_files(self) -> None:
        '''
        Downloads 'intact.txt' file from the specified IntAct version.
        '''

        # Set up output directory
        output_dir = path.INTACT / self.version
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download file
        url_file = f'https://ftp.ebi.ac.uk/pub/databases/intact/{self.version}/psimitab/intact.zip'
        wget = f'wget {url_file} -P {output_dir} -q'
        subprocess.run(wget, shell = True)

        # Unzip file and remove compressed and negatives file
        unzip = f'unzip -qq {output_dir}/intact.zip -d {output_dir}'
        subprocess.run(unzip, shell = True)
        rm = f'rm {output_dir}/intact.zip'
        subprocess.run(rm, shell = True)
        rm = f'rm {output_dir}/intact_negative.txt'
        subprocess.run(rm, shell = True)

        # Logging
        logger.info(f'IntAct {self.version} "intact.txt" file downloaded')
    
    def reduce_to_plants(self) -> None:
        '''
        Reduces IntAct 'intact.txt' file to only plant interactors to
        reduce computational burden when searching for UniProt IDs with
        grep in the whole file.
        '''

        # Set up file paths
        all_filepath = path.INTACT / self.version / 'intact.txt'
        plant_filepath = path.INTACT / self.version / 'plants.txt'

        # Read IntAct 'intact.txt' file as pandas DataFrame
        df = pd.read_csv(all_filepath, sep = '\t', dtype=str)

        # Filter interactors to only plants
        ncbi_tx = NcbiTx()
        is_plant = lambda x: ncbi_tx.parent_rank(re.split(r'[:\(]', x)[1], 'kingdom') == '33090'
        df_plants = df[df['Taxid interactor A'].apply(is_plant)]
             
        # Save filtered file
        df_plants.to_csv(plant_filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'IntAct {self.version} "intact.txt" file ({len(df)} PPIs) reduced to only plant interactions ({len(df_plants)} PPIs)')
    
    def mads_vs_all(self, mads_uniprots: set[str]) -> None:
        '''
        Searches for MADS interactors in the IntAct 'plants' file and
        retrieves their interactions in IntAct.

        Parameters
        ----------
        uniprots : set[str]
            List of UniProt IDs of MADS-box proteins.
        '''
        # Read IntAct 'plants' file
        plant_filepath = path.INTACT / self.version / f'plants.txt'
        df_plants = pd.read_csv(plant_filepath, sep = '\t', dtype=str)

        # Helper function
        def has_match(cell: str) -> bool:
            '''
            Helper function to check whether a cell contains any UniProt ID
            from the given set. Some parsing is needed to get to the UniProt ID
            For example: intact:EBI-301649|uniprotkb:Q96288-1

            Parameters
            ----------
            cell : str
                Cell content

            Returns
            -------
            bool
                Whether the cell contains any UniProt ID from the set
            '''

            if not cell:
                return False
            ids = [
                elem.split(':')[1].split('-')[0]
                if elem.startswith('uniprotkb:')
                else ''
                for elem in cell.split('|')
                ]
            return any(i in mads_uniprots for i in ids)

        # Filter
        mask = (
            df_plants["#ID(s) interactor A"].apply(has_match) |
            df_plants["ID(s) interactor B"].apply(has_match) |
            df_plants["Alt. ID(s) interactor A"].apply(has_match) |
            df_plants["Alt. ID(s) interactor B"].apply(has_match)
        )
        mads_vs_all = df_plants[mask]
        
        # Save DataFrame
        filepath = path.INTACT / self.version / 'MADS_vs_ALL.tsv'
        mads_vs_all.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. all PPIs in IntAct {self.version} "plants" file -> dim{mads_vs_all.shape}')
    
    def mads_vs_mads(self, mads_uniprots: set[str]) -> None:
        '''
        Filters MADS vs. MADS interactions from the MADS vs. ALL 
        interactions by checking whether any of the UniProt IDs are in 
        the MIKC list from InterPro.

        Parameters
        ----------
        uniprots : set[str]
            List of UniProt IDs of MADS-box proteins.
        '''
        # Load MADS_vs_ALL DataFrame
        filepath = path.INTACT / self.version / 'MADS_vs_ALL.tsv'
        mads_vs_all = pd.read_csv(filepath, sep = '\t')

        # Filter MADS vs MADS interactions
        is_there_mikc = lambda x: x.split('-')[0].split(':')[1] in mads_uniprots
        mads_vs_mads_A = mads_vs_all['#ID(s) interactor A'].apply(is_there_mikc)
        mads_vs_mads_B = mads_vs_all['ID(s) interactor B'].apply(is_there_mikc)
        mads_vs_mads = mads_vs_all[mads_vs_mads_A & mads_vs_mads_B]

        # Save DataFrame
        filepath = path.INTACT / self.version / 'MADS_vs_MADS.tsv'
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. MADS PPIs in IntAct {self.version} "plants" file -> dim{mads_vs_mads.shape}')

    def ppi_uniprot_accessions(self) -> list[tuple[str, str]]:
        '''
        Retrieves all UniProt accessions of the PPIs in the MADS vs. MADS 
        interactions file.

        Returns
        -------
        list[tuple[str, str]]
            List of tuples with the UniProt accessions of the interactors
            A and B.
        '''
        # Load MADS_vs_MADS DataFrame
        filepath = path.INTACT / self.version / 'MADS_vs_MADS.tsv'
        mads_vs_mads = pd.read_csv(filepath, sep = '\t')

        # Retrieve UniProt accessions
        ppi_accessions = []
        for _, row in mads_vs_mads.iterrows():
            uniprot_A = row['#ID(s) interactor A'].split(':')[1]
            uniprot_B = row['ID(s) interactor B'].split(':')[1]
            ppi_accessions.append((uniprot_A, uniprot_B))
        # Logging
        logger.info(f'Retrieved {len(ppi_accessions)} UniProt pairs accessions from MADS_vs_MADS file')
        
        return ppi_accessions

if __name__ == '__main__':
    # Gather UniProt IDs of MADS-box proteins
    from src.misc import path
    from src.entities.collection import ProteinCollection
    file_path = path.DATA / 'mikc_proteins.h5'
    collection = ProteinCollection(file_path=file_path, datasets=['uniprot'])
    mads_uniprots = set([protein.uniprot for protein in collection])
    
    # IntAct
    intact = IntAct('2025-08-08') 
    #intact.download_files()
    #intact.reduce_to_plants()
    #intact.mads_vs_all(mads_uniprots)
    intact.mads_vs_mads(mads_uniprots)
    ppis = intact.ppi_uniprot_accessions()
    print(ppis)