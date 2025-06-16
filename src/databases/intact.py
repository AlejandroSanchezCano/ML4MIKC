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
from src.entities.interactor import Interactor

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

    def _grep(self, uniprot_id: str) -> pd.DataFrame:
        '''
        Searches for a specific UniProt ID in the IntAct 'plants' file and
        retrieves its interactions in IntAct.

        Parameters
        ----------
        uniprot_id : str
            UniProt ID to search for.

        Returns
        -------
        pd.DataFrame
            Interaction table of the given UniProt ID.
        '''

        # Set up file path
        file_path = path.INTACT / self.version / 'plants.txt'

        # First line as columns
        cat = f'head -1 {file_path}'
        result = subprocess.run(cat, shell = True, text = True, stdout = subprocess.PIPE)
        columns = result.stdout.lstrip('#').rstrip().split('\t')

        # Search for UniProt ID in IntAct 'ALL' file
        grep = f'grep {uniprot_id} {file_path}'
        result = subprocess.run(grep, shell = True, text = True, stdout = subprocess.PIPE)
        
        # Convert result to DataFrame
        if result.stdout == '':
            return pd.DataFrame(columns = columns)
        else:
            result = StringIO(result.stdout)
            return pd.read_csv(result, sep = '\t', names = columns)
    
    def mads_vs_all(self) -> None:
        '''
        Searches for all MADS interactors in the IntAct 'plants.txt' 
        file and retrieves their interactions in IntAct.
        '''
        
        # Initialize DataFrame
        mads_vs_all = pd.DataFrame()

        # Iterate over MADS interactors
        for interactor in Interactor.iterate():
            # Search for UniProt ID in IntAct 'plants' file
            df = self._grep(interactor.uniprot_id)
            # Append to DataFrame if not empty
            if not df.empty:
                mads_vs_all = pd.concat([mads_vs_all, df], ignore_index = True)
        
        # Save DataFrame
        filepath = path.NETWORKS / f'IntAct_{self.version}_MADS_vs_ALL.tsv'
        mads_vs_all.to_csv(filepath, sep = '\t', index = False)   

        # Logging
        logger.info(f'MADS vs. all PPIs in IntAct {self.version} "plants" file -> dim{mads_vs_all.shape}')

    def mads_vs_mads(self) -> None:
        '''
        Filters MADS vs. MADS interactions from the MADS vs. ALL 
        interactions by checking whether any of the UniProt IDs are in 
        the MIKC list from InterPro.
        '''
        # Load MADS_vs_ALL DataFrame
        filepath = path.NETWORKS / f'IntAct_{self.version}_MADS_vs_ALL.tsv'
        mads_vs_all = pd.read_csv(filepath, sep = '\t')

        # MADS UniProt IDs
        mads = set([interactor.uniprot_id for interactor in Interactor.iterate()])

        # Filter MADS vs MADS interactions
        is_there_mikc = lambda x: x.split('-')[0].split(':')[1] in mads
        mads_vs_mads_A = mads_vs_all['ID(s) interactor A'].apply(is_there_mikc)
        mads_vs_mads_B = mads_vs_all['ID(s) interactor B'].apply(is_there_mikc)
        mads_vs_mads = mads_vs_all[mads_vs_mads_A & mads_vs_mads_B]

        # Save DataFrame
        filepath = path.NETWORKS / f'IntAct_{self.version}_MADS_vs_MADS.tsv'
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. MADS PPIs in IntAct {self.version} "plants" file -> dim{mads_vs_mads.shape}')

    def standarize(self) -> None:
        '''
        Format data frame interaction network to accommodate standard naming
        convention of columns to homogenize the data frames from different 
        databases. It contains:
        - A: UniProt ID of interactor A
        - B: UniProt ID of interactor B
        - A=B: Concatenation of A and B sorted alphabetically
        - Species_A: Species ID of interactor A
        - Species_B: Species ID of interactor B
        - Seq_A: Sequence of interactor A
        - Seq_B: Sequence of interactor B
        - Seq: Sequence of interactor A : sequence of interactor B
        '''
        # Load MADS_vs_MADS DataFrame
        filepath = path.NETWORKS / f'IntAct_{self.version}_MADS_vs_MADS.tsv'
        mads_vs_mads = pd.read_csv(filepath, sep = '\t')

        # Assign columns
        mads_vs_mads['A'] = mads_vs_mads['ID(s) interactor A'].apply(lambda x: x.split('-')[0].split(':')[1])
        mads_vs_mads['B'] = mads_vs_mads['ID(s) interactor B'].apply(lambda x: x.split('-')[0].split(':')[1])
        mads_vs_mads['A=B'] = mads_vs_mads[['A', 'B']].apply(lambda x: '='.join(sorted(x)), axis = 1)
        mads_vs_mads['Species_A'] = mads_vs_mads['Taxid interactor A'].apply(lambda x: re.split(r'[:\(]', x)[1])
        mads_vs_mads['Species_B'] = mads_vs_mads['Taxid interactor B'].apply(lambda x: re.split(r'[:\(]', x)[1])
        mads_vs_mads['Seq_A'] = mads_vs_mads['A'].apply(lambda x: Interactor(x).seq)
        mads_vs_mads['Seq_B'] = mads_vs_mads['B'].apply(lambda x: Interactor(x).seq)
        mads_vs_mads['Seq'] = mads_vs_mads['Seq_A'] + ':' + mads_vs_mads['Seq_B']
                                                                         
        # Remove duplicated columns
        mads_vs_mads = mads_vs_mads.drop_duplicates('A=B')
        mads_vs_mads = mads_vs_mads.drop_duplicates('Seq')

        # Save DataFrame
        filepath = path.NETWORKS / f'IntAct_{self.version}_MADS_vs_MADS_standarized.tsv'
        mads_vs_mads = mads_vs_mads[['A', 'B', 'A=B', 'Species_A', 'Species_B', 'Seq_A', 'Seq_B', 'Seq']]
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. MADS PPIs in IntAct {self.version} file standarized -> dim({mads_vs_mads.shape})')